# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-environment UR10e, dual-jaw gripper, and rigid-chicken scene."""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from chicgrasp_policy.stage0_visuals import validate_config

from . import mdp
from .mdp import policy_interface as policy_mdp
from .robot_cfg import (
    ARM_JOINT_NAMES,
    CHICGRASP_UR10E_CFG,
    CHICGRASP_UR10E_FIXED_CHICKEN_CFG,
    GRIPPER_BASE_BODY_NAME,
    JAW_CLOSED_POSITION_M,
    JAW_JOINT_NAMES,
    JAW_OPEN_POSITION_M,
)


PROJECT_ROOT = Path(__file__).resolve().parents[6]
POLICY_CAMERA_CALIBRATION_PATH = PROJECT_ROOT / "config" / "policy_camera_calibration.json"
GREY_STUDIO_USD_PATH = str(
    PROJECT_ROOT / "assets" / "lighting" / "grey_studio" / "light_rig_environment.usda"
)
_POLICY_CAMERA_CALIBRATION = validate_config(json.loads(
    POLICY_CAMERA_CALIBRATION_PATH.read_text(encoding="utf-8")
))
CHICKEN_USD_PATH = str(
    PROJECT_ROOT
    / "assets"
    / "grasp_contract"
    / "chicken_rigid_wide_torso_spread_legs_protected_shank_grasp_v2"
    / "asset"
    / "chicken_rigid_wide_torso_spread_legs_protected_shank_grasp_v2.usda"
)
WORK_SURFACE_HEIGHT = 0.75
TABLE_CENTER_X = 0.65
TABLE_SIZE = (0.60, 0.80, 0.04)
TABLE_TOP_CENTER_Z = WORK_SURFACE_HEIGHT - 0.5 * TABLE_SIZE[2]
ROBOT_PEDESTAL_RADIUS = 0.10
ROBOT_BASE_POS = (0.0, 0.0, WORK_SURFACE_HEIGHT)
JAW_CONTACT_FILTERS = ["{ENV_REGEX_NS}/Chicken", "{ENV_REGEX_NS}/Tabletop"]
# The cavity-up asset's lowest collision point is approximately -0.0651 m.
# Keep the rigid-body root unrotated and place that geometry on the tabletop.
CHICKEN_SUPPORT_HEIGHT = 0.0655
CHICKEN_INITIAL_POS = (TABLE_CENTER_X, 0.0, WORK_SURFACE_HEIGHT + CHICKEN_SUPPORT_HEIGHT)


def _policy_camera_cfg(camera_name: str) -> CameraCfg:
    """Build a fixed RGB camera from the shared, explicitly status-marked config."""
    calibration = _POLICY_CAMERA_CALIBRATION

    record = calibration["cameras"][camera_name]
    shared = calibration["shared_camera_settings"]
    if record["camera_prim_name"] != camera_name:
        raise ValueError(f"Policy camera record/name mismatch for {camera_name}")
    if not record["direct_child_of_environment_root"]:
        raise ValueError(f"{camera_name} must be a direct child of the environment root")
    if not record["camera_not_under_robot_or_chicken"]:
        raise ValueError(f"{camera_name} must remain outside Robot and Chicken")
    if not shared["rgb_only"] or shared["data_types"] != ["rgb"]:
        raise ValueError("Policy cameras must be RGB-only")

    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{record['camera_prim_name']}",
        update_period=0.0,
        height=shared["height"],
        width=shared["width"],
        data_types=list(shared["data_types"]),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=shared["focal_length"],
            focus_distance=shared["focus_distance"],
            horizontal_aperture=shared["horizontal_aperture"],
            vertical_aperture=shared["vertical_aperture"],
            horizontal_aperture_offset=shared["horizontal_aperture_offset"],
            vertical_aperture_offset=shared["vertical_aperture_offset"],
            f_stop=shared["f_stop"],
            projection_type=shared["projection_type"],
            clipping_range=tuple(shared["clipping_range"]),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(record["position_environment_m"]),
            rot=tuple(record["quaternion_environment_wxyz"]),
            # Gate 0B saved the raw Usd.Camera prim rotation. Declare its
            # native OpenGL convention so CameraCfg does not reinterpret it.
            convention="opengl",
        ),
    )


def _policy_light_cfg() -> AssetBaseCfg:
    """The existing scene DomeLight, shared by GUI and headless recorders."""
    light = _POLICY_CAMERA_CALIBRATION["scene_light"]
    properties = dict(light["properties"], color=tuple(light["properties"]["color"]))
    return AssetBaseCfg(
        prim_path=light["prim_path"],
        spawn=sim_utils.DomeLightCfg(**properties),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(light["position_parent_m"]), rot=tuple(light["quaternion_parent_wxyz"])),
    )


def _spawn_grey_studio(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Reference the exact export once, without overwriting a pre-existing rig."""
    if prim_path != "/Environment" or Path(cfg.usd_path).resolve() != Path(GREY_STUDIO_USD_PATH):
        raise ValueError("Grey Studio must reference the configured export at /Environment")
    if translation is not None and tuple(translation) != (0.0, 0.0, 0.0):
        raise ValueError("Grey Studio root translation must preserve the export")
    if orientation is not None and tuple(orientation) != (1.0, 0.0, 0.0, 0.0):
        raise ValueError("Grey Studio root orientation must preserve the export")
    existing = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    if existing.IsValid():
        references = existing.GetMetadata("references")
        items = [] if references is None else references.GetAppliedItems()
        if (len(items) == 1 and str(items[0].primPath) == ""
                and Path(items[0].assetPath).resolve() == Path(GREY_STUDIO_USD_PATH)):
            # Complete composed transforms/settings are checked before Stage0 capture.
            return existing
        raise ValueError(
            f"Grey Studio /Environment conflict: references={references!r}; "
            f"prim_stack={existing.GetPrimStack()!r}"
        )
    # The export's defaultPrim is Environment. None preserves its authored root
    # identity; the non-identity Grey_Studio child matrix remains in its layer.
    return sim_utils.spawn_from_usd(
        prim_path, cfg, translation=None, orientation=None, **kwargs
    )


def _spawn_single_dome_scene(prim_path, cfg, **kwargs):
    """Compose the owner rig, then isolate its dome before the first render."""
    from chicgrasp_policy.dome_recovery import apply_overrides
    prim = _spawn_grey_studio(prim_path, cfg, **kwargs)
    apply_overrides(sim_utils.get_current_stage())
    return prim


@configclass
class ChicgraspIsaacSceneCfg(InteractiveSceneCfg):
    """One robot and one chicken on a ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0)),
    )

    tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.78, 0.035, 0.02),
                roughness=0.45,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.50,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, 0.0, TABLE_TOP_CENTER_Z),
        ),
    )

    table_leg_front_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.045, 0.045, TABLE_TOP_CENTER_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.12, 0.13)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.91, 0.185, 0.5 * TABLE_TOP_CENTER_Z)),
    )
    table_leg_front_right = table_leg_front_left.replace(
        prim_path="{ENV_REGEX_NS}/TableLegFrontRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.91, -0.185, 0.5 * TABLE_TOP_CENTER_Z)),
    )
    table_leg_back_left = table_leg_front_left.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBackLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.39, 0.185, 0.5 * TABLE_TOP_CENTER_Z)),
    )
    table_leg_back_right = table_leg_front_left.replace(
        prim_path="{ENV_REGEX_NS}/TableLegBackRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.39, -0.185, 0.5 * TABLE_TOP_CENTER_Z)),
    )

    robot_pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotPedestal",
        spawn=sim_utils.CylinderCfg(
            radius=ROBOT_PEDESTAL_RADIUS,
            height=WORK_SURFACE_HEIGHT,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.16, 0.18, 0.20),
                metallic=0.65,
                roughness=0.30,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * WORK_SURFACE_HEIGHT)),
    )

    robot: ArticulationCfg = CHICGRASP_UR10E_FIXED_CHICKEN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=CHICGRASP_UR10E_FIXED_CHICKEN_CFG.init_state.replace(
            pos=ROBOT_BASE_POS
        ),
    )

    # Camera identity follows the fixed chicken orientation and the leg
    # primarily observed, never a robot/world/table/view-screen side.
    # PolicyCameraLeft is the fixed external view of the chicken's left-leg
    # grasp region. Its direct environment-root prim is not under Robot or Chicken.
    camera_left = _policy_camera_cfg("PolicyCameraLeft")

    # PolicyCameraRight is the fixed external view of the chicken's right-leg
    # grasp region. Keep this calibration record paired with that chicken leg.
    camera_right = _policy_camera_cfg("PolicyCameraRight")

    # Non-policy wrist instrumentation only; PolicyCfg never references it.
    # The published CAD contains a D435 visual at this mount, but its authored
    # mesh axis looks away from the two jaws in the top-down grasp orientation.
    # For the virtual sensor, aim from that mount toward the distal-leg surface
    # midpoint when the jaw-station plane is 150 mm above the chicken root, then
    # move 40 mm along that ray to clear the housing. This keeps the CAD camera
    # origin while centering both distal shanks during the guarded approach.
    # Quaternion order is WXYZ; world convention uses +X forward and +Z up.
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/GripperRig/gripper_base_link/wrist_camera",
        update_period=0.0,
        height=240,
        width=320,
        # Depth is retained as a calibration channel for converting selected
        # leg pixels into 3-D targets. The learned-policy observation remains
        # RGB-only, matching the paper's image input.
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.00163296, 0.04714322, -0.20186633),
            rot=(0.65549984, 0.25928047, -0.26517912, 0.65785533),
            convention="world",
        ),
        update_latest_camera_pose=True,
    )

    chicken = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Chicken",
        # Clearance calibration needs a stationary obstacle. Change this to a
        # dynamic rigid body only when the grasp-and-lift phase is implemented.
        spawn=sim_utils.UsdFileCfg(
            usd_path=CHICKEN_USD_PATH,
            # Preserve the prior experiment's authored 1.5 kg chicken mass;
            # the replacement package intentionally contains no mass opinion.
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CHICKEN_INITIAL_POS,
            # The 180-degree inversion is baked into the asset's Visuals and
            # Collisions children, not expressed as a scene/root rotation.
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    jaw_1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/GripperRig/jaw_1_link",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=JAW_CONTACT_FILTERS,
    )
    jaw_2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/GripperRig/jaw_2_link",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=JAW_CONTACT_FILTERS,
    )
    jaw_3_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/GripperRig/jaw_3_link",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=JAW_CONTACT_FILTERS,
    )
    jaw_4_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/GripperRig/jaw_4_link",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=JAW_CONTACT_FILTERS,
    )

    dome_light = _policy_light_cfg()

    # Shared scene construction, not a reset callback or a viewport lighting preset.
    grey_studio = AssetBaseCfg(
        prim_path="/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=GREY_STUDIO_USD_PATH, func=_spawn_single_dome_scene),
    )


@configclass
class ChicgraspDynamicChickenSceneCfg(ChicgraspIsaacSceneCfg):
    """Dynamic-chicken scene retaining the original shared robot reset pose."""

    robot: ArticulationCfg = CHICGRASP_UR10E_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=CHICGRASP_UR10E_CFG.init_state.replace(pos=ROBOT_BASE_POS),
    )


@configclass
class ActionsCfg:
    """One exact ``[x, y, z, gL, gR]`` policy action term."""

    policy_action = policy_mdp.FiveDimensionalPolicyActionCfg(
        asset_name="robot",
        arm_joint_names=ARM_JOINT_NAMES,
        jaw_joint_names=JAW_JOINT_NAMES,
        body_name=GRIPPER_BASE_BODY_NAME,
        jaw_open_position_m=JAW_OPEN_POSITION_M,
        jaw_closed_position_m=JAW_CLOSED_POSITION_M,
    )


@configclass
class ObservationsCfg:
    """Exact four-term learned-policy observation surface."""

    @configclass
    class PolicyCfg(ObsGroup):
        camera_left_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_left"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        camera_right_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_right"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        arm_joint_pos = ObsTerm(
            func=policy_mdp.arm_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ARM_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        jaw_state = ObsTerm(
            func=policy_mdp.jaw_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JAW_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ChicgraspIsaacEnvCfg(ManagerBasedRLEnvCfg):
    """Scene and control configuration used by the clearance script."""

    scene: ChicgraspIsaacSceneCfg = ChicgraspIsaacSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 60.0
        # The robot reaches from its base at X=0 toward the chicken at X=0.65.
        # Inspect from that -X approach side with a small -Y offset, close to a
        # plausible wrist-camera line of sight and low enough to expose the legs.
        self.viewer.eye = (0.12, -0.55, 1.28)
        self.viewer.lookat = (TABLE_CENTER_X, 0.0, 0.86)
        self.num_rerenders_on_reset = 3
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05


@configclass
class ChicgraspDynamicChickenEnvCfg(ChicgraspIsaacEnvCfg):
    """Closure/lift variant with a gravity-enabled dynamic chicken proxy."""

    scene: ChicgraspDynamicChickenSceneCfg = ChicgraspDynamicChickenSceneCfg(
        num_envs=1,
        env_spacing=2.5,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # A worst-case no-contact close-and-reopen sweep can exceed the
        # calibration task's 60 s horizon; never allow an automatic reset
        # during guarded cleanup.
        self.episode_length_s = 120.0
        # Keep the accepted USD and the verified calibration task unchanged.
        # This spawn-time override is the only kinematic-to-dynamic transition.
        self.scene.chicken.spawn.rigid_props = self.scene.chicken.spawn.rigid_props.replace(
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=0.25,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        )
