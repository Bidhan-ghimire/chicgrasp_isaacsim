# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configuration for the corrected UR10e and ChicGrasp gripper."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[6]
ROBOT_USD_PATH = str(
    PROJECT_ROOT
    / "assets"
    / "grasp_contract"
    / "ur10e_chicgrasp_serrated_tip_collision_package"
    / "ur10e_chicgrasp_serrated_tip_collision.usda"
)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
JAW_JOINT_NAMES = ["PrismaticJoint1", "PrismaticJoint2", "PrismaticJoint3", "PrismaticJoint4"]
LEFT_JAW_JOINT_NAMES = JAW_JOINT_NAMES[:2]
RIGHT_JAW_JOINT_NAMES = JAW_JOINT_NAMES[2:]
# The real CAD stops at 0.0 m. This rigid-proxy wrapper adds 9.3 mm of
# outward travel per finger because the simulated chicken cannot deform.
JAW_OPEN_POSITION_M = 0.0093
JAW_CLOSED_POSITION_M = -0.0093
GRIPPER_BASE_BODY_NAME = "gripper_base_link"

FIXED_CHICKEN_ARM_RESET_JOINT_POSITIONS_RAD = {
    "shoulder_pan_joint": -0.05085492506623268,
    "shoulder_lift_joint": -1.5838452577590942,
    "elbow_joint": 1.5867072343826294,
    "wrist_1_joint": -1.56606125831604,
    "wrist_2_joint": -1.4226365089416504,
    "wrist_3_joint": 1.5203697681427002,
}


CHICGRASP_UR10E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Face the arm toward the chicken workspace on the robot's +X side.
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -0.5 * math.pi,
            "elbow_joint": 0.5 * math.pi,
            "wrist_1_joint": -0.5 * math.pi,
            "wrist_2_joint": -0.5 * math.pi,
            "wrist_3_joint": 0.0,
            "PrismaticJoint[1-4]": JAW_OPEN_POSITION_M,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            effort_limit_sim=330.0,
            stiffness=1320.0,
            damping=72.6636085,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=150.0,
            stiffness=600.0,
            damping=34.64101615,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=56.0,
            stiffness=216.0,
            damping=29.39387691,
        ),
        "jaws": ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint[1-4]"],
            effort_limit_sim=35.0,
            velocity_limit_sim=0.02,
            stiffness=2500.0,
            damping=120.0,
        ),
    },
)


# Approved only for the fixed-chicken Diffusion Policy vertical slice on 2026-08-20.
# Source: gate1_pre_aligned_reset_pose_validation_report.json, SHA-256
# e1a10afb0d0b7fab272df6326204fdce537011c33dfc24bd23f70985c700436d.
# The candidate converged kinematically, remained inside runtime hard and soft
# joint limits, and kept all four jaws open. Candidate generation and visual
# inspection used no physics-driven motion. Human visual inspection found no
# obvious robot self, table, gripper-chicken, or jaw-leg intersection. The
# installed Isaac and PhysX APIs do not support a reliable no-step, filter-aware,
# exact collision-overlap query, so this approval does not claim query-proven
# collision clearance. The dynamic-chicken task has no equivalent approval.
CHICGRASP_UR10E_FIXED_CHICKEN_CFG = CHICGRASP_UR10E_CFG.replace(
    init_state=CHICGRASP_UR10E_CFG.init_state.replace(
        joint_pos={
            **CHICGRASP_UR10E_CFG.init_state.joint_pos,
            **FIXED_CHICKEN_ARM_RESET_JOINT_POSITIONS_RAD,
        },
    ),
)
