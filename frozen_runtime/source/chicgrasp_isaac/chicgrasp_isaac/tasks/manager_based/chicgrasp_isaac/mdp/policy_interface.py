# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac bridge for the fixed ChicGrasp policy observation and action contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from chicgrasp_policy.interface import (
    ACTION_POSITION_FRAME,
    ACTION_POSITION_UNITS,
    ARM_JOINT_NAMES,
    FIXED_TCP_QUATERNION_BASE_WXYZ,
    InvalidPolicyFrame,
    JAW_ACTION_THRESHOLD,
    JAW_CLOSED_BIT,
    JAW_OBSERVATION_THRESHOLD_M,
    JAW_OPEN_BIT,
    JAW_STATE_NAMES,
    POLICY_ACTION_DIM,
    POLICY_TCP_LOCAL_POSITION_M,
    POLICY_TCP_LOCAL_QUATERNION_WXYZ,
    POLICY_TCP_PARENT_BODY,
    TCP_X_RANGE_M,
    TCP_Y_RANGE_M,
    TCP_Z_RANGE_M,
)
from chicgrasp_policy.jaw_state_diagnostics import attach_jaw_state_diagnostic

from ..robot_cfg import JAW_CLOSED_POSITION_M, JAW_JOINT_NAMES, JAW_OPEN_POSITION_M

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def _require_ordered_joint_selector(
    asset_cfg: SceneEntityCfg,
    expected_joint_names: Sequence[str],
    observation_name: str,
) -> None:
    """Reject selectors that could reorder or broaden a policy observation."""
    selected_names = asset_cfg.joint_names
    if selected_names is None:
        raise ValueError(f"{observation_name} requires an explicit joint-name selector")
    if isinstance(selected_names, str):
        selected_names = [selected_names]
    if tuple(selected_names) != tuple(expected_joint_names):
        raise ValueError(
            f"{observation_name} requires joints {tuple(expected_joint_names)!r}; "
            f"received {tuple(selected_names)!r}"
        )
    if not asset_cfg.preserve_order:
        raise ValueError(f"{observation_name} requires preserve_order=True")


def arm_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the six policy arm positions in the approved UR10e joint order."""
    _require_ordered_joint_selector(asset_cfg, ARM_JOINT_NAMES, "arm_joint_pos")
    asset: Articulation = env.scene[asset_cfg.name]
    positions = asset.data.joint_pos[:, asset_cfg.joint_ids]
    expected_shape = (env.num_envs, len(ARM_JOINT_NAMES))
    if tuple(positions.shape) != expected_shape:
        raise ValueError(f"arm_joint_pos must have shape {expected_shape}; received {tuple(positions.shape)}")
    if positions.dtype != torch.float32:
        raise TypeError(f"arm_joint_pos must be torch.float32; received {positions.dtype}")
    if not bool(torch.all(torch.isfinite(positions)).item()):
        raise InvalidPolicyFrame("arm_joint_pos contains a non-finite value")
    return positions.clone()


def jaw_state_dimensions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sizing only for the qualified CPU constructor query; never a policy frame."""
    _require_ordered_joint_selector(asset_cfg, JAW_JOINT_NAMES, "jaw_state")
    return torch.empty((env.num_envs, len(JAW_STATE_NAMES)), dtype=torch.uint8, device=env.device)


def jaw_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Copy existing accepted command intent, never the next policy action.

    ActionManager exists before ObservationManager. Its action term's existing
    reset initializes open commands; successful target issuance publishes accepted
    commands. A prepared future action cannot change this snapshot. No joint
    read, actuation or thresholding here.
    """
    _require_ordered_joint_selector(asset_cfg, JAW_JOINT_NAMES, "jaw_state")
    try:
        bits = env.action_manager.get_term("policy_action").commanded_jaw_state
    except (AttributeError, KeyError) as exc:
        raise InvalidPolicyFrame("accepted jaw command state unavailable") from exc
    expected_shape = (env.num_envs, len(JAW_STATE_NAMES))
    if not isinstance(bits, torch.Tensor) or tuple(bits.shape) != expected_shape:
        raise InvalidPolicyFrame(f"accepted jaw command state must have shape {expected_shape}")
    if bits.dtype != torch.uint8:
        raise InvalidPolicyFrame("accepted jaw command state must be uint8")
    if bool(torch.any((bits != JAW_CLOSED_BIT) & (bits != JAW_OPEN_BIT)).item()):
        raise InvalidPolicyFrame("accepted jaw command state must be binary")
    return bits.clone()




class FiveDimensionalPolicyAction(ActionTerm):
    """Map ``[x, y, z, gL, gR]`` to fixed-pose DLS and paired jaw targets.

    Cartesian positions are absolute robot-base metres. The requested position is
    retained only as diagnostic output and is never added to policy observations.
    """

    cfg: FiveDimensionalPolicyActionCfg
    _asset: Articulation

    def __init__(self, cfg: FiveDimensionalPolicyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._validate_configuration()

        arm_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=cfg.asset_name,
            joint_names=list(ARM_JOINT_NAMES),
            body_name=POLICY_TCP_PARENT_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=tuple(POLICY_TCP_LOCAL_POSITION_M),
                rot=tuple(POLICY_TCP_LOCAL_QUATERNION_WXYZ),
            ),
            scale=1.0,
            clip=None,
        )
        self._arm_action = DifferentialInverseKinematicsAction(arm_cfg, env)
        if tuple(self._arm_action._joint_names) != tuple(ARM_JOINT_NAMES):
            raise ValueError(
                "absolute DLS resolved an unexpected arm-joint order: "
                f"{tuple(self._arm_action._joint_names)!r}"
            )
        self._arm_joint_ids = self._arm_action._joint_ids

        self._jaw_joint_ids, resolved_jaw_names = self._asset.find_joints(
            cfg.jaw_joint_names,
            preserve_order=True,
        )
        if tuple(resolved_jaw_names) != tuple(JAW_JOINT_NAMES):
            raise ValueError(f"unexpected resolved jaw-joint order: {tuple(resolved_jaw_names)!r}")

        self._raw_actions = torch.zeros(
            (self.num_envs, POLICY_ACTION_DIM),
            dtype=torch.float64,
            device=self.device,
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._pose_actions = torch.zeros(
            (self.num_envs, 7),
            dtype=torch.float64,
            device=self.device,
        )
        self._jaw_bits = torch.full(
            (self.num_envs, 2),
            JAW_OPEN_BIT,
            dtype=torch.uint8,
            device=self.device,
        )
        # Observation metadata only. Pending process_actions is not target issue.
        self._issued_jaw_bits = self._jaw_bits.clone()
        self._jaw_position_targets = torch.full(
            (self.num_envs, len(JAW_JOINT_NAMES)),
            float(cfg.jaw_open_position_m),
            dtype=self._asset.data.joint_pos.dtype,
            device=self.device,
        )
        self._requested_tcp_position_m = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float64,
            device=self.device,
        )
        self._has_valid_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    @property
    def action_dim(self) -> int:
        """Public action dimension; orientation and four jaw targets stay internal."""
        return POLICY_ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Validated positions followed by thresholded ``[gL, gR]`` bits."""
        return self._processed_actions

    @property
    def requested_tcp_position_m(self) -> torch.Tensor:
        """Return a defensive copy of the requested robot-base TCP position."""
        return self._requested_tcp_position_m.clone()

    @property
    def requested_tcp_position_frame(self) -> str:
        return ACTION_POSITION_FRAME

    @property
    def requested_tcp_position_units(self) -> str:
        return ACTION_POSITION_UNITS

    @property
    def commanded_jaw_state(self) -> torch.Tensor:
        """Already accepted/reset-issued binary intent, independent of motion."""
        return self._issued_jaw_bits.clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        """Validate without clipping, then build the private pose and jaw commands."""
        try:
            validated_actions = self._validate_actions(actions)
            jaw_bits = (validated_actions[:, 3:5] >= JAW_ACTION_THRESHOLD).to(dtype=torch.uint8)

            fixed_quaternion = torch.as_tensor(
                FIXED_TCP_QUATERNION_BASE_WXYZ,
                dtype=validated_actions.dtype,
                device=self.device,
            ).expand(self.num_envs, -1)
            pose_actions = torch.cat((validated_actions[:, :3], fixed_quaternion), dim=-1)

            expanded_open_bits = jaw_bits.to(dtype=torch.bool).repeat_interleave(2, dim=-1)
            jaw_targets = torch.where(
                expanded_open_bits,
                torch.as_tensor(
                    self.cfg.jaw_open_position_m,
                    dtype=self._jaw_position_targets.dtype,
                    device=self.device,
                ),
                torch.as_tensor(
                    self.cfg.jaw_closed_position_m,
                    dtype=self._jaw_position_targets.dtype,
                    device=self.device,
                ),
            )

            # Delegate only after the complete public action has passed validation.
            self._arm_action.process_actions(pose_actions)
            self._raw_actions.copy_(validated_actions)
            self._processed_actions[:, :3].copy_(validated_actions[:, :3])
            self._processed_actions[:, 3:5].copy_(jaw_bits)
            self._pose_actions.copy_(pose_actions)
            self._jaw_bits.copy_(jaw_bits)
            self._jaw_position_targets.copy_(jaw_targets)
            self._requested_tcp_position_m.copy_(validated_actions[:, :3])
            self._has_valid_command.fill_(True)
        except Exception:
            # A caller may catch the validation error. Invalidate the old command
            # first so a later apply cannot execute stale pose or jaw targets.
            issued = self._issued_jaw_bits.clone()
            self.reset()
            # That fail-closed reset prepares open targets but has not issued
            # them. Keep observation intent until apply succeeds or real reset.
            self._issued_jaw_bits.copy_(issued)
            raise

    def apply_actions(self) -> None:
        """Apply the internal absolute-DLS command and four endpoint targets."""
        if bool(torch.any(self._has_valid_command).item()):
            self._arm_action.apply_actions()

        invalid_env_ids = torch.nonzero(~self._has_valid_command, as_tuple=False).flatten()
        if invalid_env_ids.numel() > 0:
            current_arm_positions = self._asset.data.joint_pos[invalid_env_ids][:, self._arm_joint_ids]
            self._asset.set_joint_position_target(
                current_arm_positions,
                joint_ids=self._arm_joint_ids,
                env_ids=invalid_env_ids,
            )

        self._asset.set_joint_position_target(
            self._jaw_position_targets,
            joint_ids=self._jaw_joint_ids,
        )
        self._issued_jaw_bits.copy_(self._jaw_bits)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear prior commands and restore deterministic open jaw state."""
        selected_envs: Sequence[int] | torch.Tensor | slice
        selected_envs = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_arm_action"):
            self._arm_action.reset(env_ids=selected_envs)
            self._arm_action.processed_actions[selected_envs] = 0.0
        if not hasattr(self, "_raw_actions"):
            return

        self._raw_actions[selected_envs] = 0.0
        self._raw_actions[selected_envs, 3:5] = float(JAW_OPEN_BIT)
        self._processed_actions[selected_envs] = 0.0
        self._processed_actions[selected_envs, 3:5] = float(JAW_OPEN_BIT)
        self._pose_actions[selected_envs] = 0.0
        self._pose_actions[selected_envs, 3:7] = torch.as_tensor(
            FIXED_TCP_QUATERNION_BASE_WXYZ,
            dtype=self._pose_actions.dtype,
            device=self.device,
        )
        self._jaw_bits[selected_envs] = JAW_OPEN_BIT
        self._issued_jaw_bits[selected_envs] = self._jaw_bits[selected_envs]
        self._jaw_position_targets[selected_envs] = float(self.cfg.jaw_open_position_m)
        self._requested_tcp_position_m[selected_envs] = 0.0
        self._has_valid_command[selected_envs] = False

    def _validate_configuration(self) -> None:
        if self.cfg.asset_name != "robot":
            raise ValueError("FiveDimensionalPolicyAction requires asset_name='robot'")
        if tuple(self.cfg.arm_joint_names) != tuple(ARM_JOINT_NAMES):
            raise ValueError("FiveDimensionalPolicyAction requires the exact six arm joints")
        if tuple(self.cfg.jaw_joint_names) != tuple(JAW_JOINT_NAMES):
            raise ValueError("FiveDimensionalPolicyAction requires the exact four ordered jaw joints")
        if self.cfg.body_name != POLICY_TCP_PARENT_BODY:
            raise ValueError(f"FiveDimensionalPolicyAction requires body_name={POLICY_TCP_PARENT_BODY!r}")
        if float(self.cfg.jaw_open_position_m) != JAW_OPEN_POSITION_M:
            raise ValueError("jaw_open_position_m differs from the existing actuator endpoint")
        if float(self.cfg.jaw_closed_position_m) != JAW_CLOSED_POSITION_M:
            raise ValueError("jaw_closed_position_m differs from the existing actuator endpoint")
        if self.cfg.clip is not None:
            raise ValueError("FiveDimensionalPolicyAction rejects out-of-range actions; clipping is prohibited")

    def _validate_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"policy action must be a torch.Tensor; received {type(actions)!r}")
        expected_shape = (self.num_envs, POLICY_ACTION_DIM)
        if actions.ndim != 2 or tuple(actions.shape) != expected_shape:
            raise ValueError(f"policy action must have shape {expected_shape}; received {tuple(actions.shape)}")
        if torch.is_complex(actions):
            raise TypeError("policy action must contain real values")

        # Validate in float64 before simulator conversion so an outside value cannot
        # enter the closed workspace through lower-precision rounding.
        validation_actions = actions.detach().to(device=self.device, dtype=torch.float64)
        if not bool(torch.all(torch.isfinite(validation_actions)).item()):
            raise ValueError("policy action contains NaN or infinity")

        for axis, axis_name, bounds in (
            (0, "x", TCP_X_RANGE_M),
            (1, "y", TCP_Y_RANGE_M),
            (2, "z", TCP_Z_RANGE_M),
        ):
            lower, upper = bounds
            outside = (validation_actions[:, axis] < lower) | (validation_actions[:, axis] > upper)
            if bool(torch.any(outside).item()):
                raise ValueError(
                    f"policy {axis_name} target is outside the inclusive [{lower}, {upper}] m workspace"
                )

        jaw_commands = validation_actions[:, 3:5]
        if bool(torch.any((jaw_commands < JAW_CLOSED_BIT) | (jaw_commands > JAW_OPEN_BIT)).item()):
            raise ValueError("raw gL and gR commands must be within the inclusive [0, 1] domain")
        return validation_actions.clone()


@configclass
class FiveDimensionalPolicyActionCfg(ActionTermCfg):
    """Configuration for the exact five-dimensional ChicGrasp action term."""

    class_type: type[ActionTerm] = FiveDimensionalPolicyAction

    arm_joint_names: list[str] = MISSING
    jaw_joint_names: list[str] = MISSING
    body_name: str = MISSING
    jaw_open_position_m: float = MISSING
    jaw_closed_position_m: float = MISSING
