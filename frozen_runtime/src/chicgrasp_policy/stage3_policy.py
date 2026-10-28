"""Checkpoint-bound live Stage3 selection. Import is inert; no demo loader."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import interface as api, overfit_policy
from .attachment_capture import TARGETS
from .deployment_adapter import adapt_returned_actions_for_scheduler
from .overfit_data import sha256_file
from .stage1_demo import append, load, write

PROJECT = Path(__file__).resolve().parents[2]
BINDING = PROJECT / "docs/checkpoints/gate6_stage3/binding.json"
CHECKPOINT_SHA256 = "f1a954f9e0ea973f402accd271ce9e6742332fc853f5d6f0ace805c20585f54a"
SETUP_ACTIONS, LEARNED_ACTIONS, TOTAL_ACTIONS = 50, 150, 200


def binding():
    value = load(BINDING)
    if (value["checkpoint_files"]["checkpoint.pt"]["sha256"] != CHECKPOINT_SHA256
            or value["stage2_verdict"] != "GATE6_STAGE2_ONE_DEMO_OVERFIT_FAIL"
            or value["offline_metrics"]["cartesian_max_abs_error_m"] != 0.00627686083316803
            or value["model_configuration"] != overfit_policy.DEFAULT_POLICY_CONFIG
            or value["derived_view"]["bootstrap_original_observation_indices"] != [49, 50]):
        raise ValueError("stage3_binding_contract")
    return value


def verify_binding():
    """Authenticate small provenance files and exact model bytes, never demo arrays."""
    value = binding()
    for name, item in value["checkpoint_files"].items():
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError("stage3_bound_file_changed:" + name)
    for relative, key in (("config/policy_camera_calibration.json", "visual_config_sha256"),
            ("docs/grants/GATE6_STAGE0_VISUAL_APPROVAL_20261012.json", "visual_approval_sha256")):
        if sha256_file(PROJECT / relative) != value[key]:
            raise ValueError("stage3_visual_binding:" + relative)
    report = load(value["checkpoint_files"]["training_report.json"]["path"])
    if (report["checkpoint_sha256"] != CHECKPOINT_SHA256
            or report["evaluation"] != value["offline_metrics"]
            or report["status"] != value["stage2_verdict"]
            or report["source"] != value["source"]
            or not report["reload"]["reproduced"]):
        raise ValueError("stage3_report_binding")
    return value


def load_bound_policy(value, device="cuda:0"):
    """Use saved normalization verbatim. No accuracy PASS prerequisite or refit."""
    if device != "cuda:0":
        raise ValueError("stage3_GPU_only_no_CPU_fallback")
    policy, metadata = overfit_policy.load_checkpoint(value["checkpoint_files"]["checkpoint.pt"]["path"], device=device)
    expected = dict(optimizer_step=10000, weights_kind="EMA", source=value["source"],
        normalization=value["normalization"], training_configuration=value["training_configuration"],
        evaluation=value["offline_metrics"])
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise ValueError("stage3_checkpoint_metadata:" + key)
    for name, field in value["normalization"]["fields"].items():
        for key in ("scale", "offset"):
            if policy.normalizer[name].params_dict[key].detach().cpu().tolist() != field[key]:
                raise ValueError("stage3_saved_normalizer:" + name + ":" + key)
    return policy, metadata


def prepare_predictor():
    """Future child only. Resolve installed core packages before support append."""
    import torch
    import torchvision
    value = verify_binding()
    native = (torch.__file__, torchvision.__file__, np.__file__)
    support = Path(value["policy_support_path"]).resolve(strict=True)
    if str(support) not in sys.path:
        sys.path.append(str(support))
    if native != (torch.__file__, torchvision.__file__, np.__file__):
        raise RuntimeError("stage3_support_replaced_core_packages")
    overfit_policy.set_deterministic(42)
    policy, metadata = load_bound_policy(value)
    return make_predictor(policy, value["evaluation_seed"]), dict(
        binding=value, checkpoint_metadata=metadata,
        libraries=dict(torch=torch.__version__, numpy=np.__version__, torchvision=torchvision.__version__),
        core_package_paths=list(native), device=str(policy.device))


def make_predictor(policy, evaluation_seed):
    """Same packing/DDIM adapter as Gate4, seeded by learned action start."""
    def predict(history, chunk_index):
        import torch
        packed = api.pack_model_observation(history)
        obs = {key: torch.from_numpy(value).to(policy.device) for key, value in packed.items()}
        return overfit_policy.predict_windows(policy, obs,
            seed=evaluation_seed + 12*chunk_index, microbatch_size=1)[0].copy()
    return predict


def chunks(total_actions=200):
    if type(total_actions) is not int or not 200 <= total_actions <= 279:
        raise ValueError("stage3_episode_bound")
    return [dict(chunk_index=i, learned_start=12*i, global_start=50+12*i,
        global_observations=[49+12*i, 50+12*i], execute_count=min(12, total_actions-50-12*i),
        full_prediction_slots=list(range(1, min(12, total_actions-50-12*i)+1))) for i in range((total_actions-50+11)//12)]


def setup_action():
    # Exact existing approved settle-hold, with its float32 representation.
    return np.asarray((*TARGETS["A"], 1, 1), dtype=np.float32)


def selected_actions(full, count):
    raw = np.asarray(full)
    if raw.shape != (16, 5) or raw.dtype.kind != "f" or not np.isfinite(raw).all():
        raise ValueError("stage3_invalid_full_prediction_shape_type_or_finite")
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError("stage3_invalid_chunk_count")
    # Preserve official slots and established strict returned-action adapter.
    adapted = adapt_returned_actions_for_scheduler(raw[1:13])
    actions = adapted.scheduler_input[:count].copy()
    actions[:, 3:] = api.threshold_jaw_commands(actions[:, 3:])
    api.validate_policy_action(actions)
    # The native boundary already uses float32; reject representational overflow.
    dispatched = actions.astype(np.float32)
    api.validate_policy_action(dispatched)
    return dispatched


def array_identity(value):
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def with_requested_motion(outcome, actions, *, source='selected_checkpoint_commands_robot_base_metres'):
    """Replace only scripted command diagnostics, never physical acceptance math."""
    result = deepcopy(outcome)
    values = np.asarray(actions)
    if values.shape != (200, 5):
        raise ValueError("requested_motion_requires_full_selected_schedule")
    result.update(requested_tcp_lift_m=float(values[179, 2])-float(values[139, 2]),
        requested_tcp_lift_max_m=float(values[140:180, 2].max())-float(values[139, 2]),
        requested_tcp_final_delta_m=float(values[199, 2])-float(values[139, 2]),
        requested_tcp_source=source)
    return result


class PolicyWriter:
    """Images stored once; model input records reference those exact acquisitions."""
    prefix = "stage3"
    command_schema = "stage3-command-lifecycle/v1"

    def observation_indices(self, index):
        return dict(learned_action=index-50 if index >= 50 else None)

    def __init__(self, rgb, output):
        self.rgb, self.output = rgb, Path(output)
        for name in ("observations", "actions", "inferences", "substeps", "command_events"):
            with (self.output / (self.prefix + "_" + name + ".jsonl")).open("x") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        (self.output / "predictions").mkdir()

    def write_pre_candidate(self, mapping, metadata):
        return self.rgb.write_pre_candidate(mapping, metadata)

    def write_bundle(self, action_frame, mapping, metadata):
        result = self.rgb.write_bundle(action_frame, mapping, metadata)
        append(self.output / (self.prefix + "_observations.jsonl"), dict(global_action=action_frame,
            **self.observation_indices(action_frame),
            rgb_bundle=f"rgb/{action_frame:06d}.json", acquisition=deepcopy(metadata),
            arm_joint_pos=np.asarray(mapping["arm_joint_pos"]), jaw_state=np.asarray(mapping["jaw_state"]),
            input_identities={k: array_identity(v) for k, v in mapping.items()}))
        return result

    def write_substep(self, row):
        append(self.output / (self.prefix + "_substeps.jsonl"), row)

    def inference(self, index, raw, record):
        if raw is None:
            record.update(raw_file=None, raw_file_sha256=None)
            append(self.output / "stage3_inferences.jsonl", record)
            return
        path = self.output / "predictions" / f"{index:02d}.npy"
        with path.open("xb") as stream:
            np.save(stream, np.asarray(raw), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        record.update(raw_file=path.relative_to(self.output).as_posix(),
                      raw_file_sha256=sha256_file(path))
        append(self.output / "stage3_inferences.jsonl", record)

    def action(self, record):
        append(self.output / (self.prefix + "_actions.jsonl"), record)

    def command_event(self, event, index, **fields):
        append(self.output / (self.prefix + "_command_events.jsonl"), dict(
            schema=self.command_schema, event=event, global_action=index,
            wall_monotonic_ns=time.monotonic_ns(), **fields))

    def finish(self, completed, primary_errors):
        return self.rgb.finish(completed, primary_errors)


class LiveController:
    """Receives only already-acquired live observations. No trajectory fallback."""
    def __init__(self, predictor, writer, *, evaluation_seed=12345,
                 setup_actions=None, checkpoint_sha256=CHECKPOINT_SHA256, envelope_v2=False):
        self.predictor, self.writer = predictor, writer
        self.envelope_v2 = envelope_v2
        self.first_accepted_close = None
        self.total_actions = 200
        self.checkpoint_sha256 = checkpoint_sha256
        self.setup_actions = None if setup_actions is None else np.asarray(setup_actions, dtype=np.float32).copy()
        if self.setup_actions is not None:
            if self.setup_actions.shape != (50, 5):
                raise ValueError("stage3_setup_requires_exactly_fifty_actions")
            api.validate_policy_action(self.setup_actions)
        self.history = deque(maxlen=2)
        self.next_index, self.inference_count = 0, 0
        self.inference_attempts = 0
        self.evaluation_seed = evaluation_seed
        self.selected = None

    def accepted(self, sample, command):
        """Only measured action-term acceptance can extend the episode."""
        from .side_opening_admission import validate_command
        validate_command(sample, command)
        if not self.envelope_v2:
            return False
        t, sub = sample["control_tick"], sample["substep"]
        if t % 6 or sub != 0:
            raise ValueError("anchor_requires_action_acceptance_boundary")
        c = t//6
        if self.first_accepted_close is None and c <= 199 and command["bits"] == [0, 0]:
            self.first_accepted_close = c
            self.total_actions = max(200, c+80)
            return True
        return False

    def select(self, index, observation, sensor):
        if type(index) is not int or index != self.next_index or not 0 <= index < self.total_actions:
            raise ValueError("stage3_global_sequence")
        observation = api.validate_policy_observation(observation)
        self.history.append(observation)
        learned = index-50 if index >= 50 else None
        record = dict(global_action=index, learned_action=learned, setup_action=index if index < 50 else None,
            control_tick_start=6*index, native_clock=deepcopy(sensor["native_clock"]),
            source="approved_setup_hold" if index < 50 else "checkpoint_prediction",
            chunk_index=None, full_prediction_slot=None, checkpoint_sha256=None)
        if index < 50:
            action = setup_action() if self.setup_actions is None else self.setup_actions[index].copy()
        else:
            chunk_index, slot = divmod(learned, 12)
            if slot == 0:
                spec = chunks(279 if self.envelope_v2 else 200)[chunk_index]
                if len(self.history) != 2 or self.inference_count != chunk_index:
                    raise ValueError("stage3_live_history_or_chunk_sequence")
                packed = api.pack_model_observation(tuple(self.history))
                inference = dict(spec, seed=self.evaluation_seed+12*chunk_index,
                    checkpoint_sha256=self.checkpoint_sha256, native_clock=deepcopy(sensor["native_clock"]),
                    input_identities={k: array_identity(v) for k, v in packed.items()},
                    start_monotonic_ns=time.monotonic_ns())
                if self.envelope_v2:
                    inference["authorized_bound_at_prediction"] = self.total_actions
                    inference["forecast_retained_for_possible_extension"] = True
                self.inference_attempts += 1
                try:
                    raw = self.predictor(tuple(self.history), chunk_index)
                except Exception as exc:
                    inference.update(status="ERROR", end_monotonic_ns=time.monotonic_ns(),
                                     error=type(exc).__name__+":"+str(exc))
                    self.writer.inference(chunk_index, None, inference)
                    raise
                self.inference_count += 1
                inference["end_monotonic_ns"] = time.monotonic_ns()
                # Persist raw returns, including rejected/NaN numeric arrays.
                try:
                    self.selected = selected_actions(raw, spec["execute_count"])
                    inference.update(status="VALID", selected=self.selected.tolist())
                except Exception as exc:
                    inference.update(status="REJECTED", error=type(exc).__name__+":"+str(exc))
                    raise
                finally:
                    self.writer.inference(chunk_index, raw, inference)
            action = self.selected[slot].copy()
            record.update(chunk_index=chunk_index, full_prediction_slot=slot+1,
                          checkpoint_sha256=self.checkpoint_sha256)
        api.validate_policy_action(action)
        record["action"] = action.tolist()
        self.writer.action(record)
        self.writer.command_event("SELECTED_VALIDATED", index, decision=record,
            observation_references=[f"rgb/{i:06d}.json" for i in (
                [index] if learned is None else chunks(279 if self.envelope_v2 else 200)[chunk_index]["global_observations"])],
            scheduled_boundary=[6*index, 0])
        self.next_index += 1
        return action

    def summary(self, completed_actions):
        return dict(actual_inferences=self.inference_count,
            inference_attempts=self.inference_attempts,
            selected_setup_actions=min(self.next_index, 50), selected_learned_actions=max(0, self.next_index-50),
            completed_setup_actions=min(completed_actions, 50), completed_learned_actions=max(0, completed_actions-50),
            schedule=chunks(self.total_actions), checkpoint_sha256=self.checkpoint_sha256,
            **(dict(envelope_v2=True, first_accepted_close=self.first_accepted_close,
                    total_actions=self.total_actions) if self.envelope_v2 else {}))
