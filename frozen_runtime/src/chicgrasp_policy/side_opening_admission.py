"""Owner-approved Stage3 geometry, with causal measured closure references."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from . import proposed_geometric_witness as pure

SOURCE = "side-opening-progress/v1"
RULE = Path(__file__).resolve().parents[2] / "docs/diagnostics/gate6_stage3_repro_witness/RULE.json"
RULE_SHA256 = "770faeab856c154bf6f9ab48512fce3932bfe1fee494e451f7ec56da8eb00782"
COMMAND_SOURCE = "env.action_manager.get_term('policy_action').commanded_jaw_state"


def frozen_rule():
    raw = RULE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RULE_SHA256:
        raise ValueError("frozen_witness_rule_identity")
    rule = json.loads(raw)
    if rule["parameters"] != pure.parameters():
        raise ValueError("frozen_witness_parameters_changed")
    return rule


def validate_command(sample, command):
    if (command.get("bits") != sample["jaw_commands"]
            or command.get("acquisition_native_clock") != [sample["physics_step"], sample["simulation_time_s"]]
            or command.get("source") != COMMAND_SOURCE
            or command.get("semantics") != "commanded-jaw-state/v1"
            or command.get("dtype") != "uint8" or command.get("order") != ["LEFT", "RIGHT"]
            or any(type(v) is not int or v not in (0, 1) for v in command["bits"])):
        raise ValueError("accepted_command_provenance_invalid")


class WitnessBuffer:
    """Keep the real 61 samples; references outlive deque rotation, never closure."""
    def __init__(self, sink=None, *, run_type="HARNESS"):
        if run_type not in ("HARNESS", "EVIDENCE"):
            raise ValueError("explicit_witness_run_type_required")
        self.run_type = run_type
        frozen_rule()
        self.window = deque(maxlen=61)
        self.transitions = []
        self.bits = [1, 1]  # Independently verified reset-open sensor state.
        self.sink = sink

    def append(self, sample, accepted_command=None):
        previous = self.window[-1] if self.window else None
        if sample["control_tick"] % 6 == 0 and sample["substep"] == 0:
            if accepted_command is None:
                raise ValueError("accepted_command_measurement_missing")
            validate_command(sample, accepted_command)
            for side in range(2):
                bit = accepted_command["bits"][side]
                if bit != self.bits[side]:
                    event = dict(side=side, before=self.bits[side], after=bit,
                        pre_sample=deepcopy(previous), first_active=deepcopy(sample),
                        accepted_command=deepcopy(accepted_command))
                    # Persist before the only causal pre-close sample can rotate out.
                    if self.sink:
                        self.sink(event)
                    self.transitions.append(event)
            self.bits = list(accepted_command["bits"])
        elif sample["jaw_commands"] != self.bits:
            raise ValueError("command_changed_outside_action_boundary")
        self.window.append(deepcopy(sample))

    def evaluate(self, *, expected_end=pure.END):
        return evaluate(list(self.window), self.transitions, run_type=self.run_type, expected_end=expected_end)


def evaluate(window, transitions, *, run_type="HARNESS", expected_end=pure.END):
    """Recompute from measurements and accepted transitions, never an admission bit."""
    if run_type not in ("HARNESS", "EVIDENCE"):
        raise ValueError("explicit_witness_run_type_required")
    rule = frozen_rule()
    refs, bits, problems = [None, None], [1, 1], []
    try:
        last = [-1, -1]
        end = window[-1]
        for record in transitions:
            side, first, pre = record["side"], record["first_active"], record["pre_sample"]
            if type(side) is not int or side not in (0, 1):
                raise ValueError("transition_side")
            index = 2*first["control_tick"]+first["substep"]
            if (index <= last[side] or first["physics_step"] > end["physics_step"]
                    or first["control_tick"] % 6 or first["substep"] != 0
                    or first["episode_id"] != end["episode_id"]
                    or record["before"] != bits[side] or record["after"] != 1-bits[side]
                    or record["after"] != first["jaw_commands"][side]):
                raise ValueError("transition_order_or_command")
            validate_command(first, record["accepted_command"])
            if pre is not None:
                if (pre["episode_id"] != first["episode_id"]
                        or pre["physics_step"]+1 != first["physics_step"]
                        or 2*pre["control_tick"]+pre["substep"]+1 != index
                        or not math.isclose(first["simulation_time_s"]-pre["simulation_time_s"], 1/120, abs_tol=1e-8)
                        or pre["jaw_commands"][side] != record["before"]
                        or not pure._vector(pre["jaw_positions_m"], 4)):
                    raise ValueError("pre_close_not_causal_measured_sample")
            bits[side], last[side] = record["after"], index
            refs[side] = record if record["after"] == 0 else None
        if bits != end["jaw_commands"]:
            raise ValueError("transition_endpoint_command_mismatch")
        # Each window command must be explained by the persisted transition stream.
        for row in window:
            expected = [1, 1]
            for record in transitions:
                if record["first_active"]["physics_step"] <= row["physics_step"]:
                    expected[record["side"]] = record["after"]
            if expected != row["jaw_commands"]:
                raise ValueError("window_transition_command_mismatch")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        problems.append(str(exc))
    result = pure.evaluate_history(window, rule["reference_opening_m"], pre_close_records=refs, expected_end=expected_end) if not problems else dict(
        status="UNAVAILABLE", issues=problems, sides=[])
    sample = {k: window[-1][k] for k in ("episode_id", "control_tick", "substep", "physics_step", "simulation_time_s")} if window else None
    result.update(label="STAGE3_EVIDENCE_AUTHORIZED" if run_type == "EVIDENCE" else "STAGE3_HARNESS_ONLY", source=SOURCE, rule_sha256=RULE_SHA256,
                  sample=sample, history=deepcopy(window), transitions=deepcopy(transitions))
    return result


def replay(rows, saved_transitions, *, run_type="HARNESS", envelope_v2=False, protocol="v2"):
    """Independently join the transition stream to actual saved measurement rows."""
    from .geometric_capture import history_sample
    from .envelope_schedule import event_schedule
    buffer, gate = WitnessBuffer(run_type=run_type), None
    anchor = None
    for row in rows:
        buffer.append(history_sample(row), row.get("accepted_command"))
        t, sub = row["context"]["control_tick"], row["context"]["substep"]
        if envelope_v2 and anchor is None and t % 6 == 0 and sub == 0 and t//6 <= 199 and row.get("accepted_command", {}).get("bits") == [0, 0]:
            anchor = t//6
        end = (839, 1) if not envelope_v2 else (None if anchor is None else divmod(event_schedule(anchor, protocol=protocol)["decision"], 2))
        if (t, sub) == end:
            gate = buffer.evaluate(expected_end=end)
    if buffer.transitions != saved_transitions:
        raise ValueError("saved_transition_stream_measurement_disagreement")
    return gate
