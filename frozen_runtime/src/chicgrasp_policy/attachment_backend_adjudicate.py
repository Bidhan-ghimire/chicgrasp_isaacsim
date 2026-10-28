"""Independent saved CPU-profile checks and diagnostic-only report labels.

No runtime configuration, native imports, contact decisions, or evaluator math.
The ordinary historical path has no added fields and remains GPU-only.
"""
from copy import deepcopy
import math


PRODUCTION_GPU = "PRODUCTION_GPU"
CPU = "CPU_BACKEND_DIAGNOSTIC"
APP = {"headless": True, "enable_cameras": True, "device": "cuda:0",
       "fast_shutdown": True, "livestream": 0, "experience": ""}
CPU_EXPECTED = {
    "task_id": "Isaac-ChicGrasp-GraspLift-DynamicChicken-IK-Abs-v0",
    "cfg_class": "ChicgraspDynamicChickenEnvCfg", "num_envs": 1,
    "device": "cpu", "simulation_device": "cpu", "physics_device": "cpu",
    "gpu_sim": False, "gpu_pipeline": False, "gpu_dynamics": False, "broadphase": "MBP",
    "requested_simulation_device": "cpu", "suppress_readback": False,
    "backend_mode": CPU, "physics_backend": "CPU",
    "decimation": 2, "render_interval": 2, "action_dimension": 5,
    "cameras_enabled": True, "offscreen_render": True, "rtx_sensors": True,
    "fast_shutdown": True, "has_gui": False, "configured_seed": 42,
}


def _same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def scoped_candidate_result(evaluated, *, incomplete=False):
    """Copy the unchanged mathematical result, adding an unambiguous CPU scope."""
    answer = deepcopy(evaluated)
    label = ("NOT_EVALUABLE" if incomplete else
             answer.get("label", "PIPELINE_VERTICAL_SLICE_PASS_WITH_TEMPORARY_ATTACHMENT")
             if answer.get("status") == "PASS" else "ATTACHMENT_ASSISTED_CANDIDATE_FAIL")
    answer.update(backend_mode=CPU, gpu_gate5_acceptance=False, label=CPU + ":" + label)
    return answer


def check_cpu_backend(result, *, invocation=None, managed_anchor=None):
    """Recompute from requested and effective observations, never the live verdict.

    Invocation identity is authenticated by the parent caller. Child-side report
    construction has no invocation yet; its result is independently recomputed
    with that additional input after the process finishes.
    """
    missing, contradictory = [], []
    def equal(value, wanted, field):
        if value is None:
            missing.append(field)
        elif type(value) is not type(wanted) or value != wanted:
            contradictory.append(field)
    equal(result.get("backend_mode"), CPU, "backend_mode")
    equal(result.get("observation_mode"), True, "observation_mode")
    equal(result.get("contact_transport"), "callback-contact/v1", "contact_transport")
    equal(result.get("contact_storage_schema"), "gate5-compact-contact/v1", "contact_storage_schema")
    equal(result.get("gpu_gate5_acceptance"), False, "gpu_gate5_acceptance")
    equal(result.get("end_state_scope"), CPU, "end_state_scope")
    diagnostic = result.get("backend_diagnostic")
    if not isinstance(diagnostic, dict):
        missing.append("backend_diagnostic")
        diagnostic = {}
    equal(diagnostic.get("schema"), "cpu-backend-diagnostic/v1", "diagnostic.schema")
    equal(diagnostic.get("backend_mode"), CPU, "diagnostic.backend_mode")
    requested = diagnostic.get("requested_configuration", {})
    if not isinstance(requested, dict):
        contradictory.append("requested_configuration")
        requested = {}
    equal(requested.get("simulation_device"), "cpu", "requested.simulation_device")
    application = requested.get("application_configuration")
    if not isinstance(application, dict):
        missing.append("requested.application_configuration")
    else:
        if set(application) != set(APP):
            contradictory.append("requested.application_configuration.membership")
        for key, value in APP.items():
            equal(application.get(key), value, "requested.application_configuration." + key)
    config = result.get("effective_configuration")
    if not isinstance(config, dict):
        missing.append("effective_configuration")
        config = {}
    observed = diagnostic.get("effective_configuration")
    if not isinstance(observed, dict):
        missing.append("diagnostic.effective_configuration")
    elif not _same(observed, config):
        contradictory.append("diagnostic.effective_configuration_disagreement")
    for key, value in CPU_EXPECTED.items():
        if key != "physics_backend":
            equal(config.get(key), value, "effective." + key)
    # The convenience label is derived metadata, not an independent operand.
    # A missing getter must stay UNAVAILABLE rather than being contradicted by
    # the resulting truthful UNAVAILABLE label.
    cpu_physics = {key: CPU_EXPECTED[key] for key in (
        "device", "simulation_device", "physics_device", "gpu_sim", "gpu_pipeline",
        "gpu_dynamics", "broadphase", "suppress_readback")}
    gpu_physics = dict(device="cuda:0", simulation_device="cuda:0", physics_device="cuda:0",
                       gpu_sim=True, gpu_pipeline=True, gpu_dynamics=True,
                       broadphase="GPU", suppress_readback=True)
    derived = ("CPU" if all(_same(config.get(key), value) for key, value in cpu_physics.items()) else
               "GPU" if all(_same(config.get(key), value) for key, value in gpu_physics.items()) else "UNAVAILABLE")
    equal(config.get("physics_backend"), derived, "effective.physics_backend_derived_label")
    for key, value in (("physics_dt_s", 1 / 120), ("control_dt_s", 1 / 60)):
        actual = config.get(key)
        if actual is None:
            missing.append("effective." + key)
        elif type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(
                actual, value, rel_tol=0, abs_tol=1e-12):
            contradictory.append("effective." + key)
    # The raw renderer selector is separate metadata; None is not GPU/CPU physics.
    if "renderer_active_gpu" not in config:
        missing.append("effective.renderer_active_gpu_record_missing")
    errors = diagnostic.get("read_errors")
    if not isinstance(errors, list):
        missing.append("diagnostic.read_errors")
    elif any(not isinstance(error, dict) or error.get("field") != "renderer_active_gpu" for error in errors):
        missing.append("diagnostic.required_read_error")
    phase = diagnostic.get("observation_phase")
    if phase == "not_observed" or phase is None:
        missing.append("diagnostic.observation_phase")
    elif phase != "after_initial_reset_before_candidate":
        contradictory.append("diagnostic.observation_phase")
    wall = diagnostic.get("monotonic_ns")
    if wall is None:
        missing.append("diagnostic.monotonic_ns")
    elif type(wall) is not int or wall < 0:
        contradictory.append("diagnostic.monotonic_ns")
    anchor = diagnostic.get("physics_scene_identity")
    if not isinstance(anchor, dict) or anchor.get("stage_cache_id") is None:
        missing.append("diagnostic.physics_scene_identity")
    elif (type(anchor["stage_cache_id"]) is not int or anchor["stage_cache_id"] <= 0
          or anchor.get("physics_scene_path") != "/physicsScene"
          or (managed_anchor is not None and not _same(anchor, managed_anchor))):
        contradictory.append("diagnostic.physics_scene_identity")
    if invocation is not None:
        equal(invocation.get("backend_mode"), CPU, "invocation.backend_mode")
        equal(invocation.get("observation_mode"), True, "invocation.observation_mode")
        equal(invocation.get("contact_transport"), "callback-contact/v1", "invocation.contact_transport")
        arguments = invocation.get("arguments")
        if not isinstance(arguments, list) or arguments.count("--backend-mode") != 1:
            contradictory.append("invocation.backend_mode_not_explicit")
        else:
            index = arguments.index("--backend-mode")
            if index + 1 >= len(arguments) or arguments[index + 1] != CPU:
                contradictory.append("invocation.backend_mode_not_explicit")
    if not missing and not contradictory:
        if diagnostic.get("verification") != "CPU_CONFIGURATION_VERIFIED":
            contradictory.append("diagnostic.verification_disagreement")
        if diagnostic.get("validation_issues") != []:
            contradictory.append("diagnostic.validation_issues_disagreement")
    status = ("CPU_CONFIGURATION_CONTRADICTED" if contradictory else
              "CPU_CONFIGURATION_UNAVAILABLE" if missing else "CPU_CONFIGURATION_VERIFIED")
    return {"schema": "cpu-backend-saved-adjudication/v1", "backend_mode": CPU,
            "BACKEND_DIAGNOSTIC_RESULT": status,
            "missing": list(dict.fromkeys(missing)), "contradictions": list(dict.fromkeys(contradictory)),
            "gpu_gate5_acceptance": False, "end_state_scope": CPU}


def interpret_cpu_contact(control_result, production_witnesses, bilateral_boundaries, *, qualified):
    """Report-only A/B/C/D meanings after saved evidence has been validated."""
    answer = {"backend_mode": CPU, "gpu_gate5_acceptance": False,
        "control_finding": "CPU_CONTACT_DELIVERY_NOT_EVALUABLE", "cases": [],
        "production_current_witnesses": production_witnesses,
        "fresh_bilateral_boundaries": bilateral_boundaries, "interpretation": [],
        "limitations": ["CPU diagnostic is not GPU Gate5 acceptance",
            "no unique root cause, production fix, or physical grasp miss is established",
            "multiple reporting causes can coexist"]}
    if not qualified:
        return answer
    if control_result == "CONTROL_CONTACT_REPORTED":
        answer["control_finding"] = "CPU_CONTACT_DELIVERY_PRESENT"
        answer["cases"].append("A")
        answer["interpretation"].append("A GPU-dependent configuration/path difference is implicated in the previous silence; its responsible component is unresolved.")
        if production_witnesses == 0:
            answer["cases"].append("C")
            answer["interpretation"].append("Primitive control delivery is demonstrated on CPU; articulated/compound production coverage remains unresolved, not a physical grasp miss.")
    elif control_result == "CONTROL_CONTACT_EXPECTED_BUT_NOT_REPORTED":
        answer["control_finding"] = "CPU_CONTACT_DELIVERY_ABSENT"
        answer["cases"].append("B")
        answer["interpretation"].append("Silence reproduces without GPU physics; a GPU-only cause is insufficient and a common reporting setup/producer problem remains implicated, not uniquely established.")
    if bilateral_boundaries:
        answer["cases"].append("D")
        answer["interpretation"].append("Fresh bilateral production evidence is present under unchanged admission semantics; attachment and END STATE remain separately evaluated CPU diagnostics.")
    return answer
