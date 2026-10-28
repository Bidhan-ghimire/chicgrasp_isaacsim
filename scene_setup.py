"""Edit the research scene here; preview, recorder and live run share this file.

The supplied callbacks preserve the original table setup. Edit configure_scene
or build_scene to make your own scene, then inspect it before recording. A scene
name is only a label: naming a scene does not create a task or change physics.
All functions are imported without starting Isaac Sim.
"""

SCENE_NAME = "table"  # Any nonempty descriptive name; no built-in task selection.
USE_ORIGINAL_TABLE_TASK = True  # Keeps original placement, attachment and evaluation.
# For a different task, set this to False and edit the callbacks below. The
# generic custom path adds no physics and reports no task success by default.

# Declare additional mesh/texture/config files used by your scene, relative to
# the repository. Their hashes become part of approval and model provenance.
ASSET_FILES = []


def describe_scene(options=None):
    """Pure description for plans and identities; the name never selects behavior."""
    if not isinstance(SCENE_NAME, str) or not SCENE_NAME.strip():
        raise ValueError("SCENE_NAME must be a nonempty descriptive name")
    if type(USE_ORIGINAL_TABLE_TASK) is not bool:
        raise ValueError("USE_ORIGINAL_TABLE_TASK must be True or False")
    if USE_ORIGINAL_TABLE_TASK and options:
        raise ValueError("The table default has no scene_options")
    return {"id": SCENE_NAME,
            "kind": "table" if USE_ORIGINAL_TABLE_TASK else "custom"}


def validate_position(position, options=None):
    """Finite XYZ and unit quaternion checks are also enforced by the bridge."""
    definition = describe_scene(options)
    if definition["kind"] == "table":
        x, y, z = position["authored_xyz_m"]
        if not (0.6400000393390657 <= x <= 0.699988579750061
                and -0.009999997913837447 <= y <= 0.295 and z == 0.8155):
            raise ValueError("Position is outside the supported attempt005 placement rectangle")
        if position.get("authored_quaternion_wxyz", [1, 0, 0, 0]) != [1, 0, 0, 0]:
            raise ValueError("The table scene requires identity chicken orientation")
    # Add your own custom-task placement bounds here. Validation of finite
    # coordinates does not establish reachability or physical task validity.


def configure_scene(cfg, position, options=None):
    """Edit Isaac Lab scene configuration before environment construction.

    The shared adapter already sets chicken init_state.pos/rot from position.
    Add task-specific config edits here. Never rewrite frozen source/assets.
    """
    describe_scene(options)


def build_scene(stage, position, options=None):
    """Add your own live-stage edits before the first physics reset.

    This default creates nothing. Author additions in the session layer; do not
    save changes into frozen assets. No custom task physics is supplied here.
    """
    definition = describe_scene(options)
    return {"created_prims": [], "scene": definition["id"]}


def evaluate_episode(capture, default_metrics, options=None):
    """Define your custom task's success criteria here, using saved diagnostics.

    The generic fallback admits technically complete scripted demos for training
    but assigns no task success. Replace this rule for your own task. The adapter
    additionally rejects incomplete/nonfinite/reset captures and live runs can
    never become training demonstrations through this callback.
    """
    if describe_scene(options)["kind"] == "table":
        return dict(default_metrics,
                    training_eligible=default_metrics.get("status") == "PASS",
                    training_eligibility_basis="ORIGINAL_TABLE_END_STATE")
    complete = (capture.completed and not capture.lifecycle_events
                and not getattr(capture, "recording_errors", []))
    return {"status": "UNSCORED", "label": "CUSTOM_SCENE_TASK_NOT_SCORED",
            "training_eligible": bool(complete),
            "training_eligibility_basis": "TECHNICALLY_COMPLETE_SCRIPTED_DEMONSTRATION",
            "temporary_gripper_attachment": False}
