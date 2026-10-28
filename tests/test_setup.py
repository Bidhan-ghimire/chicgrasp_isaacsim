"""Offline regression tests grouped by setup."""
from __future__ import annotations



# Experiments

"""Named-config creation tests use only temporary fixture repositories."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from unittest.mock import patch

from pipeline import setup as experiments


class ExperimentCreationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experiment_creation_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"repository with spaces"
        (self.root/"configs").mkdir(parents=True)
        self.template = self.root/"configs/research.json"
        source = experiments.ROOT/"configs/research.json"
        self.template.write_bytes(source.read_bytes())
        self.before = self.template.read_bytes()
        self.defaults = json.loads(self.before)

    def create(self, name="Demo-01_A"):
        return experiments.create_experiment(name, repo_root=self.root)

    def test_exact_paths_and_only_four_config_fields_change(self):
        receipt = self.create()
        self.assertEqual(receipt, {
            "status":"CREATED", "name":"Demo-01_A", "config":"configs/experiments/Demo-01_A.json",
            "root":"research_outputs/Demo-01_A",
            "outputs":{"scene":"research_outputs/Demo-01_A/scene001",
                       "collection":"research_outputs/Demo-01_A/collection001",
                       "training":"research_outputs/Demo-01_A/training001",
                       "live":"research_outputs/Demo-01_A/live001"},
            "scene_approval":"research_outputs/Demo-01_A/scene-approval001.json",
            "dataset":"research_outputs/Demo-01_A/collection001/dataset.json",
            "artifact":"research_outputs/Demo-01_A/training001/model/artifact.json",
        })
        generated = json.loads((self.root/receipt["config"]).read_text())
        expected = dict(self.defaults, experiment_id=receipt["name"], scene_approval=receipt["scene_approval"],
                        dataset=receipt["dataset"], model_artifact=receipt["artifact"])
        self.assertEqual(generated, expected)
        self.assertEqual(self.template.read_bytes(), self.before)
        self.assertFalse((self.root/"research_outputs").exists())
        self.assertEqual(sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()),
                         [receipt["config"], "configs/research.json"])

    def test_valid_boundary_names_preserve_exact_case(self):
        for name in ("a", "9", "A"*64, "Mixed_CASE-123", "COM10", "console"):
            with self.subTest(name=name):
                receipt = self.create(name)
                self.assertEqual(receipt["name"], name)
                self.assertEqual(json.loads((self.root/receipt["config"]).read_text())["experiment_id"], name)

    def test_generated_config_loads_with_existing_settings_validation(self):
        from pipeline.settings import load_experiment
        receipt = self.create()
        generated_path = self.root/receipt["config"]
        generated_before = generated_path.read_bytes()
        loaded = load_experiment(self.root/receipt["config"])
        self.assertEqual(loaded["experiment_id"], receipt["name"])
        self.assertEqual(loaded["scene_approval"], receipt["scene_approval"])
        self.assertEqual(loaded["dataset"], receipt["dataset"])
        self.assertEqual(loaded["model_artifact"], receipt["artifact"])
        changed = {"experiment_id", "scene_approval", "dataset", "model_artifact", "limits"}
        self.assertEqual({key:loaded[key] for key in self.defaults if key not in changed},
                         {key:value for key,value in self.defaults.items() if key not in changed})
        self.assertEqual(loaded["limits"], dict(self.defaults["limits"], training_seconds=None))
        self.assertEqual(generated_path.read_bytes(), generated_before)
        self.assertEqual(json.loads(generated_before)["limits"], self.defaults["limits"])
        self.assertFalse((self.root/"research_outputs").exists())

    def test_invalid_names_are_rejected_before_creating_files(self):
        names = ("", "a"*65, "../escape", "..", ".", "a/b", "a\\b", "C:drive",
                 "a.json", " name", "name ", "name.", "two words", "_start", "-start",
                 "é", "a\n", "a\t", "a\x00", "你好", None, 1, "CONIN$", "CONOUT$")
        for name in names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.create(name)
        self.assertFalse((self.root/"configs/experiments").exists())
        self.assertFalse((self.root/"research_outputs").exists())
        self.assertEqual(self.template.read_bytes(), self.before)

    def test_windows_reserved_names_are_case_insensitive(self):
        for name in ("CON", "prn", "Aux", "nul", "com1", "COM9", "lpt1", "LPT9"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    self.create(name)

    def test_config_collision_preserves_original_bytes(self):
        receipt = self.create()
        target = self.root/receipt["config"]
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self.template.read_bytes(), self.before)

    def test_existing_output_directory_refuses_config_creation(self):
        output = self.root/"research_outputs/Demo-01_A"
        output.mkdir(parents=True)
        retained = output/"retained.json"
        retained.write_text("original evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(retained.read_text(), "original evidence")
        self.assertFalse((self.root/"configs/experiments/Demo-01_A.json").exists())

    def test_existing_output_file_also_refuses_config_creation(self):
        output = self.root/"research_outputs/Demo-01_A"
        output.parent.mkdir()
        output.write_bytes(b"retained")
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(output.read_bytes(), b"retained")
        self.assertFalse((self.root/"configs/experiments").exists())

    def test_unrelated_output_directory_is_preserved(self):
        other = self.root/"research_outputs/other"
        other.mkdir(parents=True)
        self.create()
        self.assertTrue(other.is_dir())
        self.assertFalse((self.root/"research_outputs/Demo-01_A").exists())

    def test_resolved_config_or_output_escape_is_rejected(self):
        original = Path.resolve
        outside = self.root.parent/"outside"
        for part in ("configs/experiments", "research_outputs/Demo-01_A"):
            def pretend_redirect(path, strict=False):
                if part in path.as_posix():
                    return outside/path.name
                return original(path, strict=strict)
            with self.subTest(part=part), patch.object(Path, "resolve", pretend_redirect):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    self.create()
        self.assertFalse(outside.exists())
        self.assertFalse((self.root/"configs/experiments").exists())

    def test_missing_template_writes_no_config(self):
        empty = self.root.parent/"empty repo"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            experiments.create_experiment("demo", repo_root=empty)
        self.assertFalse((empty/"configs").exists())

    def test_malformed_template_writes_no_config(self):
        for body in ("[]", "{broken", '{"schema":"wrong"}'):
            with self.subTest(body=body):
                self.template.write_text(body, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.create()
                self.assertFalse((self.root/"configs/experiments").exists())

    def test_nonfinite_template_rejected_before_file_creation(self):
        self.template.write_text('{"schema":"research-experiment/v1","bad":NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse((self.root/"configs/experiments").exists())

    def test_main_json_is_clean_and_errors_are_on_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = experiments.create_experiment_main(["demo", "--json"], repo_root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue())["status"], "CREATED")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = experiments.create_experiment_main(["demo", "--json"], repo_root=self.root)
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Already exists", stderr.getvalue())

    def _cli(self, *args):
        shutil.copyfile(experiments.ROOT/"create_experiment.py", self.root/"create_experiment.py")
        (self.root/"pipeline").mkdir(exist_ok=True)
        (self.root/"pipeline/__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(experiments.__file__, self.root/'pipeline/setup.py')
        shutil.copyfile(experiments.ROOT/"pipeline/settings.py", self.root/"pipeline/settings.py")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        return subprocess.run([sys.executable, "-B", str(self.root/"create_experiment.py"), *args],
            cwd=self.root, env=env, text=True, capture_output=True, timeout=15, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def test_cli_subprocess_json_and_exclusive_failure(self):
        result = self._cli("cli-demo", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["config"], "configs/experiments/cli-demo.json")
        self.assertFalse((self.root/receipt["root"]).exists())
        again = self._cli("cli-demo", "--json")
        self.assertEqual(again.returncode, 2)
        self.assertEqual(again.stdout, "")
        self.assertIn("Already exists", again.stderr)

    def test_cli_human_output_help_and_invalid_name(self):
        result = self._cli("human-demo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Created experiment: human-demo", result.stdout)
        self.assertIn("Scene: research_outputs/human-demo/scene001", result.stdout)
        help_result = self._cli("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("NAME", help_result.stdout)
        bad = self._cli("../bad", "--json")
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(bad.stdout, "")
        self.assertIn("1-64 ASCII", bad.stderr)

    def test_no_overwrite_flag_or_heavy_imports(self):
        result = self._cli("demo", "--overwrite")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root/"configs/experiments/demo.json").exists())
        tree = ast.parse(Path(experiments.__file__).read_text(encoding="utf-8"))
        # Creation now shares a module with discovery/installation. Check its
        # complete creation path and eager imports, excluding unrelated helpers.
        creation_names = {"create_experiment_main", "create_experiment", "validate_name", "_inside"}
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 or isinstance(node, ast.FunctionDef) and node.name in creation_names]
        eager_settings = ast.parse((experiments.ROOT/"pipeline/settings.py").read_text(encoding="utf-8"))
        nodes = [node for node in nodes if not isinstance(node, ast.ImportFrom) or node.module != "pipeline.settings"]
        nodes += [node for node in eager_settings.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        creation_tree = ast.Module(body=nodes, type_ignores=[])
        imports = {node.module.split(".")[0] for node in ast.walk(creation_tree) if isinstance(node, ast.ImportFrom)}
        imports.update(alias.name.split(".")[0] for node in ast.walk(creation_tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue(imports <= set(sys.stdlib_module_names), imports)
        self.assertEqual(hashlib.sha256(self.template.read_bytes()).hexdigest(), hashlib.sha256(self.before).hexdigest())


# Setup discovery

"""Offline discovery fixtures. Never launch SDK Python, Kit, pip or FFmpeg."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import setup as subject


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.lock_text = (subject.ROOT/"environments/policy-support-win311.txt").read_text(encoding="utf-8")
        self.temp = tempfile.TemporaryDirectory(prefix="research_setup_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root/"repository with spaces"
        self.repo.mkdir()
        self.root_patch = patch.object(subject, "ROOT", self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.write(self.repo/"environments/policy-support-win311.txt", self.lock_text)
        rows = []
        for name, key, path in (("experience", "isaac_lab", "apps/isaaclab.python.rendering.kit"),
            ("ground_shader", "isaac_sim", "kit/mdl/core/Base/OmniPBR.mdl"),
            ("physx_binding", "isaac_sim", "extscache/pinned/physx.pyd"),
            ("sim_version", "isaac_sim", "VERSION"), ("lab_version", "isaac_lab", "VERSION")):
            data = ("fixture " + name).encode("ascii")
            rows.append(dict(name=name, config_key=key, relative_path=path, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest()))
        self.rows = rows
        self.write(self.repo/'pipeline/runtime/data/SDK_RESOURCE_IDENTITIES.json',
            json.dumps(dict(schema="hc-sdk-resources/v1", resources=rows)))
        self.sim = self.make_sdk(self.root/"IsaacSim", "isaac_sim")
        self.lab = self.make_sdk(self.root/"IsaacLab", "isaac_lab")
        self.support = self.repo/".local/policy_support"
        self.support.mkdir(parents=True)
        self.pins = subject.load_support_pins()
        for name, version in self.pins.items():
            self.add_distribution(name, version)
        self.resource_calls = []

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def add_distribution(self, name, version, suffix=""):
        directory = self.support/(name.replace("-", "_")+"-"+version+suffix+".dist-info")
        self.write(directory/"METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return directory

    def make_sdk(self, root, key):
        layout = subject.SIM_LAYOUT if key == "isaac_sim" else subject.LAB_LAYOUT
        for part in layout:
            if Path(part).suffix:
                self.write(root/part, "inert fixture file")
            else:
                (root/part).mkdir(parents=True, exist_ok=True)
        for row in self.rows:
            if row["config_key"] == key:
                self.write(root/row["relative_path"], "fixture " + row["name"])
        return root

    def git_reader(self, path):
        return dict(head=subject.LAB_COMMIT, tracked_changes="")

    def resources(self, config):
        self.resource_calls.append(dict(config))
        return {row["name"]:row for row in self.rows}

    def discover(self, **kwargs):
        args = dict(env={}, repo_root=self.repo, include_common=False,
            git_reader=self.git_reader, resource_checker=self.resources)
        args.update(kwargs)
        return subject.discover(**args)

    def test_discovers_siblings_support_and_derives_sdk_python_without_execution(self):
        with patch.object(subject.subprocess, "run", side_effect=AssertionError("No subprocess permitted")):
            result = self.discover()
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertEqual(result["selected"]["training_python"], str(self.sim/"kit/python/python.exe"))
        self.assertTrue(result["sdk_validated"])
        self.assertEqual(len(self.resource_calls), 1)
        self.assertEqual(result["video_status"], "video_unconfigured")

    def test_invalid_override_never_falls_back_to_valid_environment(self):
        result = self.discover(overrides={"isaac_sim":str(self.root/"missing")}, env={"ISAAC_PATH":str(self.sim)})
        self.assertNotIn("isaac_sim", result["selected"])
        self.assertNotIn("training_python", result["selected"])
        self.assertIn("isaac_sim", result["missing_keys"])

    def test_invalid_existing_host_never_replaced_by_discovered_path(self):
        result = self.discover(existing_host={"isaac_lab":str(self.root/"missing")})
        self.assertNotIn("isaac_lab", result["selected"])
        self.assertIn("isaac_lab", result["missing_keys"])

    def test_existing_valid_setting_beats_other_valid_discovery(self):
        alternative = self.make_sdk(self.root/"isaacsim-second", "isaac_sim")
        result = self.discover(existing_host={"isaac_sim":str(alternative)})
        self.assertEqual(result["selected"]["isaac_sim"], str(alternative))
        self.assertFalse(result["issues"])

    def test_environment_beats_existing_host(self):
        result = self.discover(env={"ISAACSIM_PATH":str(self.sim)}, existing_host={"isaac_sim":str(self.root/"missing")})
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertFalse(result["issues"])

    def test_windows_environment_names_and_expansion_are_case_insensitive(self):
        result = self.discover(env={"isaac_sim_path":"%SdkRoot%", "SdkRoot":str(self.sim)})
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertFalse(result["issues"])

    def test_invalid_losing_priority_does_not_block_valid_override(self):
        result = self.discover(overrides={"isaac_sim":str(self.sim)},
                               existing_host={"isaac_sim":""})
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertFalse(result["issues"])
        self.assertTrue(result["invalid_inputs"]["isaac_sim"])

    def test_two_valid_discovered_sdks_require_choice(self):
        self.make_sdk(self.root/"isaacsim-second", "isaac_sim")
        result = self.discover()
        self.assertNotIn("isaac_sim", result["selected"])
        self.assertTrue(any("Multiple valid isaac_sim" in issue for issue in result["issues"]))

    def test_two_environment_aliases_are_not_silently_resolved(self):
        alternative = self.make_sdk(self.root/"sdk alternate", "isaac_sim")
        result = self.discover(env={"ISAACSIM_PATH":str(self.sim), "ISAAC_PATH":str(alternative)})
        self.assertNotIn("isaac_sim", result["selected"])

    def test_wrong_sdk_resource_hash_rejected(self):
        self.write(self.sim/"VERSION", "altered bytes")
        result = self.discover()
        self.assertNotIn("isaac_sim", result["selected"])
        self.assertNotIn("training_python", result["selected"])

    def test_missing_sitecustomize_rejected(self):
        with patch.object(subject, "SIM_LAYOUT", (*subject.SIM_LAYOUT, "site/missing_sitecustomize.py")):
            result = self.discover()
        self.assertNotIn("isaac_sim", result["selected"])

    def test_wrong_lab_revision_dirty_checkout_and_unauthenticated_archive_rejected(self):
        for identity in (dict(head="bad", tracked_changes=""),
                         dict(head=subject.LAB_COMMIT, tracked_changes=" M source/file.py"),
                         dict(error="Not a Git checkout")):
            with self.subTest(identity=identity):
                result = self.discover(git_reader=lambda path:identity)
                self.assertNotIn("isaac_lab", result["selected"])

    def test_full_support_metadata_passes_without_importing_packages(self):
        result = subject.validate_support(self.support)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checks"]["matched_pin_count"], 36)
        self.assertFalse(result["checks"]["imported_packages"])

    def test_support_wrong_version_missing_and_duplicate_metadata_rejected(self):
        missing = self.root/"empty support"
        missing.mkdir()
        self.assertEqual(subject.validate_support(missing)["status"], "FAIL")
        self.add_distribution("diffusers", "999")
        result = subject.validate_support(self.support)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("Duplicate distribution" in error for error in result["errors"]))

    def test_forbidden_core_package_directories_and_metadata_rejected(self):
        (self.support/"torch").mkdir()
        self.write(self.support/"numpy.py", "raise RuntimeError('must never import')")
        self.add_distribution("torchvision", "999")
        result = subject.validate_support(self.support)
        for name in ("torch", "numpy", "torchvision"):
            self.assertTrue(any("core package: " + name in error for error in result["errors"]))

    def test_ffmpeg_requires_probe_optional_when_not_found(self):
        exe = self.root/"tools/bin/ffmpeg.exe"
        self.write(exe, "inert")
        result = self.discover(overrides={"ffmpeg":str(exe)})
        self.assertIn("ffmpeg", result["missing_keys"])
        self.assertNotIn("ffmpeg", result["selected"])
        self.write(exe.with_name("ffprobe.exe"), "inert")
        result = self.discover(overrides={"ffmpeg":str(exe.parent.parent)})
        self.assertEqual(result["selected"]["ffmpeg"], str(exe))
        self.assertEqual(result["video_status"], "READY")

    def test_empty_explicit_optional_values_do_not_fall_back(self):
        result = self.discover(overrides={"ffmpeg":"", "training_python":""})
        self.assertIn("ffmpeg", result["missing_keys"])
        self.assertIn("training_python", result["missing_keys"])
        self.assertNotIn("training_python", result["selected"])

    def test_custom_training_interpreter_preserved_as_file_only(self):
        custom = self.root/"training tools/python.exe"
        self.write(custom, "inert")
        result = self.discover(existing_host={"training_python":str(custom)})
        self.assertEqual(result["selected"]["training_python"], str(custom))
        self.assertFalse(result["candidates"]["training_python"][0]["checks"]["interpreter_executed"])

    def test_path_python_ancestors_and_ffmpeg_are_bounded_candidates(self):
        exe = self.root/"tools/ffmpeg.exe"
        self.write(exe, "inert")
        self.write(exe.with_name("ffprobe.exe"), "inert")
        result = self.discover(env={"PATH":os.pathsep.join([str(self.sim/"kit/python"), str(exe.parent)])})
        matching = [row for row in result["candidates"]["isaac_sim"] if row["path"]==str(self.sim)][0]
        self.assertIn("PATH Python bounded ancestor", matching["sources"])
        self.assertEqual(result["selected"]["ffmpeg"], str(exe))

    def test_lab_conventional_link_candidate_resolves_same_sdk_once(self):
        link = self.lab/"_isaac_sim"
        # Use a real directory fixture as conventional target; symlinks require Windows privileges.
        self.make_sdk(link, "isaac_sim")
        result = self.discover(overrides={"isaac_sim":str(link)})
        matching = [row for row in result["candidates"]["isaac_sim"] if row["path"]==str(link)][0]
        self.assertIn("Isaac Lab _isaac_sim link", matching["sources"])
        self.assertEqual(result["selected"]["isaac_sim"], str(link))

    def test_relative_existing_host_paths_resolve_from_host_directory(self):
        host = self.root/"stored/config.json"
        self.write(host, json.dumps({"isaac_sim":"../IsaacSim", "isaac_lab":"../IsaacLab"}))
        result = self.discover(existing_host=host)
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertEqual(result["selected"]["isaac_lab"], str(self.lab))

    def test_malformed_existing_host_does_not_silently_create_replacement(self):
        host = self.root/"stored/config.json"
        self.write(host, "[]")
        result = self.discover(existing_host=host)
        self.assertFalse(result["selected"])
        self.assertTrue(result["issues"])

    def test_full_host_calls_resource_verifier_and_reports_optional_video(self):
        config = {"isaac_sim":str(self.sim), "isaac_lab":str(self.lab), "policy_support":str(self.support)}
        result = subject.validate_host(config, git_reader=self.git_reader, resource_checker=self.resources)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checks"]["sdk_resource_pin"]["status"], "PASS")
        self.assertEqual(result["video_status"], "video_unconfigured")
        self.assertEqual(len(self.resource_calls), 1)
        self.assertFalse(result["simulator_started"])

    def test_combined_sdk_verification_failure_prevents_auto_python(self):
        def reject(config):
            raise ValueError("Resource set failed")
        result = self.discover(resource_checker=reject)
        self.assertFalse(result["sdk_validated"])
        self.assertIn("isaac_sim", result["missing_keys"])
        self.assertIn("isaac_lab", result["missing_keys"])
        self.assertNotIn("training_python", result["selected"])
        self.assertTrue(any("Resource set failed" in issue for issue in result["issues"]))

    def test_search_is_not_recursive_and_root_count_is_bounded(self):
        self.make_sdk(self.root/"deep/hidden/isaacsim", "isaac_sim")
        result = self.discover(search_roots=[self.root/"deep"])
        self.assertEqual(result["selected"]["isaac_sim"], str(self.sim))
        self.assertFalse(any("hidden" in row["path"] for row in result["candidates"]["isaac_sim"]))
        with self.assertRaisesRegex(ValueError, "32"):
            self.discover(search_roots=[self.root]*33)

    def test_child_scan_reports_truncation(self):
        search = self.root/"bounded search"
        search.mkdir()
        for i in range(70):
            (search/f"unrelated{i:03d}").mkdir()
        result = self.discover(search_roots=[search])
        self.assertTrue(any("truncated at 64" in warning for warning in result["warnings"]))

    def test_unknown_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown path override"):
            self.discover(overrides={"isaac_sm":str(self.sim)})

    def test_git_queries_disable_optional_writes_and_use_timeout(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            class Completed:
                returncode, stderr = 0, ""
                stdout = subject.LAB_COMMIT if command[-2:] == ["rev-parse", "HEAD"] else ""
            return Completed()
        with patch.object(subject.subprocess, "run", side_effect=fake_run):
            result = subject._git_identity(self.lab)
        self.assertEqual(result["head"], subject.LAB_COMMIT)
        self.assertEqual(len(calls), 2)
        for command, kwargs in calls:
            self.assertIn("--no-optional-locks", command)
            self.assertIn("core.fsmonitor=false", command)
            self.assertEqual(kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
            self.assertEqual(kwargs["timeout"], 10)


# Setup project

"""Offline orchestration/installer tests; installer and SDK calls are fakes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from pipeline.setup import candidate_config
from pipeline.setup import install_extras
from pipeline.setup import parser
from pipeline.setup import prepare
from pipeline.setup import save_host
from pipeline.setup import validate_install_target


class SetupProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sim = self.root / 'sim'
        self.lab = self.root / 'lab'
        self.support = self.root / 'existing-support'
        for directory in (self.sim/'kit/python', self.lab, self.support):
            directory.mkdir(parents=True)
        (self.sim/'kit/python/python.exe').write_bytes(b'fixture, never executed')
        self.selected = dict(isaac_sim=str(self.sim), isaac_lab=str(self.lab),
            policy_support=str(self.support), training_python=str(self.sim/'kit/python/python.exe'))
        self.valid = dict(status='PASS', errors=[], warnings=[], video_status='NOT_CONFIGURED')
        self.discovery = dict(selected=self.selected, candidates={}, issues=[],
            missing_keys=[], video_status='NOT_CONFIGURED')
        self.validate = Mock(return_value=self.valid)
        self.preflight = Mock(return_value={'status':'PASS', 'errors':[]})
        self.installer = Mock(side_effect=AssertionError('Unexpected installation'))

    def run_setup(self, *args, discovery=None, system='Windows'):
        return prepare(parser().parse_args(list(args)), repo_root=self.root, system=system,
            discover_fn=Mock(return_value=discovery or self.discovery),
            validate_fn=self.validate, preflight_fn=self.preflight,
            installer=self.installer, interactive=False)

    def snapshot(self):
        return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_dry_run_writes_nothing_even_with_install_flag(self):
        before = self.snapshot()
        result = self.run_setup('--dry-run', '--install-extras')
        self.assertEqual(result['status'], 'READY')
        self.assertEqual(before, self.snapshot())
        self.installer.assert_not_called()

    def test_ready_writes_interoperable_host_and_offline_report(self):
        result = self.run_setup()
        self.assertEqual(result['status'], 'READY')
        config = json.loads((self.root/'.local/host.json').read_text())
        self.assertEqual(config['schema'], 'hc-reproduction-host/v1')
        self.assertEqual(config['training_python'], str(self.sim/'kit/python/python.exe'))
        self.assertEqual(json.loads((self.root/'.local/setup-report.json').read_text())['status'], 'READY')
        self.preflight.assert_called_once_with(config, include_checkpoint=False)

    def test_missing_sdk_never_installs_or_writes_host(self):
        bad = dict(self.discovery, missing_keys=['isaac_sim','policy_support'],
            selected={}, issues=['No valid Isaac Sim'])
        result = self.run_setup('--install-extras', discovery=bad)
        self.assertEqual(result['status'], 'SETUP_INCOMPLETE')
        self.assertFalse((self.root/'.local/host.json').exists())
        self.installer.assert_not_called()

    def test_explicit_invalid_optional_path_blocks_readiness(self):
        result = self.run_setup(discovery=dict(self.discovery,
            missing_keys=['ffmpeg'], issues=['Explicit FFmpeg is invalid']))
        self.assertIn('Explicit FFmpeg is invalid', result['errors'])
        self.assertFalse((self.root/'.local/host.json').exists())

    def test_preflight_failure_blocks_config(self):
        self.preflight.return_value={'status':'FAIL','errors':['Frozen asset mismatch']}
        result = self.run_setup()
        self.assertIn('Frozen asset mismatch', result['errors'])
        self.assertFalse((self.root/'.local/host.json').exists())

    def test_video_is_optional_unless_requested(self):
        self.discovery['video_status'] = 'video_unconfigured'
        result = self.run_setup('--dry-run')
        self.assertEqual(result['status'], 'READY')
        self.assertEqual(result['video_status'], 'NOT_CONFIGURED')
        self.assertEqual(self.run_setup('--dry-run','--require-video')['status'], 'SETUP_INCOMPLETE')

    def test_check_saved_settings_is_read_only(self):
        self.run_setup()
        before = self.snapshot()
        self.assertEqual(self.run_setup('--check')['status'], 'READY')
        self.assertEqual(before, self.snapshot())

    def test_check_missing_settings_does_not_write(self):
        before = self.snapshot()
        self.assertEqual(self.run_setup('--check')['status'], 'SETUP_INCOMPLETE')
        self.assertEqual(before, self.snapshot())

    def test_unsupported_platform_does_not_discover_or_write(self):
        before = self.snapshot()
        self.assertEqual(self.run_setup(system='Linux')['status'], 'UNSUPPORTED_PLATFORM')
        self.assertEqual(before, self.snapshot())
        self.preflight.assert_not_called()

    def test_idempotent_bytes_and_update_backup(self):
        path = self.root/'host.json'
        original = b'{ "schema": "hc-reproduction-host/v1", "custom": 1 }\r\n'
        path.write_bytes(original)
        self.assertEqual(save_host(path, json.loads(original))['action'], 'UNCHANGED')
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'update-config'):
            save_host(path, {'custom':2})
        self.assertEqual(path.read_bytes(), original)
        result = save_host(path, {'custom':2}, update=True)
        self.assertEqual(Path(result['backup']).read_bytes(), original)
        self.assertEqual(json.loads(path.read_text()), {'custom':2})

    def test_unknown_settings_survive_configuration(self):
        result = candidate_config(self.selected, {'custom':'keep', 'checkpoint':'custom.pt'}, self.root)
        self.assertEqual(result['custom'], 'keep')
        self.assertEqual(result['checkpoint'], 'custom.pt')

    def test_explicit_install_rediscovers_and_validates_before_ready(self):
        initial = dict(self.discovery, selected={k:v for k,v in self.selected.items() if k!='policy_support'},
            missing_keys=['policy_support'], issues=['Missing support'])
        target = self.root/'.local/policy_support'
        final = dict(self.discovery, selected=dict(self.selected, policy_support=str(target)))
        discovery = Mock(side_effect=[initial, final])
        installer = Mock(return_value={'status':'INSTALLED','target':str(target)})
        result = prepare(parser().parse_args(['--install-extras']), repo_root=self.root,
            system='Windows', interactive=False, discover_fn=discovery,
            validate_fn=self.validate, preflight_fn=self.preflight, installer=installer)
        self.assertEqual(result['status'], 'READY')
        self.assertTrue(result['software_installed'])
        installer.assert_called_once()
        self.assertEqual(discovery.call_count, 2)
        self.assertEqual(json.loads((self.root/'.local/host.json').read_text())['policy_support'], str(target))

    def test_missing_extras_dry_run_does_not_install(self):
        initial = dict(self.discovery, missing_keys=['policy_support'], issues=['Missing support'])
        before = self.snapshot()
        result = self.run_setup('--install-extras','--dry-run',discovery=initial)
        self.assertEqual(result['status'], 'SETUP_INCOMPLETE')
        self.assertEqual(before,self.snapshot())
        self.installer.assert_not_called()

    def test_installer_promotes_only_after_metadata_validation(self):
        requirements = self.root/'requirements.txt'
        requirements.write_text('example==1.2.3\n')
        calls=[]
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            staging = Path(command[command.index('--target')+1])
            (staging/'example.py').write_text('# fixture')
            return {'success':True}
        checker = Mock(return_value={'status':'PASS','errors':[]})
        target = self.root/'.local/support-new'
        result = install_extras(self.selected, target, repo_root=self.root,
            requirements=requirements, runner=fake_run, support_checker=checker)
        self.assertEqual(result['status'], 'INSTALLED')
        self.assertTrue((target/'example.py').is_file())
        command, options = calls[0]
        self.assertFalse(Path(command[command.index('--target')+1]).exists())
        self.assertIn('--no-deps', command)
        self.assertEqual(command[0], str(self.sim/'kit/python/python.exe'))
        self.assertEqual(options['cleanup_seconds'], 5)
        self.assertEqual(options['timeout_seconds'], 600)
        checker.assert_called_once()

    def test_zero_installer_exit_is_not_sufficient(self):
        requirements = self.root/'requirements.txt'; requirements.write_text('example==1.2.3\n')
        target = self.root/'.local/support-new'
        result = install_extras(self.selected, target, repo_root=self.root, requirements=requirements,
            runner=Mock(return_value={'success':True}),
            support_checker=Mock(return_value={'status':'FAIL','errors':['Wrong version']}))
        self.assertEqual(result['status'], 'INSTALL_FAILED')
        self.assertTrue(Path(result['staging']).is_dir())
        self.assertFalse(target.exists())

    def test_failed_installer_retains_diagnostics_without_promoting(self):
        requirements = self.root/'requirements.txt'; requirements.write_text('example==1.2.3\n')
        checker=Mock()
        result = install_extras(self.selected, self.root/'.local/new', repo_root=self.root,
            requirements=requirements, runner=Mock(return_value={'success':False}), support_checker=checker)
        self.assertEqual(result['status'], 'INSTALL_FAILED')
        self.assertTrue(Path(result['staging']).is_dir())
        checker.assert_not_called()

    def test_installer_refuses_sdk_external_and_existing_targets(self):
        for target in (self.sim/'packages', self.root/'external', self.root/'.local'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_install_target(target, self.selected, self.root)
        target=self.root/'.local/existing'; target.mkdir(parents=True)
        (target/'keep.txt').write_text('keep')
        with self.assertRaises(ValueError): validate_install_target(target, self.selected, self.root)
        self.assertEqual((target/'keep.txt').read_text(), 'keep')

    def test_nested_requirements_refused_before_staging(self):
        requirements = self.root/'requirements.txt'; requirements.write_text('-r another.txt\n')
        with self.assertRaisesRegex(ValueError, 'explicit'):
            install_extras(self.selected, self.root/'.local/new', repo_root=self.root,
                requirements=requirements, runner=Mock())
        self.assertFalse((self.root/'.local').exists())


# Setup bootstrap

"""Verify SDK handoff without invoking that interpreter or importing ML."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from pipeline import runtime as bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sim=self.root/'sim'; (self.sim/'kit/python').mkdir(parents=True)
        self.python=self.sim/'kit/python/python.exe'; self.python.write_bytes(b'fixture')
        self.host=self.root/'host.json'
        self.host.write_text(json.dumps(dict(schema='hc-reproduction-host/v1',
            isaac_sim=str(self.sim), isaac_lab=str(self.root/'lab'),
            policy_support=str(self.root/'support'), checkpoint=str(self.root/'unused.pt'))))
        self.config=self.root/'experiment.json'; self.config.write_text(json.dumps({'host':str(self.host)}))

    def test_help_needs_no_host_or_sdk(self):
        runner=Mock(side_effect=AssertionError())
        self.assertIsNone(bootstrap.handoff('scene',['--help'],runner=runner))
        runner.assert_not_called()

    def test_handoff_preserves_arguments_and_never_adds_execute(self):
        runner=Mock(return_value=Mock(returncode=7))
        arguments=['--config',str(self.config),'--position','HC07','--seed','15345']
        result=bootstrap.handoff('live',arguments,runner=runner,executable=str(self.root/'system.exe'),env={})
        self.assertEqual(result,7)
        command=runner.call_args.args[0]
        self.assertEqual(command[0],str(self.python))
        self.assertEqual(command[-len(arguments):],arguments)
        self.assertNotIn('--execute',command)
        self.assertTrue(command[3].endswith('live_run.py'))
        self.assertEqual(runner.call_args.kwargs['env'][bootstrap.MARKER],str(self.host))

    def test_execute_is_only_preserved_when_explicit(self):
        runner=Mock(return_value=Mock(returncode=0))
        arguments=['--config',str(self.config),'--execute','--output','new-output']
        bootstrap.handoff('record',arguments,runner=runner,executable='other.exe',env={})
        self.assertEqual(runner.call_args.args[0][-len(arguments):],arguments)

    def test_sdk_child_does_not_recurse(self):
        runner=Mock()
        self.assertIsNone(bootstrap.handoff('scene',['--config',str(self.config)],
            runner=runner,executable=str(self.python),env={bootstrap.MARKER:str(self.host)}))
        runner.assert_not_called()

    def test_forged_or_stale_marker_fails_instead_of_looping(self):
        with self.assertRaisesRegex(ValueError,'interpreter'):
            bootstrap.handoff('scene',['--config',str(self.config)],executable='wrong.exe',
                env={bootstrap.MARKER:str(self.host)})

    def test_missing_settings_gives_setup_command(self):
        self.host.unlink()
        with self.assertRaisesRegex(ValueError,'setup_project.py'):
            bootstrap.handoff('scene',['--config',str(self.config)],env={})

    def test_missing_sdk_python_does_not_execute(self):
        self.python.unlink(); runner=Mock()
        with self.assertRaisesRegex(ValueError,'SDK Python is missing'):
            bootstrap.handoff('scene',['--config',str(self.config)],runner=runner,env={})
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
