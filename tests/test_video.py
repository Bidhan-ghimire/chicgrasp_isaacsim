"""Offline regression tests grouped by video."""
from __future__ import annotations



# Research media

"""Offline cameras/pipe fakes only; no Kit, model, FFmpeg or ffprobe runs."""
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from pipeline import video as media

FRAME_BYTES=1280*960*3


class Frame:
    shape=(960,1280,3)
    dtype="uint8"
    data=b"\x10\x20\x30"*(1280*960)
    def __getitem__(self,key):return self
    def tobytes(self,order="C"):return self.data


class Runtime:
    def __init__(self):
        self.stage=object();self.sample=0
        self.effective_configuration={"policy_camera_resolution":[320,240],"physics":"unchanged"}
        self.media_scene_binding={"chicken_version":"original","scene_profile_sha256":"4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f"}
        self.media_deadline_monotonic=time.monotonic()+100
    def clock(self):return self.sample,(self.sample+1)/120
    def camera_frames(self):return {"policy_left":100,"policy_right":100}


class Camera:
    def __init__(self,runtime):self.runtime=runtime;self.stamp_override=None;self.image=Frame();self.destroyed=False;self.paused=False
    def get_current_frame(self):return {"rendering_frame":self.runtime.sample if self.stamp_override is None else self.stamp_override,
        "rendering_time":self.runtime.clock()[1],"rgb":self.image}
    def pause(self):self.paused=True
    def destroy(self):self.destroyed=True


class Cameras:
    def __init__(self,runtime):
        self.cameras={name:Camera(runtime) for name in media.VIEWS}
        self.configs={name:{"resolution":[1280,960]} for name in media.VIEWS}
        self.checks=0;self.pauses=0
    def before_render(self):self.checks+=1
    def pause(self):self.pauses+=1


class Sink:
    def __init__(self,process):self.process=process;self.bytes=0;self.closed=False
    def write(self,data):
        if self.closed:raise BrokenPipeError("closed test pipe")
        self.bytes+=len(data);return len(data)
    def close(self):
        self.closed=True
        if not self.process.hang:self.process.returncode=0


class Process:
    def __init__(self,factory,command,kwargs):
        self.factory=factory;self.command=command;self.kwargs=kwargs
        self.pid=1000+len(factory.calls);self.returncode=None;self.hang=factory.hang;self.killed=False
        self.probe="-show_entries" in command
        if not self.probe:
            self.stdin=Sink(self);Path(command[-1]).write_bytes(b"fake h264 artifact\x00")
            factory.encoders[Path(command[-1]).name]=self
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        if self.returncode is None:raise subprocess.TimeoutExpired(self.command,timeout)
        return self.returncode
    def kill(self):self.killed=True;self.returncode=-9
    def communicate(self,timeout=None):
        if not self.probe:raise AssertionError("Only probes use communicate")
        encoder=self.factory.encoders[Path(self.command[-1]).name]
        frames=encoder.stdin.bytes//FRAME_BYTES
        if self.factory.bad_probe:frames+=1
        self.returncode=0
        return json.dumps({"streams":[{"width":1280,"height":960,"avg_frame_rate":"30/1",
            "pix_fmt":"yuv420p","codec_name":"h264","nb_read_frames":str(frames),"duration":str(frames/30)}]}).encode(),b""


class Factory:
    def __init__(self,hang=False,bad_probe=False):self.calls=[];self.encoders={};self.hang=hang;self.bad_probe=bad_probe
    def __call__(self,command,**kwargs):
        self.calls.append((command,kwargs));return Process(self,command,kwargs)


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(tempfile.mkdtemp(prefix="research_media_",dir=os.environ.get("CHICGRASP_MEDIA_TEST_ROOT")))
        cls.ffmpeg=cls.root/"ffmpeg.exe";cls.ffmpeg.write_bytes(b"offline tool placeholder")
        (cls.root/"ffprobe.exe").write_bytes(b"offline tool placeholder")

    def setUp(self):
        self.runtime=Runtime();self.cameras=Cameras(self.runtime);self.factory=Factory()
        self.path=self.root/self._testMethodName
        self.camera_patch=mock.patch("camera_presets.apply_cameras.create_demo_cameras",return_value=self.cameras)
        self.create=self.camera_patch.start();self.addCleanup(self.camera_patch.stop)
        self.process_patch=mock.patch.object(media.subprocess,"Popen",side_effect=lambda *a,**kw:self.factory(*a,**kw))
        self.popen=self.process_patch.start();self.addCleanup(self.process_patch.stop)

    def recorder(self,**kwargs):return media.MediaRecorder(self.runtime,self.path,ffmpeg_path=self.ffmpeg,**kwargs)

    def sample(self,recorder,index):self.runtime.sample=index;recorder.after_substep(index)

    def test_import_and_disabled_mode_do_not_load_native_or_start_processes(self):
        original_import=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {"pxr","isaacsim","omni","torch"}:raise AssertionError("Forbidden runtime import")
            return original_import(name,*args,**kwargs)
        with mock.patch("builtins.__import__",side_effect=guarded):
            spec=importlib.util.spec_from_file_location("media_import_probe",Path(media.__file__))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            r=module.MediaRecorder(None,self.path,ffmpeg_path="missing",enabled=False)
            self.assertEqual(r.finish()["status"],"DISABLED")
        self.create.assert_not_called();self.popen.assert_not_called();self.assertFalse(self.path.exists())

    def test_missing_tools_fail_before_camera_creation(self):
        with self.assertRaises(FileNotFoundError):media.MediaRecorder(self.runtime,self.path,ffmpeg_path=self.root/"missing.exe")
        self.create.assert_not_called();self.popen.assert_not_called();self.assertFalse(self.path.exists())

    def test_original_scene_only(self):
        self.runtime.demo_chicken_visual_receipt={"version":"capped_v1"}
        with self.assertRaisesRegex(ValueError,"uncapped"):self.recorder()
        self.create.assert_not_called();self.popen.assert_not_called()

    def test_clean_three_views_exact_schedule_and_hashes(self):
        r=self.recorder()
        baseline=json.dumps(self.runtime.effective_configuration,sort_keys=True)
        for index in (1,3,5,7):self.sample(r,index)
        receipt=r.finish()
        self.assertEqual(receipt["status"],"PASS",receipt["errors"])
        self.assertEqual(receipt["views"],["HAND","LEFT_HD","RIGHT_HD"])
        self.assertEqual(receipt["frames"],2)
        self.assertFalse(receipt["burned_in_text"])
        self.assertEqual(receipt["extra_render_calls"],0)
        self.assertEqual(receipt["extra_simulation_steps"],0)
        self.assertEqual(receipt["presentation_render_products_destroyed"],3)
        self.assertTrue(all(c.destroyed and c.paused for c in self.cameras.cameras.values()))
        self.assertEqual(json.dumps(self.runtime.effective_configuration,sort_keys=True),baseline)
        self.assertEqual(self.runtime.camera_frames(),{"policy_left":100,"policy_right":100})
        self.assertEqual(self.runtime.sample,7)
        for name,item in receipt["videos"].items():
            self.assertTrue(item["verification"]["passed"])
            self.assertEqual(item["media"]["sha256"],hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
            self.assertEqual(self.factory.encoders[name+".mp4"].stdin.bytes,2*FRAME_BYTES)
            self.assertNotIn("drawtext",str(item["command"]))
            self.assertEqual(item["command"][item["command"].index("-crf")+1],"16")
        for _,kwargs in self.factory.calls:
            self.assertFalse(kwargs.get("creationflags",0)&0x01000000)  # no BREAKAWAY_FROM_JOB
        rows=[json.loads(line) for line in (self.path/"frames.jsonl").read_text().splitlines()]
        self.assertEqual([row["sample_index"] for row in rows],[1,5])
        calls=len(self.factory.calls);self.assertEqual(r.finish(),receipt);self.assertEqual(len(self.factory.calls),calls)
        self.assertEqual(json.loads((self.path/"MEDIA_RECEIPT.json").read_text()),receipt)

    def test_stale_frame_fails_and_is_not_duplicated(self):
        r=self.recorder();self.sample(r,1)
        self.cameras.cameras["RIGHT_HD"].stamp_override=1
        with self.assertRaisesRegex(ValueError,"Stale"):self.sample(r,5)
        receipt=r.finish();self.assertEqual(receipt["status"],"FAIL");self.assertEqual(receipt["frames"],1)
        self.assertTrue(any("Stale" in e for e in receipt["errors"]))

    def test_missing_scheduled_frame_is_not_silently_skipped(self):
        r=self.recorder();self.sample(r,1)
        with self.assertRaisesRegex(ValueError,"Missing"):self.sample(r,9)
        self.assertEqual(r.finish()["status"],"FAIL")

    def test_wrong_image_shape_fails(self):
        r=self.recorder();wrong=Frame();wrong.shape=(240,320,3);self.cameras.cameras["HAND"].image=wrong
        with self.assertRaisesRegex(ValueError,"Invalid RGB"):self.sample(r,1)
        self.assertEqual(r.finish()["status"],"FAIL")

    def test_policy_mutation_is_detected_before_encoding(self):
        def mutate(*args,**kwargs):
            self.runtime.effective_configuration["physics"]="changed";return self.cameras
        self.create.side_effect=mutate
        with self.assertRaisesRegex(RuntimeError,"altered"):self.recorder()
        self.popen.assert_not_called()
        self.assertEqual(json.loads((self.path/"MEDIA_RECEIPT.json").read_text())["status"],"FAIL")

    def test_hung_children_are_killed_under_one_shared_budget(self):
        self.factory.hang=True;r=self.recorder();self.sample(r,1)
        receipt=r.finish();self.assertEqual(receipt["status"],"FAIL")
        self.assertTrue(all(p.killed for p in self.factory.encoders.values()))
        self.assertTrue(all(v["forced_termination"] for v in receipt["videos"].values()))
        self.assertLess(receipt["finish_elapsed_seconds"],4.5)

    def test_incorrect_probe_frame_count_fails(self):
        self.factory.bad_probe=True;r=self.recorder();self.sample(r,1)
        receipt=r.finish();self.assertEqual(receipt["status"],"FAIL")
        self.assertTrue(any("frame count" in e for e in receipt["errors"]))

    def test_existing_output_is_not_overwritten(self):
        self.path.mkdir();sentinel=self.path/"keep";sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):self.recorder()
        self.create.assert_not_called();self.popen.assert_not_called();self.assertEqual(sentinel.read_bytes(),b"keep")

    def test_expired_parent_deadline_prevents_camera_creation(self):
        self.runtime.media_deadline_monotonic=time.monotonic()-1
        with self.assertRaises(TimeoutError):self.recorder()
        self.create.assert_not_called();self.popen.assert_not_called()

    def test_partial_camera_factory_failure_destroys_created_sensor(self):
        created=Camera(self.runtime)
        def partial(stage,*,enabled,camera_factory):
            camera_factory(prim_path="fake",name="HAND")
            raise ValueError("injected second-camera initialization failure")
        self.create.side_effect=partial
        with mock.patch.object(media,"_new_camera",return_value=created):
            with self.assertRaisesRegex(ValueError,"second-camera"):self.recorder()
        self.assertTrue(created.destroyed and created.paused)
        self.popen.assert_not_called()
        receipt=json.loads((self.path/"MEDIA_RECEIPT.json").read_text())
        self.assertEqual(receipt["status"],"FAIL")
        self.assertEqual(receipt["presentation_render_products_destroyed"],1)

    def test_capped_scene_binding_is_rejected_before_camera_creation(self):
        self.runtime.media_scene_binding={"chicken_version":"capped_v1"}
        with self.assertRaisesRegex(ValueError,"original chicken"):self.recorder()
        self.create.assert_not_called();self.popen.assert_not_called()

    def test_partial_encoder_failure_cleans_prior_encoder_and_all_cameras(self):
        original=self.factory
        def fail_second(command,**kwargs):
            if Path(command[-1]).name=="LEFT_HD.mp4":raise OSError("injected second encoder failure")
            return original(command,**kwargs)
        self.popen.side_effect=fail_second
        with self.assertRaisesRegex(OSError,"second encoder"):self.recorder()
        self.assertTrue(all(c.destroyed and c.paused for c in self.cameras.cameras.values()))
        self.assertIsNotNone(original.encoders["HAND.mp4"].poll())
        self.assertEqual(json.loads((self.path/"MEDIA_RECEIPT.json").read_text())["status"],"FAIL")


# Camera presets

"""Pure offline checks. No pxr, Kit, CUDA, model, or original-source access."""
import math
import unittest

import camera_presets.apply_cameras as module


class Attr:
    def __init__(self, name, value=None, readonly=False):
        self.name, self.value, self.readonly = name, value, readonly
    def GetName(self): return self.name
    def Get(self): return self.value
    def GetTypeName(self): return "type"
    def IsCustom(self): return False
    def GetVariability(self): return "varying"
    def Set(self, value):
        if self.readonly:
            raise AssertionError("Attempted policy-camera mutation")
        self.value = value


class Prim:
    def __init__(self, path, values=None, readonly=False, schemas=None):
        self.path, self.readonly, self.schemas = path, readonly, schemas or []
        self.attrs = {k: Attr(k, v, readonly) for k, v in (values or {}).items()}
        self.metadata = {"apiSchemas": "CameraLensAPI"}
    def GetPath(self): return self.path
    def GetAppliedSchemas(self): return self.schemas
    def GetMetadata(self, key): return self.metadata.get(key)
    def SetMetadata(self, key, value):
        if self.readonly:
            raise AssertionError("Attempted policy-camera metadata mutation")
        self.metadata[key] = value
    def GetAttributes(self): return list(self.attrs.values())
    def CreateAttribute(self, name, *args):
        if self.readonly:
            raise AssertionError("Attempted policy-camera attribute creation")
        self.attrs.setdefault(name, Attr(name))
        return self.attrs[name]


class CameraPresetTests(unittest.TestCase):
    def test_disabled_creation_never_imports_runtime(self):
        # No pxr or Kit is installed in the offline test interpreter.
        self.assertIsNone(module.create_demo_cameras())

    def test_approved_values_and_scope(self):
        hand, hd = module.load_presets()
        for preset in (hand, hd):
            self.assertFalse(preset["enabled_by_default"])
            self.assertFalse(preset["policy_observation_member"])
            self.assertFalse(preset["changes_chicken_or_materials"])
        camera = hand["camera"]
        self.assertEqual(camera["local_position_m"],
                         [0.011170286303477273, 0.07500160828567476, -0.2599524804273602])
        self.assertEqual(camera["raw_optics"]["focalLength"], 0.05000000074505806)
        self.assertEqual(camera["raw_optics"]["horizontalAperture"], 0.699999988079071)
        self.assertFalse(camera["aspect_ratio_correction"])
        self.assertEqual(hand["provenance"]["approval"]["sha256"],
                         "05170ed16baf34cba68ae2dfe1d81045cb287a0bc0934ac83967e746b8124759")
        for name, side in (("LEFT_HD", "Left"), ("RIGHT_HD", "Right")):
            self.assertEqual(hd["views"][name]["copy_from"], "/World/envs/env_0/PolicyCamera" + side)
            self.assertEqual(hd["views"][name]["resolution"], [1280, 960])
            self.assertEqual(hd["views"][name]["pixel_K_scale"], 4)

    def test_hand_scale_and_parent_composition_are_preserved(self):
        hand, _ = module.load_presets()
        local = hand["camera"]["local_matrix_row_major"]
        parent = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [3, 4, 5, 1]]
        world = module.compose_world_matrix(local, parent)
        x, y, z = local[3][:3]
        self.assertEqual(world[3], [3-y, 4+x, 5+z, 1.0])
        for row in world[:3]:
            self.assertAlmostEqual(math.sqrt(sum(x*x for x in row[:3])), 0.9200000166893005)
        self.assertNotEqual(world, module.compose_world_matrix(parent, local))

    def test_clone_never_mutates_policy_or_copies_transform(self):
        source = Prim("/Policy", {"focalLength": 15.0, "horizontalAperture": 20.955,
                                 "xformOp:translate": [1, 2, 3]}, readonly=True)
        before = {a.GetName(): a.Get() for a in source.GetAttributes()}
        target = Prim("/Twin")
        module.copy_camera_attributes(source, target)
        self.assertEqual(target.attrs["focalLength"].Get(), 15.0)
        self.assertNotIn("xformOp:translate", target.attrs)
        self.assertEqual(before, {a.GetName(): a.Get() for a in source.GetAttributes()})
        with self.assertRaises(ValueError):
            module.copy_camera_attributes(source, source)

    def test_physics_schemas_are_rejected(self):
        source = Prim("/Policy", schemas=["PhysicsRigidBodyAPI"])
        target = Prim("/Twin")
        with self.assertRaises(ValueError):
            module.copy_camera_attributes(source, target)
        self.assertEqual(target.attrs, {})

    def test_nonfinite_and_nonaffine_transforms_are_rejected(self):
        with self.assertRaises(ValueError):
            module.checked_matrix([[float("nan")]*4]*4)
        with self.assertRaises(ValueError):
            module.checked_matrix([[1,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]])


if __name__ == "__main__":
    unittest.main()
