"""Clean presentation video; no physics/render ownership or policy inputs.

Encoder/verification adapted from original ChicGrasp demo_filming.py Encoder
and verify_video. Campaign bindings and visual-asset substitutions are excluded.
All simulator imports are deferred until camera creation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

VIEWS = ("HAND", "LEFT_HD", "RIGHT_HD")
SIZE = (1280, 960)
FINISH_SECONDS = 4.5


def _new_camera(**kwargs):
    from isaacsim.sensors.camera import Camera
    return Camera(**kwargs)


def _digest(value):
    data=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,default=str).encode()
    return hashlib.sha256(data).hexdigest()


def _identity(path):
    path=Path(path)
    with path.open("rb") as stream: sha=hashlib.file_digest(stream,"sha256").hexdigest()
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha}


def _remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0: raise TimeoutError("Shared media finalization deadline exhausted")
    return value


def validate_tools(ffmpeg_path):
    """Filesystem-only preflight, usable by the parent before Kit starts."""
    ffmpeg=Path(ffmpeg_path).resolve(strict=True)
    ffprobe=ffmpeg.with_name("ffprobe.exe" if ffmpeg.suffix.lower()==".exe" else "ffprobe")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise FileNotFoundError("Configure existing FFmpeg and sibling ffprobe executables")
    return ffmpeg,ffprobe


class _Encoder:
    """One queued frame, raw RGB pipe, owned-job inheritance, finite drain."""
    def __init__(self, executable, path, *, crf, fps):
        self.path=Path(path);self.frames=0;self.error=None;self.terminated=False
        self.cancelled=threading.Event()
        if self.path.exists():raise FileExistsError(self.path)
        self.queue=queue.Queue(maxsize=1)
        self.command=[str(executable),"-hide_banner","-loglevel","error","-nostdin",
            "-f","rawvideo","-pix_fmt","rgb24","-s","1280x960","-r",str(fps),
            "-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),
            "-tune","zerolatency","-pix_fmt","yuv420p","-threads","2","-filter_threads","1",
            "-movflags","+faststart","-n",str(self.path)]
        self.log=self.path.with_suffix(".encoder.log").open("xb",buffering=0)
        try:
            self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                stderr=self.log,bufsize=0,close_fds=True,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except BaseException:
            self.log.close();raise
        self.thread=threading.Thread(target=self._write,name="media-"+self.path.stem,daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while not self.cancelled.is_set():
                try:data=self.queue.get(timeout=.05)
                except queue.Empty:continue
                if data is None:break
                view=memoryview(data)
                while view:
                    count=self.process.stdin.write(view)
                    if not count:raise OSError("Encoder pipe accepted no bytes")
                    view=view[count:]
        except BaseException as error:self.error=repr(error)
        finally:
            try:self.process.stdin.close()
            except OSError as error:self.error=self.error or repr(error)

    def put(self,data,deadline):
        while True:
            if self.error or self.process.poll() is not None:
                raise RuntimeError("Encoder stopped: "+str(self.error))
            try:self.queue.put(data,timeout=min(.05,_remaining(deadline)));break
            except queue.Full:pass
        self.frames+=1

    def request_finish(self,deadline):
        if self.error:raise RuntimeError(self.error)
        while True:
            try:self.queue.put(None,timeout=min(.05,_remaining(deadline)));return
            except queue.Full:
                if not self.thread.is_alive():raise RuntimeError(self.error or "Encoder writer exited")

    def drain(self,deadline):
        self.thread.join(_remaining(deadline))
        if self.thread.is_alive():raise TimeoutError("Encoder writer did not drain")
        code=self.process.wait(timeout=_remaining(deadline))
        if code!=0 or self.error:raise RuntimeError("Encoder exit/error: "+repr((code,self.error)))

    def kill(self):
        self.cancelled.set()
        if self.process.poll() is None:
            self.terminated=True;self.process.kill()  # Exact child handle only.


def verify_video(path, *, ffprobe_path, frames, deadline):
    """Bounded full frame count and metadata read, inside the shared deadline."""
    command=[str(ffprobe_path),"-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_read_frames,avg_frame_rate,pix_fmt,codec_name,duration",
        "-of","json",str(path)]
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        close_fds=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    try:
        stdout,stderr=process.communicate(timeout=_remaining(deadline))
    except BaseException:
        if process.poll() is None:process.kill()
        try:process.communicate(timeout=max(.001,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:pass
        raise
    if process.returncode!=0 or stderr.strip():raise RuntimeError("ffprobe failed: "+stderr.decode(errors="replace")[:2000])
    streams=json.loads(stdout)["streams"]
    if len(streams)!=1:raise ValueError("Expected one video stream")
    stream=streams[0]
    if (stream.get("width"),stream.get("height"))!=SIZE or stream.get("avg_frame_rate")!="30/1":
        raise ValueError("Video size or fps mismatch")
    if (stream.get("codec_name")!="h264" or stream.get("pix_fmt")!="yuv420p"
        or int(stream.get("nb_read_frames",-1))!=frames
        or not math.isfinite(float(stream["duration"]))
        or abs(float(stream["duration"])-frames/30)>1e-3):
        raise ValueError("Video codec, frame count or duration mismatch")
    return {"passed":True,"command":command,"stream":stream,"verification":"ffprobe_count_frames_and_metadata"}


class MediaRecorder:
    """Attach after reset; after_substep never steps or renders the simulator.

    If available, runtime.media_deadline_monotonic shortens every operation to
    the parent's original deadline. Otherwise the owning supervisor remains the
    hard run boundary. finish uses one <=4.5s budget, including all children.
    """
    def __init__(self,runtime,output_dir,*,ffmpeg_path,enabled=True,crf=16,fps=30):
        self.enabled=enabled;self.runtime=runtime;self.output=Path(output_dir)
        self.frames=0;self.errors=[];self.encoders={};self.last_stamps={};self.last_sample=-1
        self.cameras=None;self.created_cameras=[];self.index=None;self.receipt=None;self.crf=crf;self.fps=fps
        if not enabled:return
        if type(enabled)is not bool or type(crf)is not int or not 0<=crf<=51 or fps!=30:
            raise ValueError("Media requires boolean enabled, integer CRF0..51, and 30fps")
        ffmpeg,self.ffprobe=validate_tools(ffmpeg_path)
        visual=getattr(runtime,"demo_chicken_visual_receipt",None)
        if visual is not None and visual.get("version")!="original":
            raise ValueError("Research media requires the original uncapped chicken")
        self.parent_deadline=getattr(runtime,"media_deadline_monotonic",None)
        if self.parent_deadline is not None:
            if not math.isfinite(self.parent_deadline):raise ValueError("Invalid parent media deadline")
            _remaining(self.parent_deadline)
        self.configuration_sha256=_digest(runtime.effective_configuration)
        self.scene_binding=getattr(runtime,"media_scene_binding",None)
        if self.scene_binding is not None and self.scene_binding.get("chicken_version")!="original":
            raise ValueError("Research media scene binding must identify the original chicken")
        self.scene_binding_sha256=_digest(self.scene_binding)
        self.output.mkdir(parents=False,exist_ok=False)
        try:
            from camera_presets.apply_cameras import create_demo_cameras, PRESET_DIR
            before=self._snapshot()
            def camera_factory(**kwargs):
                camera=_new_camera(**kwargs);self.created_cameras.append(camera);return camera
            self.cameras=create_demo_cameras(runtime.stage,enabled=True,camera_factory=camera_factory)
            # Also covers injected factories in offline tests.
            for camera in self.cameras.cameras.values():
                if all(camera is not old for old in self.created_cameras):self.created_cameras.append(camera)
            if set(self.cameras.cameras)!=set(VIEWS):raise ValueError("Exactly HAND, LEFT_HD and RIGHT_HD are required")
            if any(tuple(self.cameras.configs[name]["resolution"])!=SIZE for name in VIEWS):
                raise ValueError("Expected 1280x960 presentation cameras")
            self.preset_identities={name:_identity(PRESET_DIR/name) for name in
                                    ("HAND_DEMO_ONLY.json","HD_TWINS_DEMO_ONLY.json")}
            self._check_unchanged(before)
            self.index=(self.output/"frames.jsonl").open("x",encoding="utf-8",newline="\n")
            for name in VIEWS:self.encoders[name]=_Encoder(ffmpeg,self.output/(name+".mp4"),crf=crf,fps=fps)
        except BaseException as error:
            self.errors.append("initialization:"+repr(error));self.finish();raise

    def _snapshot(self):
        return (_digest(self.runtime.effective_configuration),tuple(self.runtime.clock()),
                _digest(self.runtime.camera_frames()),_digest(getattr(self.runtime,"media_scene_binding",None)))

    def _check_unchanged(self,before):
        if self.cameras is not None:self.cameras.before_render()  # Checks only; no render.
        if self._snapshot()!=before:raise RuntimeError("Media altered policy configuration, acquisition, clock or scene binding")
        if before[0]!=self.configuration_sha256 or before[3]!=self.scene_binding_sha256:
            raise RuntimeError("Bound policy or scene configuration changed during media capture")

    def _deadline(self,seconds):
        value=time.monotonic()+seconds
        return min(value,self.parent_deadline) if self.parent_deadline is not None else value

    def after_substep(self,sample_index):
        if not self.enabled:return
        if self.receipt is not None:raise RuntimeError("Media already finalized")
        if type(sample_index)is not int or sample_index<0 or sample_index<=self.last_sample:
            self.errors.append("capture:scheduled sample index must strictly increase")
            raise ValueError("Scheduled sample index must strictly increase")
        self.last_sample=sample_index
        if sample_index%4!=1:return
        if sample_index!=1+4*self.frames:
            self.errors.append("capture:missing scheduled media sample")
            raise ValueError("Missing scheduled media sample; no silent frame duplication")
        row={"frame_index":self.frames,"sample_index":sample_index,
             "relative_simulation_time_s":(sample_index+1)/120,"streams":{}}
        try:
            before=self._snapshot();self.cameras.before_render()
            prepared={}
            for name in VIEWS:
                frame=self.cameras.cameras[name].get_current_frame()
                stamp=frame.get("rendering_frame");render_time=float(frame["rendering_time"])
                if stamp is None or str(stamp)==self.last_stamps.get(name):raise ValueError("Stale frame: "+name)
                if not math.isfinite(render_time) or abs(render_time-before[1][1])>1e-5:
                    raise ValueError("Frame does not match the already scheduled simulation time: "+name)
                rgb=frame["rgb"]
                if len(rgb.shape)!=3 or tuple(rgb.shape[:2])!=(960,1280) or rgb.shape[2]<3 or str(rgb.dtype)!="uint8":
                    raise ValueError("Invalid RGB frame: "+name)
                data=rgb[:,:,:3].tobytes(order="C")
                if len(data)!=1280*960*3:raise ValueError("Wrong RGB byte count: "+name)
                prepared[name]=(data,stamp)
                row["streams"][name]={"rendering_frame":str(stamp),"rendering_time":render_time,"resolution":[1280,960]}
            self._check_unchanged(before)
            deadline=self._deadline(2)
            for name,(data,stamp) in prepared.items():
                self.encoders[name].put(data,deadline);self.last_stamps[name]=str(stamp)
            self.frames+=1
        except BaseException as error:
            row["error"]=repr(error);self.errors.append("capture:"+repr(error));raise
        finally:
            if self.index is not None:
                self.index.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n");self.index.flush()

    def finish(self):
        if self.receipt is not None:return self.receipt
        if not self.enabled:
            self.receipt={"schema":"research-media/v1","enabled":False,"status":"DISABLED","videos":{}}
            return self.receipt
        started=time.monotonic();deadline=self._deadline(FINISH_SECONDS)
        drain_deadline=min(deadline,started+3.5)
        videos={};finalized=True
        try:
            for encoder in self.encoders.values():encoder.request_finish(drain_deadline)
            for encoder in self.encoders.values():encoder.drain(drain_deadline)
        except BaseException as error:self.errors.append("drain:"+repr(error));finalized=False
        finally:
            # Kill all still-active exact children before sharing the remaining
            # cleanup time, rather than spending a fresh timeout per stream.
            for encoder in self.encoders.values():
                if not finalized or encoder.process.poll() is None:
                    try:encoder.kill()
                    except OSError as error:self.errors.append("kill:"+repr(error))
            for name,encoder in self.encoders.items():
                try:
                    if encoder.process.poll() is None:encoder.process.wait(timeout=_remaining(deadline))
                    encoder.thread.join(max(0,deadline-time.monotonic()))
                except BaseException as error:self.errors.append("cleanup:"+name+":"+repr(error));finalized=False
                encoder.log.close()
                videos[name]={"path":str(encoder.path),"pid":encoder.process.pid,"frames":encoder.frames,
                    "returncode":encoder.process.poll(),"forced_termination":encoder.terminated,
                    "writer_finished":not encoder.thread.is_alive(),"command":encoder.command,"error":encoder.error}
                if encoder.process.poll() is None or encoder.thread.is_alive() or encoder.error or encoder.terminated:finalized=False
            try:
                before=self._snapshot()
                self._check_unchanged(before)
            except BaseException as error:self.errors.append("final_camera_check:"+repr(error));finalized=False
            destroyed=0
            for camera in self.created_cameras:
                try:
                    camera.pause();camera.destroy();destroyed+=1
                except BaseException as error:self.errors.append("camera_cleanup:"+repr(error));finalized=False
            try:
                if self._snapshot()!=before:raise RuntimeError("Camera cleanup altered policy state or simulation clock")
            except BaseException as error:self.errors.append("camera_cleanup_state:"+repr(error));finalized=False
            if self.index is not None:self.index.close()
        for name,video in videos.items():
            if finalized and not self.errors and self.frames>0:
                try:
                    video["verification"]=verify_video(video["path"],ffprobe_path=self.ffprobe,frames=self.frames,deadline=deadline)
                    _remaining(deadline);video["media"]=_identity(video["path"])
                except BaseException as error:self.errors.append("verification:"+name+":"+repr(error))
        elapsed=time.monotonic()-started
        if elapsed>FINISH_SECONDS:self.errors.append("shared_finish_budget_overrun")
        if self.parent_deadline is not None and time.monotonic()>self.parent_deadline:self.errors.append("parent_deadline_exceeded")
        passed=(finalized and not self.errors and self.frames>0 and set(videos)==set(VIEWS)
                and all(v.get("verification",{}).get("passed") and v["frames"]==self.frames for v in videos.values()))
        self.receipt={"schema":"research-media/v1","enabled":True,"status":"PASS" if passed else "FAIL",
            "presentation_only":True,"policy_observation_member":False,"burned_in_text":False,
            "chicken_visual_version":"original","scene_binding":self.scene_binding,
            "scene_binding_sha256":self.scene_binding_sha256,"scene_binding_supplied":self.scene_binding is not None,
            "policy_configuration_sha256":self.configuration_sha256,"camera_presets":getattr(self,"preset_identities",{}),
            "views":list(VIEWS),"resolution":[1280,960],"fps":30,"crf":self.crf,"frames":self.frames,
            "sample_rule":"120Hz scheduled substep index modulo4 equals1","extra_simulation_steps":0,
            "extra_render_calls":0,"transient_image_files":0,"videos":videos,"errors":self.errors,
            "presentation_render_products_destroyed":destroyed,
            "finish_elapsed_seconds":elapsed,"finish_budget_seconds":FINISH_SECONDS,
            "parent_deadline_supplied":self.parent_deadline is not None,
            "composite":{"status":"NOT_REQUESTED","overview_substitution":False}}
        with (self.output/"MEDIA_RECEIPT.json").open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(self.receipt,stream,indent=2,allow_nan=False);stream.write("\n")
        return self.receipt
