"""Offline regression tests grouped by training."""
from __future__ import annotations



# Research training

"""CPU synthetic contract tests. No Torch, simulator, model loading or training."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from pipeline import training
from pipeline import training as data
from pipeline import models


def arrays(length=200, color=7):
    action = np.zeros((length, 5), dtype=np.float32)
    action[:, 0] = .65 + np.arange(length, dtype=np.float32) * .00001
    action[:, 1] = .01
    action[:, 2] = .2
    action[:, 3:] = 1
    arm = np.zeros((length, 6), dtype=np.float32)
    arm[:, 0] = np.arange(length)
    rgb = np.broadcast_to(np.uint8(color), (length, 240, 320, 3))
    return {"timestamp_ns": np.arange(length, dtype=np.int64) * 100_000_000,
            "camera_left_rgb": rgb, "camera_right_rgb": rgb,
            "arm_joint_pos": arm, "jaw_state": np.ones((length, 2), dtype=np.uint8), "action": action}


def manifest_row(name, split):
    scene = {"schema":"research-scene-setup/v1", "base_scene_sha256":"b"*64,
             "source":{"path":"scene_setup.py", "bytes":1, "sha256":"c"*64},
             "options":{}, "definition":{"id":"test-scene", "kind":"table"}, "assets":[], "bridge_files":[]}
    return {"id": name, "path": name + "/episode", "split": split, "position_id": name,
            "script_sha256": "a" * 64, "scene_sha256": "b" * 64, "complete": True,
            "scene_setup_identity":scene,
            "episode_binding":{"path":name + "/episode_binding.json", "bytes":1, "sha256":"d"*64}}


class TrainingContractTests(unittest.TestCase):
    def test_alignment_original_150_actions_exactly_once(self):
        source = arrays()
        selected = []
        for start in range(50, 200, 12):
            indices = data.window_indices(200, start, 50)
            batch = data.make_window(source, start, 50)
            self.assertEqual(batch["obs"]["arm_joint_pos"][0, :, 0].tolist(), [start - 1, start])
            selected.extend(indices["source_indices"][indices["loss_mask"]].tolist())
            self.assertEqual(batch["action"][0, 1].tolist(), source["action"][start].tolist())
            self.assertEqual(batch["obs"]["camera_left_rgb"].shape, (1, 2, 3, 240, 320))
        self.assertEqual(selected, list(range(50, 200)))

    def test_short_tail_and_start_zero_preserve_slots(self):
        first = data.window_indices(13, 0, 0)
        self.assertEqual(first["observation_indices"].tolist(), [0, 0])
        tail = data.window_indices(13, 12, 0)
        self.assertEqual(np.flatnonzero(tail["loss_mask"]).tolist(), [1])
        self.assertEqual(tail["source_indices"][1], 12)
        self.assertFalse(tail["valid_mask"][2:].any())

    def test_rgb_and_extra_key_rejection(self):
        source = arrays(2, 255)
        history = [{key: source[key][i] for key in data.OBS_KEYS} for i in range(2)]
        packed = data.pack_history(history)
        self.assertEqual(packed["camera_left_rgb"].dtype, np.float32)
        self.assertTrue((packed["camera_left_rgb"] == 1).all())
        history[0]["chicken_pose"] = [0, 0, 0]
        with self.assertRaises(ValueError): data.pack_history(history)

    def test_train_only_normalizer_no_test_files_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [manifest_row("train_a", "train"), manifest_row("val_a", "validation"), manifest_row("missing_test", "test")]
            filename = root / "dataset.json"
            payloads = {}
            for name, color in (("train_a", 2), ("val_a", 250)):
                episode = root / name / "episode"
                (episode / "policy").mkdir(parents=True)
                payloads[name] = arrays(4, color)
                for key, value in payloads[name].items(): np.save(episode / "policy" / (key + ".npy"), value)
                (episode / "episode_end.json").write_text('{"schema_version":2}')
                (episode / "manifest.json").write_text("{}")
                binding_path = episode.parent / "episode_binding.json"
                row = next(r for r in records if r['id'] == name)
                binding_path.write_text(json.dumps({'schema':'research-episode-binding/v1',
                    'scene_setup_identity':row['scene_setup_identity'], 'episode_manifest':data.identity(episode/'manifest.json')}))
                row['episode_binding'].update(data.identity(binding_path))
            filename.write_text(json.dumps({"schema": "research-dataset/v1", "episodes": records}))
            calls = []
            def checked(path):
                calls.append(Path(path).parent.name)
                return payloads[Path(path).parent.name]
            with patch.object(data, "frozen_module", return_value=types.SimpleNamespace(load_policy_episode=checked)):
                dataset = data.Dataset(filename, training_start=0)
            self.assertEqual(calls, ["train_a", "val_a"])
            values = dataset.normalizer_rows()
            self.assertEqual(len(values["action"]), 4)
            self.assertEqual(values["camera_left_rgb"].mean(), 2)
            batch, selected = dataset.sample(np.random.default_rng(42), 3)
            self.assertEqual({s["episode_id"] for s in selected}, {"train_a"})
            self.assertEqual(batch["action"].shape, (3, 16, 5))
            # Close Windows memory maps before temporary-directory cleanup.
            for row in dataset.rows:
                for value in row["arrays"].values(): value._mmap.close()

    def test_config_distinguishes_recipe_and_requires_finite_values(self):
        cfg = training.resolve_config({"training_window_start": 50}, 10000, "cuda:0")
        self.assertEqual(cfg["warmup_steps"], 500)
        self.assertEqual(cfg["recipe"], "research-aligned12/v1")
        with self.assertRaises(ValueError): training.resolve_config({"training_window_stride": 1}, 10000, "cuda:0")
        with self.assertRaises(ValueError): training.resolve_config({"learning_rate": float("nan")}, 10000, "cpu")

    def test_manifest_rejects_escape_and_plan_does_not_load_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "dataset.json"
            row = manifest_row("absent_train", "train")
            filename.write_text(json.dumps({"schema": "research-dataset/v1", "episodes": [row]}))
            plan = training.training_plan(filename, model_type="bc_cnn")
            self.assertEqual(plan["checkpoint_selection"], "FINAL_EMA")
            self.assertFalse(plan["original_142858_recipe_equivalent"])
            row["path"] = "../escape"
            filename.write_text(json.dumps({"schema": "research-dataset/v1", "episodes": [row]}))
            with self.assertRaises(ValueError): data.read_manifest(filename)

    def test_both_registry_ids_and_no_model_imports(self):
        self.assertEqual(tuple(models.ADAPTERS), ("diffusion", "bc_cnn"))
        prohibited = {"torch", "torchvision", "diffusers", "isaacsim", "isaaclab", "omni", "carb", "pxr"}
        self.assertFalse(prohibited & {key.split('.')[0] for key in sys.modules})

    def test_sidecar_inspection_and_tamper_rejection_no_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"not a serialized model; identity-only test")
            norm = root / "normalization.json"
            norm.write_text("{}")
            artifact = {"schema": "research-model-artifact/v1", "model_type": "diffusion", "policy_contract": data.CONTRACT,
                        "model_config": {"training_window_start": 50}, "training_window_start": 50,
                        "live_compatible": True, "checkpoint_selection": "FINAL_EMA", "completed_updates": 1,
                        "script_options": {},
                        "scene_setup_identity":manifest_row('a','train')['scene_setup_identity'],
                        "identities":{"episodes":[manifest_row('a','train')]},
                        "checkpoint_path": "model.pt", "checkpoint": data.identity(checkpoint),
                        "normalizer_path": "normalization.json", "normalizer": data.identity(norm)}
            (root / "artifact.json").write_text(json.dumps(artifact))
            self.assertEqual(models.inspect_checkpoint(checkpoint)["model_type"], "diffusion")
            artifact["training_window_start"] = 0
            (root / "artifact.json").write_text(json.dumps(artifact))
            with self.assertRaises(ValueError): models.inspect_checkpoint(checkpoint)
            artifact["training_window_start"] = 50
            (root / "artifact.json").write_text(json.dumps(artifact))
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError): models.inspect_checkpoint(checkpoint)

    def test_options_provenance_rejects_mixed_options_before_model_loading(self):
        rows = [manifest_row("a", "train"), manifest_row("b", "validation"), manifest_row("c", "test")]
        rows[0]["script_options"] = {"lift_extra_m": .01}
        rows[1]["script_options"] = {"lift_extra_m": .01}
        rows[2]["script_options"] = {"lift_extra_m": .05}
        self.assertEqual(data.consistent_script_options(rows), {"lift_extra_m": .01})
        rows[1]["script_options"]["lift_extra_m"] = .02
        with self.assertRaises(ValueError): data.consistent_script_options(rows)

    def test_third_adapter_dispatch_uses_same_training_and_prediction_registry(self):
        calls = []
        class FakeThird:
            def __init__(self, model_type, **kwargs): calls.append(("create", model_type, kwargs))
            @classmethod
            def load(cls, path, *, metadata, device):
                calls.append(("load", metadata["model_type"], device))
                return cls(metadata["model_type"])
            def predict(self, history, *, seed):
                calls.append(("predict", seed))
                return np.zeros((16, 5), dtype=np.float32)
            def reset(self): pass
        module = types.ModuleType("synthetic_third_adapter")
        module.FakeThird = FakeThird
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(models.ADAPTERS, {"third": "synthetic_third_adapter:FakeThird"}):
            models.create_adapter("third", normalizer={}, config={"model_options": {"width": 7}}, device="cpu", updates=2, seed=42)
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "model.pt").write_bytes(b"synthetic checkpoint")
                (root / "normalization.json").write_text("{}")
                artifact = {"schema": "research-model-artifact/v1", "model_type": "third", "policy_contract": data.CONTRACT,
                    "model_config": {"training_window_start": 50, "model_options": {"width": 7}}, "training_window_start": 50,
                    "live_compatible": True, "checkpoint_selection": "FINAL_EMA", "completed_updates": 2, "script_options": {},
                    "scene_setup_identity":manifest_row('a','train')['scene_setup_identity'],
                    "identities":{"episodes":[manifest_row('a','train')]},
                    "checkpoint_path": "model.pt", "checkpoint": data.identity(root / "model.pt"),
                    "normalizer_path": "normalization.json", "normalizer": data.identity(root / "normalization.json")}
                (root / "artifact.json").write_text(json.dumps(artifact))
                predict = models.load_predictor(root / "model.pt", device="cpu", evaluation_seed=100)
                self.assertEqual(predict(({}, {}), 2).shape, (16, 5))
        self.assertIn(("load", "third", "cpu"), calls)
        self.assertIn(("predict", 124), calls)

    def _fake_training(self, root, *, fail=False):
        """Exercise real orchestration with explicit data/model/Torch substitutes."""
        dataset_path = root / "dataset.json"
        row = manifest_row("train_a", "train")
        dataset_path.write_text(json.dumps({"schema": "research-dataset/v1", "episodes": [row]}))
        source = arrays(4)
        class FakeData:
            def __init__(self, path, **kwargs):
                self.training_start = 0
                self.train = [row]
                self.scene_setup_identity = row['scene_setup_identity']
                self.identities = {"dataset": data.identity(path), "episodes": [row]}
            def unchanged(self): pass
            def normalizer_rows(self): return source
            def sample(self, rng, count):
                return data.make_window(source, 0, 0), [{"episode_id": "train_a", "target_start": 0}]
        class FakeAdapter:
            def __init__(self): self.count = 0
            def train_step(self, batch):
                self.count += 1
                if fail and self.count == 2: raise ValueError("synthetic failure")
                return {"loss": .25, "optimizer_updates": self.count}
            def save(self, path, metadata, **kwargs): Path(path).write_bytes(b"synthetic checkpoint")
        fit_calls = []
        def fit(values, **kwargs):
            fit_calls.append(len(values["action"]))
            return None, {"fields": {}, "fit_source": kwargs["fit_source"]}
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(training, "Dataset", FakeData), \
                patch.object(training, "frozen_module", return_value=types.SimpleNamespace(_fit_normalizer_from_validated_policy_rows=fit)), \
                patch.object(models, "create_adapter", return_value=FakeAdapter()):
            if fail:
                with self.assertRaisesRegex(ValueError, "synthetic failure"):
                    training.run_training(dataset_path, root / "run", model_type="bc_cnn", updates=2, device="cpu",
                                          config={"training_window_start": 0, "checkpoint_every": 1})
            else:
                result = training.run_training(dataset_path, root / "run", updates=2, device="cpu",
                                               config={"training_window_start": 0, "checkpoint_every": 1})
                self.assertEqual(result["status"], "TRAINING_FINISHED")
                self.assertEqual(fit_calls, [4])
        return root / "run"

    def test_stubbed_orchestration_produces_verifiable_final_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            output = self._fake_training(Path(folder))
            artifact = models.inspect_checkpoint(output / "model.pt")
            self.assertEqual(artifact["checkpoint_selection"], "FINAL_EMA")
            self.assertEqual(artifact["completed_updates"], 2)
            self.assertFalse(artifact["live_compatible"])
            self.assertEqual(artifact["validation_evaluation"], "NOT_RUN")
            expected = manifest_row('train_a','train')['scene_setup_identity']
            self.assertEqual(artifact['scene_setup_identity'], expected)
            self.assertEqual(artifact['identities']['episodes'][0]['scene_setup_identity'], expected)
            self.assertEqual(json.loads((output/'plan.json').read_text())['scene_setup_identity'], expected)
            self.assertEqual(json.loads((output/'normalization.json').read_text())['scene_setup_identity'], expected)
            with self.assertRaises(ValueError): models.load_predictor(output / "model.pt")

    def test_stubbed_failure_preserves_report_without_publishing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            output = self._fake_training(Path(folder), fail=True)
            result = json.loads((output / "training_report.json").read_text())
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["completed_updates"], 1)
            self.assertTrue((output / "LAST.pt").is_file())
            self.assertFalse((output / "artifact.json").exists())


# Training resume

"""Offline resume checks: metadata, toy adapter and fake Torch only."""

import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from pipeline import training
from pipeline import training as training_entry
from pipeline import models
from pipeline import models as runtime
from pipeline import training as data


def row():
    scene = {"schema":"research-scene-setup/v1", "base_scene_sha256":"b"*64,
             "source":{"path":"scene_setup.py","bytes":1,"sha256":"c"*64},
             "options":{},"definition":{"id":"test","kind":"table"},"assets":[],"bridge_files":[]}
    return {"id":"a","path":"a/episode","split":"train","position_id":"a","script_sha256":"a"*64,
            "scene_sha256":"b"*64,"complete":True,"scene_setup_identity":scene,
            "episode_binding":{"path":"a/episode_binding.json","bytes":1,"sha256":"d"*64}}


def manifest(root):
    path=root/'dataset.json'
    path.write_text(json.dumps({"schema":"research-dataset/v1","episodes":[row()]}))
    return path


class ToyData:
    def __init__(self,path,**kwargs):
        _,records=data.read_manifest(path)
        self.training_start=50;self.train=records
        self.scene_setup_identity=records[0]['scene_setup_identity']
        self.identities={'dataset':data.identity(path),'episodes':records}
    def unchanged(self): pass
    def normalizer_rows(self): return {'fixture':True}
    def sample(self,rng,count):
        choice=rng.integers(0,1000,size=count).tolist()
        return choice,[{'episode_id':'a','target_start':i} for i in choice]


class ToyAdapter:
    failure_at=None
    fit_count=0
    def __init__(self,kind,*,normalizer,config,device,updates,seed):
        self.kind=kind;self.norm=normalizer;self.config=config;self.total=updates;self.seed=seed
        self.completed=0;self.weight=0.;self.ema=0.;self.scheduler=0
        random.seed(seed);np.random.seed(seed)
    def train_step(self,batch):
        if self.failure_at == self.completed+1: raise RuntimeError('synthetic interruption')
        self.weight += sum(batch)+random.random()+float(np.random.random())
        self.ema=self.ema*.5+self.weight*.5
        self.completed+=1;self.scheduler+=1
        return {'loss':self.weight,'optimizer_updates':self.completed,'scheduler_updates':self.scheduler,'ema_value':self.ema}
    def save(self,path,metadata,*,recovery=False):
        value={'count':self.completed,'weight':self.weight,'ema':self.ema,'scheduler':self.scheduler,'metadata':metadata,
               'normalization':self.norm,'python_rng':random.getstate(),
               'numpy_rng':[np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]]}
        Path(path).write_text(json.dumps(value,sort_keys=True))
    @classmethod
    def resume(cls,path,*,expected_metadata,device):
        saved=json.loads(Path(path).read_text());plan=expected_metadata['plan']
        assert saved['metadata']['identities']==expected_metadata['identities']
        assert saved['normalization']==expected_metadata['normalization']
        adapter=cls(plan['model_type'],normalizer=saved['normalization'],config=plan['config'],device=device,updates=plan['updates'],seed=plan['seed'])
        adapter.completed=saved['count'];adapter.weight=saved['weight'];adapter.ema=saved['ema'];adapter.scheduler=saved['scheduler']
        def tuples(value): return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        random.setstate(tuples(saved['python_rng']))
        v=saved['numpy_rng'];np.random.set_state((v[0],np.asarray(v[1],dtype=np.uint32),*v[2:]))
        return adapter,{'completed_updates':adapter.completed,'sampler_state':saved['metadata']['sampler_state']}


class Tensor:
    def __init__(self,value,dtype='float32'):
        self.value=np.asarray(value);self.dtype=dtype;self.ndim=self.value.ndim;self.device='cpu'
    def numel(self): return self.value.size
    def detach(self): return self
    def cpu(self): return self
    def __float__(self): return float(self.value)


def fake_torch(events=None):
    events=[] if events is None else events
    return types.SimpleNamespace(Tensor=Tensor,uint8='uint8',
        isfinite=lambda t:np.isfinite(t.value),equal=lambda a,b:np.array_equal(a.value,b.value),
        Generator=lambda **kw:types.SimpleNamespace(set_state=lambda value:events.append('validate_rng')),
        cuda=types.SimpleNamespace(is_available=lambda:False,device_count=lambda:0),
        set_rng_state=lambda value:events.append('restore_torch_rng'))


def saved_payload():
    scene=row()['scene_setup_identity']
    plan={'model_type':'bc_cnn','config':{},'updates':10,'seed':42,'device':'cpu','scene_setup_identity':scene}
    norm={'fields':{'fixture':True}}
    expected={'plan':plan,'identities':{'episodes':[row()]},'normalization':norm,'software_versions':{'torch':'fixture'}}
    metadata={k:copy.deepcopy(expected[k]) for k in ('plan','identities','software_versions')}
    metadata['sampler_state']=np.random.default_rng(42).bit_generator.state
    v=np.random.RandomState(42).get_state()
    group={'params':[0],'lr':.005,'initial_lr':.01,'betas':(.95,.999),'eps':1e-8,'weight_decay':1e-6,'foreach':False}
    saved={'schema':'research-policy-checkpoint/v1','policy_contract':data.CONTRACT,'model_type':'bc_cnn','model_config':{},
        'normalization':norm,'completed_updates':2,'planned_updates':10,'seed':42,'metadata':metadata,'recovery':True,
        'model_state':{'normalizer.a':Tensor([1.]),'weight':Tensor([7.])},
        'ema_state':{'normalizer.a':Tensor([1.]),'weight':Tensor([5.])},
        'optimizer_state':{'param_groups':[group],'state':{0:{'step':Tensor(2.),'exp_avg':Tensor([.1]),'exp_avg_sq':Tensor([.2])}}},
        'scheduler_state':{'last_epoch':2,'_step_count':3,'_last_lr':[.005]},
        'ema_control':{'optimization_step':2,'decay':.5,'power':.75},
        'rng':{'python':random.Random(42).getstate(),'numpy':[v[0],v[1].tolist(),*v[2:]],
               'torch_cpu':Tensor([1,2,3],'uint8'),'torch_cuda':[]}}
    return saved,expected


class ResumeTests(unittest.TestCase):
    def context(self):
        return patch.multiple(training,Dataset=ToyData,
            frozen_module=lambda name:types.SimpleNamespace(_fit_normalizer_from_validated_policy_rows=self.fit))
    def fit(self,*args,**kwargs):
        ToyAdapter.fit_count+=1
        return None,{'fields':{},'fit_source':kwargs['fit_source']}
    def run_toy(self,dataset,output,*,model_type="bc_cnn",**kw):
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with self.context(),patch.dict(sys.modules,{'torch':fake}),patch.dict(models.ADAPTERS,{'bc_cnn':__name__+':ToyAdapter','diffusion':__name__+':ToyAdapter'}):
            return training.run_training(dataset,output,model_type=model_type,updates=6,device='cpu',
                config={'checkpoint_every':2},**kw)

    def test_interrupt_resume_matches_continuous_samples_and_state_for_both_models(self):
        # Both production types share the recovery adapter; the orchestration is
        # tested with a scalar stand-in, never real neural network optimization.
        for model_type in ("diffusion","bc_cnn"):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);dataset=manifest(root);ToyAdapter.fit_count=0;ToyAdapter.failure_at=None
                self.run_toy(dataset,root/'continuous',model_type=model_type)
                ToyAdapter.failure_at=4
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    self.run_toy(dataset,root/'interrupted',model_type=model_type)
                source=root/'interrupted';before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
                self.assertEqual(json.loads((source/'training_report.json').read_text())['completed_updates'],3)
                receipt=training.inspect_resume(source/'LAST.pt')
                self.assertNotIn('completed_updates',receipt)
                ToyAdapter.failure_at=None
                result=self.run_toy(dataset,root/'continued',resume=receipt,model_type=model_type)
                self.assertEqual((result['starting_updates'],result['new_updates'],result['completed_updates']),(2,4,6))
                full=[json.loads(line) for line in (root/'continuous/updates.jsonl').read_text().splitlines()]
                rest=[json.loads(line) for line in (root/'continued/updates.jsonl').read_text().splitlines()]
                for old,new in zip(full[2:],rest):
                    for key in ('step','sampled_windows','loss','optimizer_updates','scheduler_updates','ema_value'):
                        self.assertEqual(old[key],new[key])
                left=json.loads((root/'continuous/model.pt').read_text());right=json.loads((root/'continued/model.pt').read_text())
                for key in ('count','weight','ema','scheduler','python_rng','numpy_rng'):self.assertEqual(left[key],right[key])
                self.assertEqual(ToyAdapter.fit_count,2) # Resume never fits the normalizer.
                self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir() if p.is_file()})

    def test_complete_last_exports_without_more_updates_and_pending_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset=manifest(root);ToyAdapter.failure_at=None
            self.run_toy(dataset,root/'done')
            (root/'done/LAST.pt.pending').write_text('incomplete')
            with self.assertRaisesRegex(ValueError,'published LAST'):
                training.inspect_resume(root/'done/LAST.pt.pending')
            ToyAdapter.failure_at=7
            report=self.run_toy(dataset,root/'export',resume=root/'done/LAST.pt')
            self.assertEqual(report['new_updates'],0)
            self.assertEqual((root/'export/updates.jsonl').read_text(),'')
            self.assertTrue((root/'export/artifact.json').is_file())
            ToyAdapter.failure_at=None

    def test_receipt_rejects_changed_checkpoint_json_and_total_before_new_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset=manifest(root);ToyAdapter.failure_at=None
            self.run_toy(dataset,root/'run');receipt=training.inspect_resume(root/'run/LAST.pt')
            with self.assertRaisesRegex(ValueError,'original dataset'):
                training.run_training(dataset,root/'wrong',updates=7,model_type='bc_cnn',device='cpu',config={'checkpoint_every':2},resume=receipt)
            self.assertFalse((root/'wrong').exists())
            (root/'run/LAST.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'planned receipt'):training._resume_receipt(receipt)
            with self.assertRaises(FileExistsError):self.run_toy(dataset,root/'run')

    def test_inspector_rejects_normalizer_provenance_change_without_torch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset=manifest(root);ToyAdapter.failure_at=None
            self.run_toy(dataset,root/'run')
            p=root/'run/normalization.json';v=json.loads(p.read_text());v['train_ids']=[];p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'normalization differs'):training.inspect_resume(root/'run/LAST.pt')

    def test_worker_forwards_resume_separately_from_model_config(self):
        request={'dataset':'d','output':'o','training':{'model_type':'bc_cnn','updates':120000,'seed':42,'device':'cpu'},
                 'scene_setup_identity':{},'resume':{'checkpoint_path':'LAST.pt'}}
        with patch.object(training_entry,'read',return_value=request),patch.object(sys,'argv',['worker','request']),patch.object(training,'run_training') as run:
            training_entry.worker_main()
        self.assertEqual(run.call_args.kwargs['updates'],120000)
        self.assertEqual(run.call_args.kwargs['resume'],request['resume'])
        self.assertEqual(run.call_args.kwargs['config'],{})

    def test_registry_requires_explicit_resume_capability_never_inference_fallback(self):
        cls=types.SimpleNamespace(load=lambda *a,**kw:self.fail('inference loader used'))
        with patch.object(models,'_adapter_class',return_value=cls):
            with self.assertRaisesRegex(ValueError,'does not support'):
                models.resume_adapter('third','LAST.pt',expected_metadata={},device='cpu')

    def test_payload_rejects_missing_nonfinite_counter_source_and_rng_mismatches(self):
        saved,expected=saved_payload();self.assertEqual(runtime._recovery_payload(saved,expected,fake_torch(),'cpu')['completed_updates'],2)
        mutations=[lambda x:x.pop('model_state'),lambda x:x.update(recovery=False),
            lambda x:x['model_state'].update(weight=Tensor([float('nan')])),
            lambda x:x['metadata']['software_versions'].update(torch='different'),
            lambda x:x['metadata']['identities']['episodes'][0].update(scene_sha256='e'*64),
            lambda x:x['scheduler_state'].update(last_epoch=1),lambda x:x['ema_control'].update(optimization_step=1),
            lambda x:x['rng'].update(torch_cuda=[Tensor([1],'uint8')]),
            lambda x:x['metadata'].pop('sampler_state'),lambda x:x.update(normalization={})]
        for mutate in mutations:
            bad=copy.deepcopy(saved);mutate(bad)
            with self.subTest(mutation=mutate),self.assertRaises((ValueError,KeyError,TypeError)):
                runtime._recovery_payload(bad,expected,fake_torch(),'cpu')

    def test_real_resume_method_restores_all_state_then_rng_with_fake_objects(self):
        events=[];saved,expected=saved_payload();fake=fake_torch(events)
        fake.load=lambda *a,**kw:copy.deepcopy(saved)
        class Model:
            def __init__(self,name):self.name=name;self.state={'normalizer.a':Tensor([1.]),'weight':Tensor([0.])}
            def state_dict(self):return self.state
            def load_state_dict(self,value,strict):events.append(self.name+'_load');self.state=value
            def to(self,**kw):events.append(self.name+'_place')
            def parameters(self):return list(self.state.values())
            def buffers(self):return []
            def train(self):events.append('train_mode')
            def eval(self):events.append('ema_mode')
        class StandIn(runtime.Adapter):
            def __init__(self,kind,**kw):
                events.append('construct');self.device='cpu';self.model=Model('model');self.averaged=Model('ema')
                self.ema=types.SimpleNamespace(averaged_model=self.averaged,optimization_step=0,decay=0.,power=.75)
                self.optimizer=types.SimpleNamespace(param_groups=copy.deepcopy(saved['optimizer_state']['param_groups']),
                    load_state_dict=lambda v:events.append('optimizer_load'),zero_grad=lambda **kw:events.append('zero_grad'))
                self.scheduler=types.SimpleNamespace(load_state_dict=lambda v:events.append('scheduler_load'))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'LAST.pt';p.write_bytes(b'fake bytes; never deserialized');expected['checkpoint']=data.identity(p)
            with patch.dict(sys.modules,{'torch':fake}),patch.object(runtime.random,'setstate',side_effect=lambda x:events.append('restore_python_rng')),patch.object(runtime.np.random,'set_state',side_effect=lambda x:events.append('restore_numpy_rng')):
                adapter,info=StandIn.resume(p,expected_metadata=expected,device='cpu')
        self.assertEqual(events[-3:],['restore_python_rng','restore_numpy_rng','restore_torch_rng'])
        self.assertLess(events.index('construct'),events.index('model_load'))
        self.assertLess(events.index('optimizer_load'),events.index('scheduler_load'))
        self.assertEqual(adapter.completed,2);self.assertEqual(adapter.ema.optimization_step,2)
        self.assertEqual(info['sampler_state'],saved['metadata']['sampler_state'])
        self.assertNotIn('numpy_state',info)

    def test_inference_loader_is_still_separate_and_no_heavy_modules_remain(self):
        self.assertTrue(callable(runtime.Adapter.load))
        self.assertIsNot(runtime.Adapter.load.__func__,runtime.Adapter.resume.__func__)
        self.assertFalse({'torch','torchvision','diffusers','isaacsim','isaaclab','omni','carb','pxr'} & set(sys.modules))


# Training workflow

"""Training/resume dispatch with real JSON inspection and fake process boundaries.

No Torch, checkpoint deserialization, model, optimizer, SDK or subprocess runs.
"""
import builtins
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline import settings
from pipeline import training
from pipeline import training as training_data
from pipeline import training as training_entry
from pipeline.runtime import workflow


@contextlib.contextmanager
def no_runtime_imports():
    original = builtins.__import__
    forbidden = {'torch','torchvision','isaacsim','isaaclab','omni','carb','pxr'}
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            raise AssertionError('Forbidden runtime import: ' + name)
        return original(name, *args, **kwargs)
    with patch('builtins.__import__', side_effect=guarded):
        yield


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = settings.load_experiment()
        self.scene = workflow.scene_setup_identity(self.config)
        self.row = {'id':'fixture','path':'fixture/episode','split':'train','position_id':'fixture',
                    'complete':True,'script_sha256':'a'*64,'scene_sha256':self.scene['base_scene_sha256'],
                    'scene_setup_identity':self.scene,
                    'episode_binding':{'path':'fixture/episode_binding.json','bytes':1,'sha256':'b'*64}}
        self.dataset = self.root/'dataset.json'
        self.dataset.write_text(json.dumps({'schema':'research-dataset/v1','episodes':[self.row]}), encoding='utf-8')
        self.host = {'training_python':sys.executable,'isaac_sim':str(self.root/'unused-sdk')}
        self.config['dataset'] = str(self.dataset)
        self.config_path = self.root/'experiment.json'
        self.config_path.write_text(json.dumps(self.config), encoding='utf-8')

    def recovery_fixture(self, *, target=120000, checkpoint_every=3, create_checkpoint=True, source_name='prior-model'):
        source = self.root/source_name
        source.mkdir()
        config={'learning_rate':.0007,'model_options':{'width':7}}
        if checkpoint_every is not None:
            config['checkpoint_every']=checkpoint_every
        plan = training.training_plan(self.dataset, model_type='bc_cnn', updates=target, seed=7, device='cpu',
                                      config=config)
        normalization = {'schema':'research-normalization/v1','dataset_identity':plan['dataset'],
                         'scene_setup_identity':plan['scene_setup_identity'],
                         'training_window_start':plan['config']['training_window_start'],
                         'train_ids':plan['train_ids'],'validation_rows_used':False,'test_rows_used':False}
        (source/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        (source/'normalization.json').write_text(json.dumps(normalization), encoding='utf-8')
        # Deliberately invalid serialized state: inspection must only hash bytes.
        if create_checkpoint:
            (source/'LAST.pt').write_bytes(b'metadata-only fixture; never deserialize this checkpoint')
        return source/'LAST.pt', plan

    def restart_fixture(self, *, log_bytes=None, target=120000, checkpoint_every=None, source_name='prior-model'):
        checkpoint,plan=self.recovery_fixture(target=target,checkpoint_every=checkpoint_every,
                                              create_checkpoint=False,source_name=source_name)
        if log_bytes is not None:
            (checkpoint.parent/'updates.jsonl').write_bytes(log_bytes)
        return checkpoint,plan

    @staticmethod
    def update_rows(count):
        return b''.join((json.dumps({'step':step,'loss':1.0})+'\n').encode('utf-8') for step in range(1,count+1))

    def emit_artifact(self, request, count, *, changed_scene=False):
        output = Path(request['output'])
        output.mkdir()
        scene = copy.deepcopy(self.scene)
        if changed_scene:
            scene['source']['sha256'] = '9'*64
        row = dict(self.row, scene_setup_identity=scene)
        value = {'schema':'research-model-artifact/v1','scene_setup_identity':scene,
                 'identities':{'episodes':[row]},'completed_updates':count,
                 'fixture':'metadata-only output from a fake process callback'}
        settings.write(output/'artifact.json', value)

    def run_with_fake_worker(self, output, *, count=120000, success=True, emit=True, resume=None,
                             changed_scene=False, requested=120000):
        observed = {}
        def fake_process(command, **kwargs):
            observed.update(command=command, kwargs=kwargs)
            request = settings.read(command[-1])
            observed['request'] = request
            # Exercise the actual worker's argument forwarding, replacing only
            # run_training itself; this performs no training or model creation.
            with patch.object(sys,'argv',['training_entry.py',command[-1]]), \
                 patch.object(training,'run_training') as run:
                training_entry.worker_main()
                observed['worker_call'] = run.call_args
            if emit:
                self.emit_artifact(request,count,changed_scene=changed_scene)
            return {'success':success}
        with no_runtime_imports(),patch.object(workflow,'load_host',return_value=self.host), \
             patch.object(workflow,'require_approval',return_value={'OWNER_APPROVED':True}), \
             patch('pipeline.runtime.process.run_training_owned',side_effect=fake_process) as wrapper, \
             patch('pipeline.runtime.process.run_owned',side_effect=AssertionError('Wrong simulation wrapper')):
            result = workflow.execute_training(self.config,output,updates=requested,resume=resume)
            self.assertEqual(wrapper.call_count,1)
        return result, observed

    def test_legacy_training_limit_is_ignored_in_memory_without_rewriting(self):
        for legacy in (21600,1,None):
            config = copy.deepcopy(self.config)
            config['limits']['training_seconds'] = legacy
            path = self.root/('legacy-'+str(legacy)+'.json')
            path.write_text(json.dumps(config),encoding='utf-8')
            before = path.read_bytes()
            loaded = settings.load_experiment(path)
            self.assertIsNone(loaded['limits']['training_seconds'])
            self.assertEqual(loaded['legacy_training_time_limit_ignored'],legacy is not None)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(loaded['limits']['per_simulator_seconds'],1200)
            self.assertEqual(loaded['limits']['cleanup_seconds'],5)

    def test_omitted_training_limit_defaults_to_unlimited_training_only(self):
        config=copy.deepcopy(self.config); config['limits'].pop('training_seconds',None)
        self.config_path.write_text(json.dumps(config))
        loaded=settings.load_experiment(self.config_path)
        self.assertIsNone(loaded['limits']['training_seconds'])
        self.assertFalse(loaded['legacy_training_time_limit_ignored'])

    def test_simulator_and_cleanup_limits_remain_enforced(self):
        for key,value in [('per_simulator_seconds',0),('per_simulator_seconds',1201),
                          ('per_simulator_seconds',None),('per_simulator_seconds',True),
                          ('cleanup_seconds',4),('cleanup_seconds',6)]:
            config=copy.deepcopy(self.config); config['limits'][key]=value
            self.config_path.write_text(json.dumps(config))
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'1200 seconds'):
                settings.load_experiment(self.config_path)

    def test_cli_120000_plan_is_metadata_only_and_keeps_template_unchanged(self):
        before=self.config_path.read_bytes(); stdout=io.StringIO()
        argv=['train.py','--config',str(self.config_path),'--updates','120000']
        with no_runtime_imports(),patch.object(sys,'argv',argv),contextlib.redirect_stdout(stdout), \
             patch('pipeline.runtime.process.run_training_owned',side_effect=AssertionError('Plan must not dispatch')):
            workflow.main('train')
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['training']['updates'],120000)
        self.assertEqual(result['planned_kit_starts'],0)
        self.assertFalse(result['executed']); self.assertFalse(result['model_loaded'])
        self.assertIsNone(result['training_time_limit_seconds'])
        self.assertEqual(self.config_path.read_bytes(),before)

    def test_invalid_counts_fail_before_host_output_or_dispatch(self):
        for count in (0,-1,True,False,1.5,'120000'):
            output=self.root/('invalid-'+str(count))
            with self.subTest(count=count),patch.object(workflow,'load_host',side_effect=AssertionError('Validation must precede host')), \
                 self.assertRaisesRegex(ValueError,'positive integer'):
                workflow.execute_training(self.config,output,updates=count)
            self.assertFalse(output.exists())

    def test_cli_invalid_count_rejected_before_plan(self):
        for count in ('0','-1'):
            with self.subTest(count=count),patch.object(sys,'argv',['train.py','--config',str(self.config_path),'--updates',count]), \
                 patch.object(workflow,'make_plan',side_effect=AssertionError('Invalid count must not reach plan')), \
                 contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exit_info:
                workflow.main('train')
            self.assertEqual(exit_info.exception.code,2)

    def test_resume_inspection_preserves_original_recipe_and_target(self):
        checkpoint,plan=self.recovery_fixture()
        config=copy.deepcopy(self.config)
        config['training'].update(updates=10,model_type='diffusion',seed=99,device='cuda:0',learning_rate=.1)
        before={p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()}
        with no_runtime_imports():
            prepared,receipt=workflow.prepare_training(config,resume=str(checkpoint))
        expected=dict(plan['config'],model_type=plan['model_type'],updates=plan['updates'],seed=plan['seed'],device=plan['device'])
        self.assertEqual(prepared,expected)
        self.assertEqual(receipt['checkpoint'],training_data.identity(checkpoint))
        self.assertNotIn('completed_updates',receipt)
        self.assertEqual(before,{p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()})
        accepted,_=workflow.prepare_training(config,resume=receipt,updates=120000,model_type='bc_cnn')
        self.assertEqual(accepted,expected)

    def test_resume_rejects_conflicting_target_or_model(self):
        checkpoint,_=self.recovery_fixture()
        with self.assertRaisesRegex(ValueError,'original total update target'):
            workflow.prepare_training(self.config,resume=checkpoint,updates=119999)
        with self.assertRaisesRegex(ValueError,'checkpoint model type'):
            workflow.prepare_training(self.config,resume=checkpoint,model_type='diffusion')

    def test_resume_rejects_changed_scene_and_changed_receipt(self):
        checkpoint,_=self.recovery_fixture()
        receipt=training.inspect_resume(checkpoint)
        changed=copy.deepcopy(self.scene); changed['source']['sha256']='f'*64
        with patch.object(workflow,'scene_setup_identity',return_value=changed),self.assertRaisesRegex(ValueError,'Resume scene differs'):
            workflow.prepare_training(self.config,resume=receipt)
        checkpoint.write_bytes(checkpoint.read_bytes()+b' changed')
        with self.assertRaisesRegex(ValueError,'inputs changed after planning'):
            workflow.prepare_training(self.config,resume=receipt)

    def test_resume_cli_plan_reports_original_target_without_loading_state(self):
        checkpoint,_=self.recovery_fixture(); stdout=io.StringIO()
        argv=['train.py','--config',str(self.config_path),'--resume',str(checkpoint)]
        with no_runtime_imports(),patch.object(sys,'argv',argv),contextlib.redirect_stdout(stdout), \
             patch.object(training,'run_training',side_effect=AssertionError('No training from plan')):
            workflow.main('train')
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['training']['updates'],120000)
        self.assertEqual(result['training']['model_type'],'bc_cnn')
        self.assertEqual(result['resume']['original_total_updates'],120000)
        self.assertFalse(result['resume']['state_loaded'])
        self.assertIsNone(result['training_time_limit_seconds'])

    def test_fresh_dispatch_uses_training_wrapper_with_no_timeout_and_exact_total(self):
        result,observed=self.run_with_fake_worker(self.root/'fresh')
        self.assertIsNone(observed['kwargs']['timeout_seconds'])
        self.assertEqual(observed['kwargs']['cleanup_seconds'],5)
        self.assertEqual(observed['request']['training']['updates'],120000)
        self.assertIsNone(observed['request']['resume'])
        self.assertEqual(observed['worker_call'].kwargs['updates'],120000)
        self.assertEqual(observed['worker_call'].kwargs['expected_scene_setup_identity'],self.scene)
        self.assertEqual(result['completion'],'COMPLETED')
        self.assertEqual((result['requested_updates'],result['completed_updates']),(120000,120000))
        self.assertIsNone(result['training_time_limit_seconds'])

    def test_resume_dispatch_carries_authenticated_receipt_through_worker(self):
        checkpoint,plan=self.recovery_fixture(); receipt=training.inspect_resume(checkpoint)
        before={p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()}
        result,observed=self.run_with_fake_worker(self.root/'continued',resume=receipt,requested=None)
        self.assertEqual(observed['request']['resume'],receipt)
        self.assertEqual(observed['worker_call'].kwargs['resume'],receipt)
        self.assertEqual(observed['worker_call'].kwargs['config'],plan['config'])
        self.assertEqual(observed['worker_call'].kwargs['updates'],120000)
        self.assertEqual(result['resume_checkpoint'],str(checkpoint))
        self.assertEqual(result['completion'],'COMPLETED')
        self.assertEqual(before,{p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()})

    def test_successful_process_requires_exact_integer_completed_total(self):
        for index,count in enumerate((119999,120001,0,None,True,'120000')):
            with self.subTest(count=count):
                result,_=self.run_with_fake_worker(self.root/('bad-total-'+str(index)),count=count)
                self.assertEqual(result['completion'],'FAILED')
                self.assertIn('requested total update count',result['errors'][0])

    def test_failed_process_or_missing_artifact_cannot_complete(self):
        result,_=self.run_with_fake_worker(self.root/'failed-process',success=False)
        self.assertEqual(result['completion'],'FAILED')
        result,_=self.run_with_fake_worker(self.root/'missing-artifact',emit=False)
        self.assertEqual(result['completion'],'FAILED')
        self.assertIsNone(result['artifact'])

    def test_exported_wrong_scene_cannot_complete(self):
        result,_=self.run_with_fake_worker(self.root/'wrong-scene',changed_scene=True)
        self.assertEqual(result['completion'],'FAILED')
        self.assertIn('scene differs',result['errors'][0])

    def test_resume_and_training_options_reject_other_modes(self):
        cases=[('record',['--resume','missing/LAST.pt']),('scene',['--resume','missing/LAST.pt']),
               ('live',['--resume','missing/LAST.pt']),('script',['--resume','missing/LAST.pt']),
               ('record',['--updates','120000']),('scene',['--model','bc_cnn']),
               ('train',['--selection-seed','42'])]
        for mode,flags in cases:
            argv=['entry.py','--config',str(self.config_path),*flags]
            with self.subTest(mode=mode,flags=flags),patch.object(sys,'argv',argv), \
                 contextlib.redirect_stderr(io.StringIO()),patch.object(workflow,'execute_training',side_effect=AssertionError('No dispatch')), \
                 patch.object(workflow,'execute_simulation',side_effect=AssertionError('No native dispatch')),self.assertRaises(SystemExit) as exit_info:
                workflow.main(mode)
            self.assertEqual(exit_info.exception.code,2)

    def test_missing_last_before_default_first_save_is_explicit_restart_receipt(self):
        self.assertEqual(training.resolve_config(None,120000,'cpu')['checkpoint_every'],1000)
        for count in (0,1,999):
            checkpoint,plan=self.restart_fixture(log_bytes=self.update_rows(count),source_name='prefix-'+str(count))
            source_before={p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()}
            with self.subTest(count=count),no_runtime_imports():
                receipt=training.inspect_resume(checkpoint)
                self.assertEqual(receipt['schema'],'research-restart-receipt/v1')
                self.assertIsNone(receipt['checkpoint'])
                self.assertTrue(receipt['restart_from_zero'])
                self.assertEqual(receipt['logged_updates'],count)
                self.assertEqual(receipt['checkpoint_every'],1000)
                self.assertEqual(receipt['plan'],plan)
                self.assertEqual(receipt['updates_log_identity'],training_data.identity(checkpoint.parent/'updates.jsonl'))
                self.assertFalse(receipt['trailing_partial_log'])
                self.assertIn('restart',receipt['notice'].lower())
                self.assertRegex(receipt['notice'].lower(),r'\b(0|zero)\b')
                self.assertFalse(checkpoint.exists())
                self.assertEqual(source_before,{p.name:training_data.identity(p) for p in checkpoint.parent.iterdir()})

    def test_missing_last_without_updates_log_is_zero_with_no_log_identity(self):
        checkpoint,_=self.restart_fixture()
        with no_runtime_imports(): receipt=training.inspect_resume(checkpoint)
        self.assertEqual(receipt['logged_updates'],0)
        self.assertIsNone(receipt['updates_log_identity'])
        self.assertFalse(receipt['trailing_partial_log'])

    def test_missing_last_truncated_final_json_is_ignored_but_complete_final_json_counts(self):
        cases=[(self.update_rows(2)+b'{"step":3,',2,True),
               (self.update_rows(2)+b'{"step":3}',3,False)]
        for index,(data,count,partial) in enumerate(cases):
            checkpoint,_=self.restart_fixture(log_bytes=data,source_name='tail-'+str(index))
            with self.subTest(index=index),no_runtime_imports(): receipt=training.inspect_resume(checkpoint)
            self.assertEqual(receipt['logged_updates'],count)
            self.assertEqual(receipt['trailing_partial_log'],partial)

    def test_missing_last_at_save_boundary_or_final_target_rejects_restart(self):
        for index,(target,interval,count) in enumerate([(120000,1000,1000),(120000,1000,1001),(17,1000,17),(120000,3,3)]):
            checkpoint,_=self.restart_fixture(log_bytes=self.update_rows(count),target=target,
                                              checkpoint_every=interval,source_name='boundary-'+str(index))
            with self.subTest(target=target,interval=interval,count=count),self.assertRaises(ValueError):
                training.inspect_resume(checkpoint)
        checkpoint,_=self.restart_fixture(log_bytes=self.update_rows(16),target=17,source_name='below-final')
        self.assertEqual(training.inspect_resume(checkpoint)['logged_updates'],16)

    def test_missing_last_rejects_malformed_complete_or_nonsequential_logs(self):
        cases=[b'{"step":1,}\n', b'{"step":1,}', b'not-json', b'{"step":1}\n{"step":3}\n', b'{"step":2}\n',
               b'{"step":0}\n', b'{"step":1}\n{"step":1}\n', b'{"step":true}\n',
               b'{"step":"1"}\n', b'{"step":1.0}\n', b'[]\n', b'{}\n']
        for index,data in enumerate(cases):
            checkpoint,_=self.restart_fixture(log_bytes=data,source_name='bad-log-'+str(index))
            with self.subTest(data=data),self.assertRaises(ValueError):
                training.inspect_resume(checkpoint)

    def test_missing_last_rejects_missing_metadata_and_prior_resume_markers(self):
        for index,name in enumerate(('plan.json','normalization.json')):
            checkpoint,_=self.restart_fixture(source_name='missing-metadata-'+str(index))
            (checkpoint.parent/name).unlink()
            with self.subTest(missing=name),self.assertRaises((OSError,ValueError)):
                training.inspect_resume(checkpoint)
        for index,name in enumerate(('resume_source.json','resume.json')):
            checkpoint,_=self.restart_fixture(source_name='resumed-source-'+str(index))
            (checkpoint.parent/name).write_text('{}',encoding='utf-8')
            with self.subTest(marker=name),self.assertRaises(ValueError):
                training.inspect_resume(checkpoint)

    def test_existing_invalid_last_is_never_reclassified_as_restart(self):
        for index,data in enumerate((b'not a serialized checkpoint',b'')):
            checkpoint,_=self.recovery_fixture(source_name='existing-last-'+str(index))
            checkpoint.write_bytes(data)
            with self.subTest(data=data),no_runtime_imports(): receipt=training.inspect_resume(checkpoint)
            self.assertEqual(receipt['schema'],'research-resume-receipt/v1')
            self.assertEqual(receipt['checkpoint'],training_data.identity(checkpoint))
            self.assertFalse(receipt.get('restart_from_zero',False))

    def test_restart_receipt_reinspection_rejects_new_log_rows_or_new_checkpoint(self):
        checkpoint,_=self.restart_fixture(log_bytes=self.update_rows(2),source_name='append-log')
        receipt=training.inspect_resume(checkpoint)
        (checkpoint.parent/'updates.jsonl').write_bytes(self.update_rows(3))
        with self.assertRaisesRegex(ValueError,'inputs changed after planning'):
            workflow.prepare_training(self.config,resume=receipt)
        checkpoint,_=self.restart_fixture(log_bytes=self.update_rows(2),source_name='new-last')
        receipt=training.inspect_resume(checkpoint)
        checkpoint.write_bytes(b'new checkpoint appeared after planning')
        with self.assertRaisesRegex(ValueError,'inputs changed after planning'):
            workflow.prepare_training(self.config,resume=receipt)

    def test_restart_plan_preserves_recipe_target_and_stays_clean_json(self):
        checkpoint,plan=self.restart_fixture(log_bytes=self.update_rows(4)); stdout=io.StringIO()
        argv=['train.py','--config',str(self.config_path),'--resume',str(checkpoint)]
        with no_runtime_imports(),patch.object(sys,'argv',argv),contextlib.redirect_stdout(stdout), \
             patch.object(training,'run_training',side_effect=AssertionError('Plan cannot start training')):
            workflow.main('train')
        result=json.loads(stdout.getvalue())
        expected=dict(plan['config'],model_type=plan['model_type'],updates=plan['updates'],seed=plan['seed'],device=plan['device'])
        self.assertEqual(result['training'],expected)
        self.assertTrue(result['resume']['restart_from_zero'])
        self.assertEqual(result['resume']['logged_updates'],4)
        self.assertTrue(result['resume']['notice'])
        self.assertIsNone(result['resume']['checkpoint'])
        self.assertFalse(result['resume']['state_loaded'])
        with self.assertRaisesRegex(ValueError,'original total update target'):
            workflow.prepare_training(self.config,resume=checkpoint,updates=10)

    def test_restart_dispatch_prints_notice_and_forwards_receipt_without_claiming_saved_checkpoint(self):
        checkpoint,plan=self.restart_fixture(log_bytes=self.update_rows(5))
        receipt=training.inspect_resume(checkpoint)
        stdout=io.StringIO(); stderr=io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            result,observed=self.run_with_fake_worker(self.root/'restart-output',resume=receipt,requested=None)
        self.assertEqual(observed['request']['resume'],receipt)
        self.assertEqual(observed['worker_call'].kwargs['resume'],receipt)
        self.assertEqual(observed['worker_call'].kwargs['config'],plan['config'])
        self.assertEqual(observed['worker_call'].kwargs['updates'],plan['updates'])
        self.assertIn(receipt['notice'],stdout.getvalue()+stderr.getvalue())
        self.assertTrue(result['restarted_from_zero'])
        self.assertIsNone(result['resume_checkpoint'])
        self.assertEqual(result['completion'],'COMPLETED')
        self.assertIsNone(observed['kwargs']['timeout_seconds'])


if __name__ == "__main__":
    unittest.main()
