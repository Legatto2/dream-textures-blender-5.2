"""Focused stdlib checks for the independent JasonHoku backend backports.

Run from the repository root:
    python compatibility/tests/test_backend_backports.py
    python compatibility/tests/test_backend_backports.py --source /path/to/baseline

The tests execute the production callback, cache classification and loader with
small dependency stubs. Weight fixtures establish file availability, not tensor
validity. The DPT check stops at processor construction; no AI model is loaded.
These are not Blender or CUDA integration tests.
"""

import argparse
import ast
from dataclasses import dataclass
from enum import Enum, IntEnum
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2]
NS = types.SimpleNamespace
FP16_NAMES = (
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.fp16.bin",
)
FP32_NAMES = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
)
WEIGHT_FIXTURE = b"Availability fixture, not tensor weights."


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class FutureTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules, {})
        modules.start()
        self.addCleanup(modules.stop)
        self.future_class = load_source(
            "dt_backports_future", SOURCE / "generator_process/future.py").Future

    def test_default_cancellation_callback_is_callable(self):
        self.assertIs(self.future_class().check_cancelled(), False)

    def test_actor_callback_override_remains_an_instance_function(self):
        first = self.future_class()
        second = self.future_class()
        cancelled = iter((False, True))
        first.check_cancelled = lambda: next(cancelled)
        self.assertIs(first.check_cancelled(), False)
        self.assertIs(first.check_cancelled(), True)
        self.assertIs(second.check_cancelled(), False)


class ModelType(IntEnum):
    UNKNOWN = 0
    CONTROL_NET = -1
    PROMPT_TO_IMAGE = 4


class ModelConfig(Enum):
    AUTO_DETECT = 1


@dataclass(frozen=True)
class Checkpoint:
    path: str
    config: object


class ControlNetAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cache = root / "hub"
        self.legacy_cache = root / "legacy"
        self.cache.mkdir()
        self.legacy_cache.mkdir()
        self.loads = []
        self.float16 = object()

        def from_pretrained(path, **kwargs):
            # ModelMixin prefers an existing safetensors file by default. An
            # empty preferred file fails to load instead of falling back to bin.
            stem = "diffusion_pytorch_model" + (".fp16" if kwargs.get("variant") == "fp16" else "")
            use_safetensors = kwargs.get("use_safetensors")
            suffixes = (".safetensors", ".bin") if use_safetensors is None else (
                (".safetensors",) if use_safetensors else (".bin",))
            selected = next((Path(path) / (stem + suffix) for suffix in suffixes
                             if (Path(path) / (stem + suffix)).is_file()), None)
            if selected is None or selected.stat().st_size == 0:
                raise ValueError("Selected model file is missing or empty")
            result = NS(path=path, selected_filename=selected.name, **kwargs)
            self.loads.append(result)
            return result

        mods = {
            "requests": module("requests"),
            "torch": module("torch", float16=self.float16),
            "huggingface_hub": module("huggingface_hub", __path__=[]),
            "huggingface_hub.constants": module(
                "huggingface_hub.constants", HF_HUB_CACHE=str(self.cache)),
            "diffusers": module(
                "diffusers", __path__=[], ControlNetModel=NS(from_pretrained=from_pretrained)),
            "diffusers.utils": module("diffusers.utils", __path__=[]),
            "diffusers.utils.hub_utils": module(
                "diffusers.utils.hub_utils", old_diffusers_cache=str(self.legacy_cache)),
            "dt_backports": module("dt_backports", __path__=[str(SOURCE)]),
            "dt_backports.generator_process": module(
                "dt_backports.generator_process", __path__=[str(SOURCE / "generator_process")]),
            "dt_backports.generator_process.actions": module(
                "dt_backports.generator_process.actions",
                __path__=[str(SOURCE / "generator_process/actions")]),
            "dt_backports.generator_process.models": module(
                "dt_backports.generator_process.models", ModelType=ModelType,
                ModelConfig=ModelConfig, Checkpoint=Checkpoint, Scheduler=Enum),
        }
        modules = patch.dict(sys.modules, mods)
        modules.start()
        self.addCleanup(modules.stop)
        self.downloads = load_source(
            "dt_backports.generator_process.actions.huggingface_hub",
            SOURCE / "generator_process/actions/huggingface_hub.py")
        self.loader = load_source(
            "dt_backports.generator_process.actions.load_model",
            SOURCE / "generator_process/actions/load_model.py")

    def snapshot(self, name):
        model_id = f"review/{name}"
        storage = self.cache / f"models--review--{name}"
        snapshot = storage / "snapshots" / "test-revision"
        snapshot.mkdir(parents=True)
        refs = storage / "refs"
        refs.mkdir()
        (refs / "main").write_text("test-revision", encoding="utf-8")
        (snapshot / "config.json").write_text(
            json.dumps({"_class_name": "ControlNetModel"}), encoding="utf-8")
        return model_id, storage, snapshot

    def classified_type(self, storage):
        models = self.downloads.hf_list_installed_models(None)
        return next(model.model_type for model in models if Path(model.id) == storage)

    def test_unavailable_controlnet_is_not_classified_or_loaded(self):
        cases = (
            ("config-only", None, None),
            ("empty", FP32_NAMES[0], b""),
            ("index", FP32_NAMES[0] + ".index.json", b"{}"),
            ("incomplete", FP32_NAMES[0] + ".incomplete", WEIGHT_FIXTURE),
            ("directory", FP32_NAMES[0], None),
            ("shard", "diffusion_pytorch_model-00001-of-00002.safetensors", WEIGHT_FIXTURE),
        )
        for name, filename, contents in cases:
            with self.subTest(case=name):
                model_id, storage, snapshot = self.snapshot(name)
                if filename is not None:
                    if contents is None:
                        (snapshot / filename).mkdir()
                    else:
                        (snapshot / filename).write_bytes(contents)
                self.assertEqual(self.classified_type(storage), ModelType.UNKNOWN)
                with self.assertRaises(FileNotFoundError) as caught:
                    self.loader._load_controlnet_model({}, model_id, True)
                message = str(caught.exception)
                self.assertIn(model_id, message)
                self.assertIn("download", message.lower())
                self.assertIn("incomplete", message.lower())
        self.assertEqual(self.loads, [], "unavailable weights must not reach Diffusers")

    def test_supported_nonempty_weights_are_available_to_classifier_and_loader(self):
        for index, filename in enumerate(FP16_NAMES + FP32_NAMES):
            with self.subTest(filename=filename):
                model_id, storage, snapshot = self.snapshot(f"complete-{index}")
                (snapshot / filename).write_bytes(WEIGHT_FIXTURE)
                self.assertEqual(self.classified_type(storage), ModelType.CONTROL_NET)
                half_precision = filename in FP16_NAMES
                loaded = self.loader._load_controlnet_model({}, model_id, half_precision)
                self.assertEqual(Path(loaded.path), snapshot)
                self.assertEqual(loaded.selected_filename, filename)
                self.assertEqual(loaded.variant, "fp16" if half_precision else None)
                self.assertIs(loaded.torch_dtype, self.float16 if half_precision else None)
        self.assertEqual(len(self.loads), 4)

    def test_precision_preference_and_available_fallback_are_preserved(self):
        model_id, storage, snapshot = self.snapshot("precision")
        fp16_file = snapshot / FP16_NAMES[1]
        fp32_file = snapshot / FP32_NAMES[0]
        fp16_file.write_bytes(WEIGHT_FIXTURE)
        fp32_file.write_bytes(WEIGHT_FIXTURE)
        self.assertEqual(self.loader._load_controlnet_model({}, model_id, True).variant, "fp16")
        self.assertIsNone(self.loader._load_controlnet_model({}, model_id, False).variant)

        fp16_file.write_bytes(b"")
        loaded = self.loader._load_controlnet_model({}, model_id, True)
        self.assertIsNone(loaded.variant)
        self.assertIs(loaded.torch_dtype, self.float16)

        fp16_file.write_bytes(WEIGHT_FIXTURE)
        fp32_file.write_bytes(b"")
        with self.assertLogs(self.loader.logger, level="WARNING"):
            loaded = self.loader._load_controlnet_model({}, model_id, False)
        self.assertEqual(loaded.variant, "fp16")
        self.assertIsNone(loaded.torch_dtype)
        self.assertEqual(self.classified_type(storage), ModelType.CONTROL_NET)

        # A same-precision empty safetensors file must not override the available bin.
        (snapshot / FP16_NAMES[0]).write_bytes(b"")
        loaded = self.loader._load_controlnet_model({}, model_id, True)
        self.assertEqual(loaded.selected_filename, FP16_NAMES[1])

    def test_hub_blob_symlinks_must_resolve_to_available_weights(self):
        model_id, storage, snapshot = self.snapshot("symlink")
        blob = storage / "blobs" / "test-blob"
        link = snapshot / FP32_NAMES[0]
        try:
            link.symlink_to(blob)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"File symlinks unavailable on this platform: {error}")
        self.assertEqual(self.classified_type(storage), ModelType.UNKNOWN)
        with self.assertRaises(FileNotFoundError):
            self.loader._load_controlnet_model({}, model_id, False)
        blob.parent.mkdir()
        blob.write_bytes(WEIGHT_FIXTURE)
        self.assertEqual(self.classified_type(storage), ModelType.CONTROL_NET)
        self.assertEqual(Path(self.loader._load_controlnet_model({}, model_id, False).path), snapshot)

    def test_legacy_cache_models_remain_discoverable(self):
        storage = self.legacy_cache / "local-controlnet"
        storage.mkdir()
        (storage / "config.json").write_text(
            json.dumps({"_class_name": "ControlNetModel"}), encoding="utf-8")
        (storage / FP32_NAMES[0]).write_bytes(WEIGHT_FIXTURE)
        self.assertEqual(self.classified_type(storage), ModelType.CONTROL_NET)


class DepthProcessorTests(unittest.TestCase):
    def test_depth_estimation_loads_the_current_dpt_image_processor(self):
        tree = ast.parse((SOURCE / "generator_process/actions/depth_to_image.py").read_text(encoding="utf-8"))
        pipeline = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef) and node.name == "DreamTexturesDepth2ImgPipeline")
        prepare_depth = next(node for node in pipeline.body
                             if isinstance(node, ast.FunctionDef) and node.name == "prepare_depth")
        calls = []

        class ProcessorReached(Exception):
            pass

        def from_pretrained(model_id):
            calls.append(model_id)
            raise ProcessorReached

        transformers = module(
            "transformers", DPTImageProcessor=NS(from_pretrained=from_pretrained),
            DPTForDepthEstimation=object)
        namespace = {"torch": NS(device=lambda device: NS(type=device))}
        exec(compile(ast.Module(body=[prepare_depth], type_ignores=[]), "depth-processor", "exec"), namespace)
        with patch.dict(sys.modules, {"transformers": transformers}):
            with self.assertRaises(ProcessorReached):
                namespace["prepare_depth"](None, None, object(), object(), NS(type="cuda"))
        self.assertEqual(calls, ["Intel/dpt-large"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, unittest_args = parser.parse_known_args()
    SOURCE = args.source.resolve()
    print(f"Backend backport checks against {SOURCE}; no real CUDA or Blender execution.")
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
