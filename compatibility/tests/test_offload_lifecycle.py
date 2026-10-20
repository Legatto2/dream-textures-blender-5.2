"""Behavioral regression simulation; this is NOT a CUDA or Blender integration test.

The stubs model the relevant Diffusers 0.27.2 / Accelerate ownership semantics:
AutoPipeline.from_pipe shares model modules but creates a new pipeline without
its predecessor's hook bookkeeping; model offload attaches at the enclosing
MultiControlNet, and maybe_free_model_hooks leaves CPU weights with fresh hooks.

Run with only the Python standard library:
    python compatibility/tests/test_offload_lifecycle.py
    python compatibility/tests/test_offload_lifecycle.py --source /path/to/baseline
The default source is the checked-out repository. A pre-fix baseline
intentionally demonstrates failures. These checks require no AI runtime.
"""

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2]
NS = types.SimpleNamespace


class Device(str):
    def __new__(cls, value):
        return super().__new__(cls, "cuda:0" if str(value) == "cuda" else str(value))

    @property
    def type(self):
        return self.split(":")[0]


@dataclass
class Tensor:
    device: Device


class Module:
    """A small module with placement-sensitive parameters and callable children."""
    def __init__(self, name, children=()):
        self.name = name
        self.device = Device("cpu")
        self.submodules = list(children)

    def modules(self):
        yield self
        for child in self.submodules:
            yield from child.modules()

    def children(self):
        return iter(self.submodules)

    def to(self, device=None, **kwargs):
        if device is not None and str(device) != "float16":
            self.device = Device(device)
            for child in self.submodules:
                child.to(device)
        return self

    def __call__(self, value):
        hook = getattr(self, "_hf_hook", None)
        if hook is not None:
            value = hook.pre_forward(value)
        if not self.submodules and value.device != self.device:
            raise RuntimeError(
                f"{self.name}: input {value.device}, weights {self.device}; "
                "Expected all tensors to be on the same device"
            )
        for child in self.submodules:
            value = child(value)
        return value

    def set_attn_processor(self, _):
        pass

    def enable_slicing(self):
        pass

    def disable_slicing(self):
        pass

    def decode(self, value):
        return self(value)


class ControlNet(Module):
    loads = []

    def __init__(self, name):
        super().__init__(name, [Module(name + ".time_embedding.linear_1")])

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loads.append(path)
        return cls(path)


class MultiControlNet(Module):
    def __init__(self, nets):
        self.nets = list(nets)
        super().__init__("MultiControlNet", self.nets)


class ModelHook:
    def __init__(self, model, device, previous=None, sequential=False):
        self.model = model
        self.execution_device = Device(device)
        self.previous = previous
        self.sequential = sequential
        self.detached = False

    def pre_forward(self, value):
        if self.previous is not None:
            self.previous.offload()
        self.model.to(self.execution_device)
        return Tensor(self.execution_device)

    def detach_hook(self, model):
        self.detached = True
        # AlignDevicesHook restores meta/offloaded parameters from its weights
        # map on detach. Represent that restoration by placing them on CPU.
        if self.sequential:
            model.to("cpu")
        return model


class Handle:
    all_handles = []

    def __init__(self, model, hook):
        self.model = model
        self.hook = hook
        self.offloads = 0
        self.removes = 0
        self.all_handles.append(self)

    def offload(self):
        self.offloads += 1
        self.model.to("cpu")

    def remove(self):
        self.removes += 1
        remove_hook_from_module(self.model)


def remove_hook_from_module(model, recurse=False):
    if hasattr(model, "_hf_hook"):
        model._hf_hook.detach_hook(model)
        del model._hf_hook
    if recurse:
        for child in model.children():
            remove_hook_from_module(child, recurse=True)
    return model


def attach_hook(model, device, previous=None, sequential=False):
    hook = ModelHook(model, device, previous, sequential)
    model._hf_hook = hook
    return Handle(model, hook)


class Pipeline:
    def __init__(self, *, text_encoder=None, unet=None, vae=None,
                 controlnet=None, task="text"):
        self.text_encoder = text_encoder or Module("text_encoder")
        self.unet = unet or Module("unet")
        self.vae = vae or Module("vae")
        self.controlnet = controlnet
        self.task = task

    @property
    def components(self):
        return {"text_encoder": self.text_encoder, "unet": self.unet,
                "vae": self.vae, "controlnet": self.controlnet}

    @property
    def device(self):
        # Diffusers 0.27.2 selects the first nn.Module in its signature-key
        # iteration. UNet-first is one permitted order; it also represents the
        # observed failing path, where ControlNet runs before UNet is onloaded.
        return self.unet.device

    @property
    def _execution_device(self):
        # Same early-return semantics as Diffusers 0.27.2 pipeline_utils.py.
        for model in self.components.values():
            if not isinstance(model, Module):
                continue
            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if hasattr(module, "_hf_hook"):
                    return module._hf_hook.execution_device
        return self.device

    def to(self, device, **kwargs):
        for model in self.components.values():
            if isinstance(model, Module):
                model.to(device)
        return self

    def enable_model_cpu_offload(self, device="cuda"):
        self.to("cpu")
        self._offload_device = Device(device)
        self._all_hooks = []
        previous = None
        for model in (self.text_encoder, self.unet, self.vae):
            previous = attach_hook(model, device, previous)
            self._all_hooks.append(previous)
        if self.controlnet is not None:
            self._all_hooks.append(attach_hook(self.controlnet, device))

    def enable_sequential_cpu_offload(self, device="cuda"):
        self.to("cpu")
        for model in self.components.values():
            if isinstance(model, Module):
                for module in model.modules():
                    attach_hook(module, device, sequential=True)

    def maybe_free_model_hooks(self):
        if not getattr(self, "_all_hooks", None):
            return
        for hook in self._all_hooks:
            hook.offload()
            hook.remove()
        self.enable_model_cpu_offload(self._offload_device)

    def generate(self):
        execution_device = self._execution_device
        value = self.text_encoder(Tensor(execution_device))
        if self.controlnet is not None:
            value = self.controlnet(value)
        value = self.unet(value)
        value = self.vae(value)
        self.maybe_free_model_hooks()
        return value


class XLPipeline(Pipeline):
    pass


class AutoText:
    conversions = []
    task = "text"

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return Pipeline(controlnet=kwargs.get("controlnet"), task=cls.task)

    @classmethod
    def from_pipe(cls, pipeline, **kwargs):
        cls.conversions.append(pipeline)
        # New pipeline; shared modules; _all_hooks is intentionally not copied.
        return Pipeline(text_encoder=pipeline.text_encoder, unet=pipeline.unet,
                        vae=pipeline.vae, controlnet=kwargs.get("controlnet"),
                        task=cls.task)


class AutoImage(AutoText):
    task = "image"


class Scheduler(Enum):
    DEFAULT = 1

    def create(self, pipe):
        pipe.scheduler = self
        return self


@dataclass(frozen=True)
class Checkpoint:
    path: str
    config: object


class ModelConfig(Enum):
    AUTO_DETECT = 1


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


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = str(Path(self.tmp.name) / "base")
        self.control_a = str(Path(self.tmp.name) / "depth-a")
        self.control_b = str(Path(self.tmp.name) / "depth-b")
        for path in (self.control_a, self.control_b):
            Path(path).mkdir()
            (Path(path) / "diffusion_pytorch_model.safetensors").write_bytes(b"stub")
        Handle.all_handles.clear()
        ControlNet.loads.clear()
        AutoText.conversions.clear()
        mods = {
            "torch": module("torch", nn=NS(Module=Module), device=Device,
                            float16="float16", contiguous_format="contiguous",
                            channels_last="channels_last",
                            cuda=NS(empty_cache=lambda: None,
                                    get_device_name=lambda: "simulation GPU"),
                            backends=NS(cudnn=NS(benchmark=False),
                                        cuda=NS(matmul=NS(allow_tf32=False)))),
            "diffusers": module("diffusers", DiffusionPipeline=Pipeline,
                                StableDiffusionXLPipeline=XLPipeline,
                                AutoPipelineForImage2Image=AutoImage,
                                ControlNetModel=ControlNet),
            "diffusers.pipelines": module("diffusers.pipelines"),
            "diffusers.pipelines.controlnet": module("diffusers.pipelines.controlnet"),
            "diffusers.pipelines.controlnet.multicontrolnet": module(
                "diffusers.pipelines.controlnet.multicontrolnet", MultiControlNetModel=MultiControlNet),
            "diffusers.models": module("diffusers.models"),
            "diffusers.models.attention_processor": module(
                "diffusers.models.attention_processor", AttnProcessor2_0=object),
            "accelerate": module("accelerate"),
            "accelerate.hooks": module("accelerate.hooks", remove_hook_from_module=remove_hook_from_module),
            "dt_regression": module("dt_regression"),
            "dt_regression.generator_process": module(
                "dt_regression.generator_process", __path__=[str(SOURCE / "generator_process")]),
            "dt_regression.generator_process.actions": module("dt_regression.generator_process.actions"),
            "dt_regression.generator_process.models": module(
                "dt_regression.generator_process.models", Checkpoint=Checkpoint,
                ModelConfig=ModelConfig, Scheduler=Scheduler),
            "dt_regression.generator_process.models.upscale_tiler": module(
                "dt_regression.generator_process.models.upscale_tiler", tiled_decode_latents=lambda *a, **k: None),
            "dt_regression.generator_process.directml_patches": module(
                "dt_regression.generator_process.directml_patches", enable=lambda p: None, disable=lambda p: None),
        }
        self.module_patch = patch.dict(sys.modules, mods)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.optimization = load_source(
            "dt_regression.generator_process.models.optimizations",
            SOURCE / "generator_process/models/optimizations.py")
        self.loader = load_source("dt_regression.generator_process.actions.load_model",
                                  SOURCE / "generator_process/actions/load_model.py")
        self.loader.revision_paths = lambda model, config="model_index.json": {"main": model}
        self.owner = NS(choose_device=lambda _: "cuda")
        self.options = self.optimization.Optimizations(
            cpu_offload=self.optimization.CPUOffload.MODEL)

    def pipeline(self, controls=None, task=AutoText):
        pipe = self.loader.load_model(self.owner, task, self.base, self.options,
                                      Scheduler.DEFAULT, controlnet=controls)
        return self.options.apply(pipe, "cuda")

    def test_same_controlnet_three_consecutive_generations(self):
        shared = None
        for attempt in range(3):
            with self.subTest(attempt=attempt + 1):
                pipe = self.pipeline([self.control_a])
                if shared is None:
                    shared = pipe.controlnet.nets[0]
                self.assertIs(pipe.controlnet.nets[0], shared, "weights should remain cached")
                self.assertEqual(pipe.generate().device, Device("cuda"))
                self.assertEqual(len(pipe._all_hooks), 4, "current pipeline must own cleanup hooks")
        self.assertEqual(ControlNet.loads, [self.control_a])

    def test_changed_controlnet_releases_old_wrapper_before_reuse(self):
        first = self.pipeline([self.control_a])
        first.generate()
        old_wrapper = first.controlnet
        old_handles = list(first._all_hooks)
        second = self.pipeline([self.control_b])
        self.assertEqual(second.generate().device, Device("cuda"))
        self.assertFalse(hasattr(old_wrapper, "_hf_hook"), "retired wrapper must release its hook")
        self.assertTrue(all(h.offloads and h.removes for h in old_handles))
        self.assertNotIn(self.control_a, self.owner._pipe[1])

    def test_task_conversion_keeps_shared_weights_with_current_hook_ownership(self):
        first = self.pipeline([self.control_a])
        first.generate()
        second = self.pipeline([self.control_a], task=AutoImage)
        self.assertIs(second.unet, first.unet)
        self.assertEqual(second.task, "image")
        self.assertEqual(second.generate().device, Device("cuda"))
        self.assertEqual(len(second._all_hooks), 4)
        self.assertEqual(getattr(first, "_all_hooks", []), [])

    def test_multiple_controlnets_can_be_reordered_without_losing_placement(self):
        first = self.pipeline([self.control_a, self.control_b])
        first.generate()
        original_nets = list(first.controlnet.nets)
        second = self.pipeline([self.control_b, self.control_a])
        self.assertEqual(second.controlnet.nets, list(reversed(original_nets)))
        self.assertEqual(second.generate().device, Device("cuda"))

    def test_offload_disabled_does_not_install_or_remove_hooks(self):
        self.options.cpu_offload = self.optimization.CPUOffload.OFF
        with patch.dict(sys.modules, {"accelerate.hooks": None}):
            for _ in range(3):
                self.assertEqual(self.pipeline([self.control_a]).generate().device, Device("cuda"))
        self.assertEqual(Handle.all_handles, [])

    def test_plain_model_offload_rebinds_cleanup_to_converted_pipeline(self):
        first = self.pipeline()
        first.generate()
        second = self.pipeline()
        self.assertEqual(second.generate().device, Device("cuda"))
        self.assertEqual(len(getattr(second, "_all_hooks", [])), 3)

    def test_sequential_offload_detaches_nested_hooks_before_wrapper_replacement(self):
        self.options.cpu_offload = self.optimization.CPUOffload.SUBMODULE
        first = self.pipeline([self.control_a])
        first.generate()
        old_wrapper = first.controlnet
        old_hooks = {model: model._hf_hook for model in old_wrapper.modules()}

        # Inspect the loader boundary before apply() can overwrite stale hooks
        # on the shared children and hide a missing recursive detach.
        second = self.loader.load_model(
            self.owner, AutoText, self.base, self.options, Scheduler.DEFAULT,
            controlnet=[self.control_a])
        self.assertIsNot(second.controlnet, old_wrapper)
        self.assertIs(second.controlnet.nets[0], old_wrapper.nets[0])
        for model, hook in old_hooks.items():
            with self.subTest(module=model.name):
                self.assertTrue(hook.detached, "old nested offload state must be restored")
                self.assertFalse(hasattr(model, "_hf_hook"), "stale hooks must be removed before apply")
                self.assertEqual(model.device, Device("cpu"))

        second = self.options.apply(second, "cuda")
        for model in second.controlnet.modules():
            with self.subTest(rehooked_module=model.name):
                self.assertTrue(hasattr(model, "_hf_hook"))
                self.assertIsNot(model._hf_hook, old_hooks.get(model))
        self.assertEqual(second.generate().device, Device("cuda"))


class ValidationUITests(unittest.TestCase):
    def test_incompatible_model_download_uses_current_half_precision_preference(self):
        tree = ast.parse((SOURCE / "diffusers_backend.py").read_text(encoding="utf-8"))
        backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffusersBackend")
        validate = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
        class FixItError(Exception):
            class Solution:
                pass
            def __init__(self, message, solution):
                super().__init__(message)
                self.solution = solution
            ChangeProperty = staticmethod(lambda name: NS(name=name))
        target = NS(recommended_model=lambda: "depth-model")
        wrong_type = NS(matches_task=lambda _: False, name="PROMPT_TO_IMAGE")
        lookup = {"base": NS(model_type=wrong_type)}
        class Lookup(dict):
            @property
            def _models(self):
                return self
        namespace = {"GenerationArguments": object, "model_lookup": Lookup(lookup),
                     "FixItError": FixItError, "ModelType": NS(from_task=lambda _: target),
                     "InstallModel": NS(bl_idname="install"),
                     "StableDiffusionPreferences": NS(bl_idname="dream_textures"),
                     "_template_model_download_progress": lambda *args: False}
        exec(compile(ast.Module(body=[validate], type_ignores=[]), "validation-ui", "exec"), namespace)
        task = type("DepthTask", (), {"name": classmethod(lambda cls: "DEPTH_TO_IMAGE")})()
        for prefer_half in (False, True):
            with self.subTest(prefer_half=prefer_half):
                prefs = NS(prefer_fp16_variant=prefer_half, resume_download=True)
                context = NS(preferences=NS(addons={"dream_textures": NS(preferences=prefs)}))
                operator = NS()
                layout = NS(operator=lambda *args, **kwargs: operator)
                with self.assertRaises(FixItError) as caught:
                    namespace["validate"](None, NS(model=NS(id="base"), task=task))
                caught.exception.solution._draw(None, context, layout)
                self.assertEqual(operator.model, "depth-model")
                self.assertIs(operator.prefer_fp16_variant, prefer_half)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="Source root containing generator_process and diffusers_backend.py")
    args, unittest_args = parser.parse_known_args()
    SOURCE = args.source.resolve()
    print(f"Behavioral simulation against {SOURCE}; no real CUDA or Blender execution.")
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
