"""Offline backend integration checks using tiny, randomly initialized models.

This validates Python/dependency APIs and real CPU inference. It does not validate
Windows DLL loading, GPU kernels, pretrained model quality, or Blender UI flows.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--runtime", required=True)
parser.add_argument("--addon", required=True)
args, unittest_args = parser.parse_known_args()
runtime = Path(args.runtime).resolve()
addon = Path(args.addon).resolve()
sys.path.insert(0, str(runtime))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "dream-textures-tests" / "hf-cache"))
os.environ.setdefault("BLENDER_VERSION", "5.2.2")
os.environ.setdefault("BLENDER_OCIO_CONFIG", "")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dream-textures-tests" / "matplotlib-cache"))

# Import the actual action and image modules without launching Blender's actor.
# The actor and Blender registration are tested independently by the build.
for suffix in ("", ".api", ".api.models", ".generator_process", ".generator_process.actions"):
    module = types.ModuleType("dream_textures_test" + suffix)
    module.__path__ = [str(addon.joinpath(*suffix.strip(".").split(".")))] if suffix else [str(addon)]
    sys.modules[module.__name__] = module


class DirectCalls(Exception):
    def __new__(cls, function=None):
        return super().__new__(cls) if function is None else function

    @staticmethod
    def always(function):
        return function

    @staticmethod
    def when(condition):
        return lambda function: function

    @staticmethod
    def when_raised(function):
        return function


sys.modules["dream_textures_test.generator_process"].RunInSubprocess = DirectCalls

import numpy as np
import torch
import diffusers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from dream_textures_test.generator_process.models import Optimizations, CPUOffload, Scheduler, UpscaleTiler
from dream_textures_test.api.models.seamless_axes import SeamlessAxes
from dream_textures_test.api.models.step_preview_mode import StepPreviewMode
from dream_textures_test.generator_process.models.upscale_tiler import tiled_decode_latents
from dream_textures_test.generator_process.actions.prompt_to_image import _configure_model_padding

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def action(name):
    return getattr(importlib.import_module("dream_textures_test.generator_process.actions." + name), name)


class BackendCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="dreamtextures-backend-")
        folder = Path(cls.temp.name)
        (folder / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}))
        (folder / "merges.txt").write_text("#version: 0.2\n")
        cls.tokenizer = CLIPTokenizer(str(folder / "vocab.json"), str(folder / "merges.txt"), model_max_length=16)
        cls.image = np.random.default_rng(42).random((16, 16, 4), dtype=np.float32)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def components(self, channels=4, vae_blocks=(16, 32), class_embed_type=None, xl=False):
        torch.manual_seed(42)
        text_config = CLIPTextConfig(
            vocab_size=3, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=16, bos_token_id=0,
            eos_token_id=1, pad_token_id=1, projection_dim=16,
        )
        unet_options = dict(
            sample_size=8, in_channels=channels, out_channels=4, layers_per_block=1,
            block_out_channels=(16, 32), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), norm_num_groups=4,
            cross_attention_dim=32 if xl else 16, attention_head_dim=4,
        )
        if class_embed_type:
            unet_options["class_embed_type"] = class_embed_type
        if xl:
            unet_options.update(addition_embed_type="text_time", addition_time_embed_dim=4,
                                projection_class_embeddings_input_dim=40)
        components = dict(
            unet=diffusers.UNet2DConditionModel(**unet_options),
            vae=diffusers.AutoencoderKL(
                in_channels=3, out_channels=3, latent_channels=4, layers_per_block=1,
                block_out_channels=vae_blocks,
                down_block_types=("DownEncoderBlock2D",) * len(vae_blocks),
                up_block_types=("UpDecoderBlock2D",) * len(vae_blocks),
                norm_num_groups=4, sample_size=16,
            ),
            text_encoder=CLIPTextModel(text_config), tokenizer=self.tokenizer,
            scheduler=diffusers.DDIMScheduler(clip_sample=False, set_alpha_to_one=False),
        )
        if xl:
            components.update(text_encoder_2=CLIPTextModelWithProjection(text_config), tokenizer_2=self.tokenizer)
        else:
            components.update(safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        return components

    def pipe(self, cls=diffusers.StableDiffusionPipeline, **kwargs):
        pipe = cls(**self.components(**kwargs))
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def run_action(self, name, pipeline, **overrides):
        class Harness:
            def choose_device(self, optimizations):
                return "cpu"

            def load_model(self, model_class, *args, **kwargs):
                return pipeline(model_class) if callable(pipeline) and not isinstance(pipeline, diffusers.DiffusionPipeline) else pipeline

            def detect_seamless(self, image):
                return SeamlessAxes.OFF

        options = dict(
            model="offline-tiny-random", scheduler=Scheduler.DDIM,
            optimizations=Optimizations(cpu_only=True, cpu_offload=CPUOffload.OFF),
            prompt="a", steps=2, width=16, height=16, seed=42, cfg_scale=7.5,
            use_negative_prompt=True, negative_prompt="", seamless_axes=SeamlessAxes.OFF,
            iterations=1, step_preview_mode=StepPreviewMode.FAST,
        )
        options.update(overrides)
        generator = action(name)(Harness(), **options)
        future = next(generator)
        future.check_cancelled = lambda: False
        list(generator)
        self.assertTrue(future.done)
        results = future.result(last_only=True)
        self.assertIsNotNone(results)
        image = results[-1].image
        self.assertTrue(np.isfinite(image).all())
        self.assertEqual(image.shape[-1], 3 if name != "upscale" else 4)
        return image, future

    def test_01_dependency_and_conversion_imports(self):
        import controlnet_aux
        import cv2
        import PyOpenColorIO
        import scipy
        import skimage
        import yaml
        from transformers import CLIPSegForImageSegmentation, DPTForDepthEstimation
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
            download_controlnet_from_original_ckpt, download_from_original_stable_diffusion_ckpt,
        )
        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
        from diffusers.utils.hub_utils import old_diffusers_cache
        from diffusers.pipelines.pipeline_utils import variant_compatible_siblings
        from huggingface_hub import cached_download
        from safetensors.torch import save_file, load_file
        checkpoint = Path(self.temp.name) / "tiny.safetensors"
        value = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        save_file({"weight": value}, str(checkpoint))
        self.assertTrue(torch.equal(load_file(str(checkpoint))["weight"], value))
        self.assertIsInstance(yaml.safe_load((addon / "sd_configs/v1-inference.yaml").read_text()), dict)
        self.assertTrue(Optimizations.device_supports("half_precision", "cuda"))
        self.assertFalse(Optimizations.device_supports("half_precision", "cpu"))

    def test_02_seamless_convolutions(self):
        conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        image = torch.randn(1, 3, 12, 16)
        initial = conv(image)
        for axes in (SeamlessAxes.HORIZONTAL, SeamlessAxes.VERTICAL, SeamlessAxes.BOTH):
            _configure_model_padding(conv, axes)
            actual = conv(image)
            x_mode = "circular" if axes.x else "constant"
            y_mode = "circular" if axes.y else "constant"
            padded = torch.nn.functional.pad(image, (1, 1, 0, 0), mode=x_mode)
            padded = torch.nn.functional.pad(padded, (0, 0, 1, 1), mode=y_mode)
            expected = torch.nn.functional.conv2d(padded, conv.weight, conv.bias)
            torch.testing.assert_close(actual, expected)
        _configure_model_padding(conv, SeamlessAxes.OFF)
        torch.testing.assert_close(conv(image), initial)

    def test_03_tiled_vae_and_accurate_preview(self):
        pipe = self.pipe()
        latents = torch.randn(1, 4, 8, 8)
        opts = Optimizations(cpu_only=True, vae_tiling="full")
        with torch.inference_mode():
            expected = pipe.vae.decode(latents).sample
            actual = tiled_decode_latents(pipe, latents, pre_patch=pipe.vae.decode, optimizations=opts)
            torch.testing.assert_close(actual.sample, expected)
            result = tiled_decode_latents(pipe, latents, False, pre_patch=pipe.vae.decode, optimizations=opts)
            torch.testing.assert_close(result[0], expected)
        opts.vae_tiling = "manual"
        opts.vae_tile_size = 8
        opts.vae_tile_blend = 2
        image, future = self.run_action("prompt_to_image", pipe, optimizations=opts, step_preview_mode=StepPreviewMode.ACCURATE)
        self.assertEqual(image.shape, (16, 16, 3))
        self.assertGreater(len(future._responses), 1)

    def test_04_text_to_image_reproducible_and_seamless(self):
        first, _ = self.run_action("prompt_to_image", self.pipe(), seamless_axes=SeamlessAxes.BOTH)
        second, _ = self.run_action("prompt_to_image", self.pipe(), seamless_axes=SeamlessAxes.BOTH)
        self.assertEqual(first.shape, (16, 16, 3))
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_05_image_to_image(self):
        pipe = self.pipe(diffusers.StableDiffusionImg2ImgPipeline)
        image, _ = self.run_action("image_to_image", pipe, image=self.image, fit=False, strength=1.0)
        self.assertEqual(image.shape, (16, 16, 3))

    def test_06_inpaint_alpha(self):
        pipe = self.pipe(diffusers.StableDiffusionInpaintPipeline, channels=9)
        image, _ = self.run_action("inpaint", pipe, image=self.image, fit=False, strength=1.0,
                                   inpaint_mask_src="alpha", text_mask="", text_mask_confidence=0.5)
        self.assertEqual(image.shape, (16, 16, 3))

    def test_07_depth_to_image(self):
        for guidance in (7.5, 1.0):
            with self.subTest(guidance=guidance):
                image, _ = self.run_action("depth_to_image", lambda cls: self.pipe(cls, channels=5),
                                           image=None, depth=self.image[..., 0], strength=1.0, cfg_scale=guidance)
                self.assertEqual(image.shape, (16, 16, 3))

    def test_08_controlnet_generation(self):
        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
        components = self.components()
        controlnet = diffusers.ControlNetModel.from_unet(components["unet"], conditioning_embedding_out_channels=(4, 8))
        pipe = diffusers.StableDiffusionControlNetPipeline(**components, controlnet=MultiControlNetModel([controlnet]))
        pipe.set_progress_bar_config(disable=True)
        image, _ = self.run_action("control_net", pipe, control_net=["tiny-control"], control=[self.image],
                                   controlnet_conditioning_scale=[1.0], image=None, inpaint=False,
                                   inpaint_mask_src="alpha", text_mask="", text_mask_confidence=0.5, strength=1.0)
        self.assertEqual(image.shape, (16, 16, 3))

    def test_09_controlnet_preprocessors(self):
        from controlnet_aux.processor import Processor
        from PIL import Image
        image = Image.fromarray((self.image[..., :3] * 255).astype(np.uint8)).resize((64, 64))
        for name in ("canny", "shuffle"):
            with self.subTest(processor=name):
                processed = Processor(name)(image)
                self.assertEqual(np.asarray(processed).ndim, 3)

    def test_10_watermark_round_trip(self):
        from imwatermark import WatermarkEncoder, WatermarkDecoder
        image = np.random.default_rng(9).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", b"DT52")
        encoded = encoder.encode(image, "dwtDct")
        decoded = WatermarkDecoder("bytes", 32).decode(encoded, "dwtDct")
        self.assertEqual(decoded, b"DT52")

    def test_11_sdxl_text_to_image(self):
        pipe = self.pipe(diffusers.StableDiffusionXLPipeline, xl=True)
        image, _ = self.run_action("prompt_to_image", pipe)
        self.assertEqual(image.shape, (16, 16, 3))

    def test_12_upscale(self):
        components = self.components(channels=7, vae_blocks=(16, 32, 32), class_embed_type="timestep")
        components.pop("requires_safety_checker")
        pipe = diffusers.StableDiffusionUpscalePipeline(**components,
                    low_res_scheduler=diffusers.DDPMScheduler(), watermarker=None)
        pipe.set_progress_bar_config(disable=True)
        image, _ = self.run_action("upscale", pipe, image=self.image, tile_size=16, blend=2)
        self.assertEqual(image.shape, (64, 64, 4))

    def test_13_download_progress_unknown_lengths(self):
        from dream_textures_test.generator_process.actions.huggingface_hub import DownloadStatus
        from dream_textures_test.generator_process.future import Future
        from huggingface_hub import file_download
        original = file_download.tqdm
        future = Future()
        try:
            DownloadStatus.hook_download_tqdm(future)
            for total in (None, 0, 10):
                with file_download.tqdm(total=total, disable=True) as progress:
                    progress.update(1)
            self.assertTrue(future._responses)
            self.assertTrue(all(response.total > 0 for response in future._responses))
        finally:
            file_download.tqdm = original

    def test_14_numpy2_grayscale_pillow_round_trip(self):
        from dream_textures_test.image_utils import np_to_pil, pil_to_np
        pixels = np.array([[0, 32, 64], [128, 192, 255]], dtype=np.uint8)
        for dtype in (np.uint8, np.float32):
            source = pixels if dtype is np.uint8 else pixels.astype(np.float32) / 255
            for singleton_channel in (False, True):
                for mode in (None, "L"):
                    with self.subTest(dtype=dtype.__name__, singleton=singleton_channel, mode=mode):
                        array = source[..., None] if singleton_channel else source
                        image = np_to_pil(array, mode=mode)
                        self.assertEqual(image.mode, "L")
                        self.assertEqual(image.size, (3, 2))
                        np.testing.assert_array_equal(pil_to_np(image, dtype=np.uint8), pixels)


if __name__ == "__main__":
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                      "diffusers": diffusers.__version__, "numpy": np.__version__,
                      "device": "cpu", "pretrained_weights_downloaded": False}), flush=True)
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
