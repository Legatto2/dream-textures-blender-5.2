"""Focused projection checks using NumPy and simulated Blender/GPU objects.

Runs the actual pure helpers and AST-extracted capture/operator functions.
No Blender, GPU drawing, model download, Torch, or neural inference is executed.
Run: python3 -B compatibility/tests/test_projection_inputs.py
"""

import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_source(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


helpers = load_source("projection_helpers_under_test", ROOT / "projection_utils.py")
tasks = load_source("projection_tasks_under_test", ROOT / "api/models/task.py")


def extract(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def project_function(name, namespace, class_name=None):
    base = {
        "np": np, "os": os, "tempfile": tempfile,
        "temporary_render_settings": helpers.temporary_render_settings,
        "validate_color_capture": helpers.validate_color_capture,
        "validate_depth_capture": helpers.validate_depth_capture,
        "projection_task": helpers.projection_task,
        "_CAPTURE_OWNER": "dream_textures_projection_capture",
    }
    base.update(namespace)
    return extract(ROOT / "operators/project.py", name, base, class_name)


def context_fixture(width=2, height=2):
    render = NS(
        resolution_x=1920, resolution_y=1080, resolution_percentage=25,
        filepath="previous.exr", use_border=True, use_crop_to_border=True,
        image_settings=NS(file_format="OPEN_EXR", color_mode="RGBA", color_depth="32"),
    )
    region = NS(type="WINDOW", width=width, height=height)
    ui_region = NS(type="UI", width=240, height=height)
    view = NS(view_matrix="view matrix", window_matrix="projection matrix")
    space = NS(type="VIEW_3D", overlay=NS(show_overlays=True), show_gizmo=True, region_3d=view)
    area = NS(type="VIEW_3D", regions=[ui_region, region])
    context = NS(
        area=area, region=ui_region, region_data=None, space_data=space,
        scene=NS(render=render, dream_textures_info="", dream_textures_progress=0,
                 dream_textures_project_framebuffer_arguments="color"),
        evaluated_depsgraph_get=lambda: "evaluated scene",
        temp_override=Mock(side_effect=lambda **kwargs: nullcontext()),
    )
    return context, region, space, view


def settings_snapshot(context, space):
    render = context.scene.render
    return (render.resolution_x, render.resolution_y, render.resolution_percentage,
            render.filepath, render.image_settings.file_format,
            render.image_settings.color_mode, render.image_settings.color_depth,
            render.use_border, render.use_crop_to_border,
            space.overlay.show_overlays, space.show_gizmo)


class DepthNormalizationTests(unittest.TestCase):
    def test_empty_mask_is_black_in_both_directions(self):
        for invert in (False, True):
            result = helpers.normalize_depth(np.ones((2, 3)), np.zeros((2, 3)), invert=invert)
            np.testing.assert_array_equal(result, np.zeros((2, 3)))
            with self.assertRaisesRegex(ValueError, "empty"):
                helpers.validate_depth_capture(result)

    def test_flat_geometry_is_mid_depth_and_keeps_background_black(self):
        mask = np.array([[0, 1, 1], [0, 1, 0]])
        for raw in (0.0, 0.7, 1.0):
            for invert in (False, True):
                result = helpers.normalize_depth(np.full(mask.shape, raw), mask, invert=invert)
                np.testing.assert_allclose(result, mask * 0.5)
                helpers.validate_depth_capture(result)

    def test_geometry_endpoints_and_inversion_ignore_background(self):
        raw = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 1.0]])
        mask = np.array([[0, 1, 1], [1, 1, 0]])
        normal = helpers.normalize_depth(raw, mask, invert=False)
        inverse = helpers.normalize_depth(raw, mask, invert=True)
        np.testing.assert_allclose(normal, [[0, 0, 1/3], [2/3, 1, 0]])
        np.testing.assert_allclose(inverse, [[0, 1, 2/3], [1/3, 0, 0]], atol=1e-7)
        self.assertEqual(normal.dtype, np.float32)

    def test_nonfinite_pixels_are_excluded_and_invalid_shapes_are_rejected(self):
        result = helpers.normalize_depth([[np.nan, np.inf], [-np.inf, 0.4]], np.ones((2, 2)))
        np.testing.assert_array_equal(result, [[0, 0], [0, 0.5]])
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            helpers.normalize_depth(np.zeros((2, 3)), np.ones((2, 2)))


class RenderSettingsTests(unittest.TestCase):
    def test_capture_exception_restores_resolution_format_border_and_overlays(self):
        context, _, space, _ = context_fixture()
        context.scene.render.image_settings.file_format = "PNG"
        context.scene.render.image_settings.color_mode = "BW"
        context.scene.render.image_settings.color_depth = "8"
        before = settings_snapshot(context, space)
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with helpers.temporary_render_settings(context.scene.render, space, (640, 384), "capture.png"):
                self.assertEqual(context.scene.render.resolution_percentage, 100)
                self.assertEqual(context.scene.render.image_settings.file_format, "PNG")
                self.assertEqual(context.scene.render.image_settings.color_mode, "RGBA")
                self.assertFalse(context.scene.render.use_border)
                self.assertFalse(context.scene.render.use_crop_to_border)
                self.assertFalse(space.overlay.show_overlays)
                self.assertFalse(space.show_gizmo)
                raise RuntimeError("capture failed")
        self.assertEqual(settings_snapshot(context, space), before)

    def test_restore_continues_after_one_setting_raises(self):
        context, _, space, _ = context_fixture()
        class FragileRender:
            def __init__(self, source):
                self.__dict__.update(source.__dict__)
                self.fail_restore = False
            def __setattr__(self, name, value):
                if name == "resolution_x" and value == 1920 and getattr(self, "fail_restore", False):
                    raise RuntimeError("read-only simulated setting")
                object.__setattr__(self, name, value)
        render = FragileRender(context.scene.render)
        with self.assertRaisesRegex(RuntimeError, "restore projection capture settings"):
            with helpers.temporary_render_settings(render, space, (512, 512), "capture.png"):
                render.fail_restore = True
        self.assertEqual(render.resolution_y, 1080)
        self.assertEqual(render.resolution_percentage, 25)
        self.assertEqual(render.filepath, "previous.exr")
        self.assertEqual(render.image_settings.file_format, "OPEN_EXR")
        self.assertTrue(space.overlay.show_overlays)
        self.assertTrue(space.show_gizmo)

    def test_invalid_size_does_not_change_settings(self):
        context, _, space, _ = context_fixture()
        before = settings_snapshot(context, space)
        with self.assertRaisesRegex(ValueError, "positive size"):
            with helpers.temporary_render_settings(context.scene.render, space, (0, 512), "capture.png"):
                self.fail("invalid capture size entered context")
        self.assertEqual(settings_snapshot(context, space), before)


class ProjectionCaptureTests(unittest.TestCase):
    def test_view_uses_invoking_area_not_another_viewport(self):
        context, region, space, view = context_fixture(640, 384)
        context.screen = NS(areas=[context.area, NS(type="VIEW_3D", regions=[NS(type="WINDOW", width=10, height=20)])])
        resolve = project_function("_projection_view", {})
        result = resolve(context)
        self.assertIs(result[0], region)
        self.assertIs(result[1], space)
        self.assertIs(result[2], view)

    def test_render_failure_restores_settings_and_removes_temporary_directory(self):
        context, region, space, _ = context_fixture()
        before = settings_snapshot(context, space)
        paths = []
        def render_failure(**kwargs):
            path = context.scene.render.filepath
            paths.append(path)
            self.assertTrue(Path(path).parent.is_dir())
            raise RuntimeError("OpenGL capture failed")
        images = NS(load=Mock(), remove=Mock())
        capture = project_function("_capture_projection_color", {
            "bpy": NS(ops=NS(render=NS(opengl=render_failure)), data=NS(images=images)),
            "image_utils": NS(), "_store_projection_capture": Mock(),
        })
        with self.assertRaisesRegex(RuntimeError, "OpenGL capture failed"):
            capture(context, region, space)
        self.assertEqual(settings_snapshot(context, space), before)
        self.assertFalse(Path(paths[0]).parent.exists())
        images.load.assert_not_called()
        context.temp_override.assert_called_once_with(area=context.area, region=region)

    def test_color_uses_real_top_to_bottom_conversion_and_removes_loaded_image(self):
        context, region, space, _ = context_fixture()
        bottom_up = np.array([[[1, 0, 0, 1], [1, 0, 0, 1]],
                              [[0, 0, 1, 1], [0, 0, 1, 1]]], dtype=np.float32)
        loaded = NS(type="IMAGE", size=(2, 2), channels=4,
                    colorspace_settings=NS(name="sRGB"),
                    pixels=NS(foreach_get=lambda target: target.__setitem__(slice(None), bottom_up.ravel())))
        to_numpy = extract(ROOT / "image_utils.py", "bpy_to_np", {
            "np": np, "NDArray": np.ndarray, "rgba": lambda value: value,
            "color_transform": lambda array, *args, **kwargs: array,
        })
        images = NS(load=Mock(return_value=loaded), remove=Mock())
        store = Mock()
        capture = project_function("_capture_projection_color", {
            "bpy": NS(ops=NS(render=NS(opengl=Mock())), data=NS(images=images)),
            "image_utils": NS(bpy_to_np=to_numpy), "_store_projection_capture": store,
        })
        result = capture(context, region, space)
        np.testing.assert_array_equal(result, np.flipud(bottom_up))
        np.testing.assert_array_equal(store.call_args.args[0], result)
        self.assertEqual(store.call_args.args[1], "color")
        images.remove.assert_called_once_with(loaded)
        self.assertFalse(Path(images.load.call_args.args[0]).parent.exists())

    def test_mismatched_color_dimensions_reject_and_cleanup(self):
        context, region, space, _ = context_fixture()
        loaded = object()
        images = NS(load=Mock(return_value=loaded), remove=Mock())
        store = Mock()
        capture = project_function("_capture_projection_color", {
            "bpy": NS(ops=NS(render=NS(opengl=Mock())), data=NS(images=images)),
            "image_utils": NS(bpy_to_np=lambda *args, **kwargs: np.zeros((3, 2, 4))),
            "_store_projection_capture": store,
        })
        with self.assertRaisesRegex(ValueError, "does not match"):
            capture(context, region, space)
        images.remove.assert_called_once_with(loaded)
        store.assert_not_called()

    def test_owned_diagnostics_do_not_overwrite_an_unrelated_name_collision(self):
        class Image(dict):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.use_fake_user = False
        user_image = Image("Dream Textures Projection Depth")
        images = [user_image]
        calls = []
        def to_blender(array, name, existing, **kwargs):
            calls.append((existing, kwargs))
            if existing is not None:
                return existing
            result = Image(name + ".001")
            images.append(result)
            return result
        store = project_function("_store_projection_capture", {
            "bpy": NS(data=NS(images=images)),
            "image_utils": NS(np_to_bpy=to_blender, rgba=lambda value: value),
        })
        first = store(np.full((2, 2), 0.5), "depth")
        self.assertIsNot(first, user_image)
        self.assertEqual(user_image, {})
        self.assertTrue(first.use_fake_user)
        self.assertIs(store(np.ones((2, 2)), "depth"), first)
        self.assertIsNone(calls[0][0])
        self.assertIs(calls[1][0], first)
        self.assertEqual(calls[0][1]["color_space"], "Non-Color")

    def test_empty_capture_rejects_before_material_uv_or_color_work(self):
        context, region, space, view = context_fixture()
        context.scene.dream_textures_project_prompt = NS(steps=20)
        color = Mock(side_effect=AssertionError("must not capture color after empty depth"))
        diagnostics = Mock()
        capture = project_function("_capture_projection_inputs", {
            "render_depth_map": Mock(return_value=np.zeros((2, 2))),
            "_store_projection_capture": diagnostics, "_capture_projection_color": color,
        })
        material_new = Mock(side_effect=AssertionError("must not mutate materials"))
        execute = project_function("execute", {
            "bpy": NS(types=NS(Scene=type("Scene", (), {})), props=NS(IntProperty=lambda **kwargs: None),
                      data=NS(materials=NS(new=material_new))),
            "_projection_view": lambda _: (region, space, view),
            "_capture_projection_inputs": capture, "logger": Mock(),
        }, "ProjectDreamTexture")
        operator = NS(report=Mock())
        self.assertEqual(execute(operator, context), {"CANCELLED"})
        material_new.assert_not_called()
        color.assert_not_called()
        diagnostics.assert_called_once()
        self.assertEqual(context.scene.dream_textures_info, "")
        self.assertEqual(context.scene.dream_textures_progress, 0)
        self.assertIn("empty", operator.report.call_args.args[1])

    def test_operator_explicitly_dispatches_each_input_mode_before_material_work(self):
        class StopBeforeMaterials(Exception):
            pass
        for control_net in (False, True):
            for color in (False, True):
                with self.subTest(control_net=control_net, color=color):
                    context, region, space, view = context_fixture()
                    depth = np.full((2, 2), 0.5)
                    image = np.ones((2, 2, 4)) if color else None
                    arguments = NS(task="unrelated hidden init-image task")
                    backend = NS(validate=Mock())
                    prompt = NS(steps=20, strength=0.6, get_backend=lambda: backend,
                                generate_args=Mock(return_value=arguments))
                    context.scene.dream_textures_project_prompt = prompt
                    context.scene.dream_textures_project_use_control_net = control_net
                    execute = project_function("execute", {
                        "bpy": NS(types=NS(Scene=type("Scene", (), {})), props=NS(IntProperty=lambda **kwargs: None),
                                  data=NS(materials=NS(new=Mock(side_effect=StopBeforeMaterials)))),
                        "_projection_view": lambda _: (region, space, view),
                        "_capture_projection_inputs": lambda *args: (depth, image),
                        "api": tasks, "image_utils": NS(rgba=lambda value: value), "logger": Mock(),
                    }, "ProjectDreamTexture")
                    with self.assertRaises(StopBeforeMaterials):
                        execute(NS(report=Mock()), context)
                    expected = tasks.ImageToImage if control_net and color else tasks.PromptToImage if control_net else tasks.DepthToImage
                    self.assertIsInstance(arguments.task, expected)
                    if hasattr(arguments.task, "image"):
                        self.assertIs(arguments.task.image, image)
                        self.assertEqual(arguments.task.strength, 0.6)
                    backend.validate.assert_called_once_with(arguments)
                    self.assertNotIn("init_image", prompt.generate_args.call_args.kwargs)


class DepthGPUCleanupTests(unittest.TestCase):
    def renderer(self, *, raw=None, mask=None, error=None):
        raw = np.full((2, 2), 0.4) if raw is None else np.asarray(raw)
        mask = np.ones((2, 2)) if mask is None else np.asarray(mask)
        buffer = lambda array: NS(to_list=lambda: array.tolist())
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 3] = mask * 255
        framebuffer = NS(clear=Mock(), read_depth=Mock(side_effect=error, return_value=buffer(raw)),
                         read_color=Mock(return_value=buffer(rgba)))
        state = NS(depth_test="GREATER", depth_mask=False)
        state.depth_test_get = lambda: state.depth_test
        state.depth_mask_get = lambda: state.depth_mask
        state.depth_test_set = lambda value: setattr(state, "depth_test", value)
        state.depth_mask_set = lambda value: setattr(state, "depth_mask", value)
        state.active_framebuffer_get = lambda: framebuffer
        shader = NS(bound=False, color=None)
        shader.bind = lambda: setattr(shader, "bound", True)
        def uniform(name, value):
            self.assertTrue(shader.bound)
            self.assertEqual(name, "color")
            shader.color = value
        shader.uniform_float = uniform
        offscreen = NS(bind=lambda: nullcontext(), free=Mock())
        gpu = NS(state=state, types=NS(GPUOffScreen=lambda *args: offscreen),
                 matrix=NS(push_pop=lambda: nullcontext(), load_matrix=Mock(), load_projection_matrix=Mock()),
                 shader=NS(from_builtin=lambda _: shader))
        timers = NS(register=lambda callback, first_interval: callback())
        render = extract(ROOT / "engine/annotations/depth.py", "render_depth_map", {
            "np": np, "threading": threading, "gpu": gpu, "bpy": NS(app=NS(timers=timers)),
            "UNIFORM_COLOR": "UNIFORM_COLOR", "normalize_depth": helpers.normalize_depth,
            "batch_for_shader": Mock(),
        })
        call = lambda: render(NS(object_instances=[]), width=2, height=2, matrix="view", projection_matrix="projection")
        return call, state, shader, offscreen

    def test_explicit_mask_uniform_and_gpu_state_restoration(self):
        call, state, shader, offscreen = self.renderer(mask=[[0, 1], [1, 0]])
        np.testing.assert_array_equal(call(), [[0, 0.5], [0.5, 0]])
        self.assertEqual(shader.color, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(state.depth_test, "GREATER")
        self.assertFalse(state.depth_mask)
        offscreen.free.assert_called_once()

    def test_gpu_read_failure_restores_state_and_frees_offscreen(self):
        call, state, _, offscreen = self.renderer(error=RuntimeError("depth read failed"))
        with self.assertRaisesRegex(RuntimeError, "depth read failed"):
            call()
        self.assertEqual(state.depth_test, "GREATER")
        self.assertFalse(state.depth_mask)
        offscreen.free.assert_called_once()

    def test_timer_failure_unblocks_waiting_thread_and_propagates(self):
        call, _, _, offscreen = self.renderer(error=RuntimeError("timer depth read failed"))
        errors = []
        def worker():
            try:
                call()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive(), "failed GPU timer callback left the caller waiting")
        self.assertEqual(len(errors), 1)
        self.assertIn("timer depth read failed", str(errors[0]))
        offscreen.free.assert_called_once()


if __name__ == "__main__":
    print("Projection input checks: NumPy + simulated Blender/GPU; no live rendering or CUDA inference.")
    unittest.main(verbosity=2)
