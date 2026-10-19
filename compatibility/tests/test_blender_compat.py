"""Run with Blender's Python, without models, CUDA, or network access.

Example:
    blender --background --factory-startup --offline-mode --python-exit-code 1 \
        --python tests/test_blender_compat.py -- /path/to/package

This validates Blender integration. It does not claim a successful GPU generation.
"""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import addon_utils
import bpy
import cycles
import numpy as np


PACKAGE = (
    Path(sys.argv[sys.argv.index("--") + 1])
    if "--" in sys.argv else Path(__file__).resolve().parents[1] / "package"
)
sys.path.insert(0, str(PACKAGE))
ORIGINAL_CYCLES_RENDER = cycles.CyclesRender.render
ORIGINAL_CYCLES_PASSES = cycles.CyclesRender.update_render_passes


def enable_addon():
    module = addon_utils.enable("dream_textures", default_set=True, persistent=False)
    if module is None:
        raise RuntimeError("Dream Textures failed to enable; see the preceding traceback.")
    return module


class BlenderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = enable_addon()

    def setUp(self):
        self.images = []
        self.trees = []

    def tearDown(self):
        for tree in self.trees:
            bpy.data.node_groups.remove(tree, do_unlink=True)
        for image in self.images:
            bpy.data.images.remove(image, do_unlink=True)

    def new_tree(self, name="Compatibility Test"):
        tree = bpy.data.node_groups.new(name, "DreamTexturesNodeTree")
        self.trees.append(tree)
        return tree

    def test_01_registration_without_ai_dependencies(self):
        self.assertTrue(hasattr(bpy.types.Scene, "dream_textures_prompt"))
        self.assertTrue(hasattr(bpy.types.Scene, "dream_textures_render_engine"))
        self.assertIn("dream_textures", bpy.context.preferences.addons)
        self.assertIsNot(cycles.CyclesRender.render, ORIGINAL_CYCLES_RENDER)
        self.assertIsNot(cycles.CyclesRender.update_render_passes, ORIGINAL_CYCLES_PASSES)

    def test_02_group_input_execution_uses_current_interface(self):
        from dream_textures.engine.node_executor import execute
        from dream_textures.engine.node_tree import input_sockets

        tree = self.new_tree()
        # A panel in the interface must not be mistaken for an input socket.
        panel = tree.interface.new_panel("Settings")
        value = tree.interface.new_socket(
            name="Value", in_out='INPUT', socket_type='NodeSocketFloat', parent=panel
        )
        value.default_value = 0.375
        tree.interface.new_socket(name="Result", in_out='OUTPUT', socket_type='NodeSocketFloat')
        group_input = tree.nodes.new("NodeGroupInput")
        group_output = tree.nodes.new("NodeGroupOutput")
        tree.links.new(group_input.outputs["Value"], group_output.inputs["Result"])
        bpy.context.scene.dream_textures_render_engine.node_tree = tree

        self.assertEqual([socket.name for socket in input_sockets(tree)], ["Value"])
        self.assertEqual(execute(tree, bpy.context.evaluated_depsgraph_get()), [("Result", 0.375)])

    def test_03_new_tree_operator_assigns_its_own_tree(self):
        # Lexical ordering must not select an unrelated existing node group.
        unrelated = self.new_tree("ZZZ Existing User Tree")
        result = bpy.ops.dream_textures.new_engine_node_tree()
        self.assertEqual(result, {'FINISHED'})
        tree = bpy.context.scene.dream_textures_render_engine.node_tree
        self.assertIsNotNone(tree)
        self.assertNotEqual(tree, unrelated)
        self.trees.append(tree)
        self.assertEqual(len(unrelated.nodes), 0)
        self.assertEqual(len(tree.nodes), 3)
        self.assertEqual(len(tree.links), 3)
        self.assertEqual([item.name for item in tree.interface.items_tree], ["Image"])

    def test_04_image_pixels_round_trip_and_orientation(self):
        from dream_textures import image_utils

        pixels = np.array(
            [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]],
             [[0.0, 0.0, 1.0, 0.25], [0.5, 0.25, 0.75, 1.0]]],
            dtype=np.float32,
        )
        image = image_utils.np_to_bpy(pixels, name="Compatibility Pixels", float_buffer=True)
        self.images.append(image)
        actual = image_utils.bpy_to_np(image, color_space=None)
        np.testing.assert_allclose(actual, pixels, atol=1e-7)
        self.assertTrue(image.packed_file)

    def test_05_integer_image_conversion_with_numpy_2(self):
        from dream_textures import image_utils

        pixels = np.array([0, 127, 255], dtype=np.uint8)
        np.testing.assert_array_equal(
            image_utils.to_dtype(pixels, np.uint16), np.array([0, 32639, 65535], dtype=np.uint16)
        )
        np.testing.assert_allclose(
            image_utils.to_dtype(pixels, np.float32), [0.0, 127.0 / 255.0, 1.0]
        )

    def test_06_render_pass_color_conversion(self):
        from dream_textures import image_utils

        class Rect:
            def foreach_set(self, pixels):
                self.pixels = pixels.copy()

        render_pass = SimpleNamespace(channels=4, name="Combined", rect=Rect())
        pixels = np.array([[[0.0, 0.5, 1.0, 0.75]]], dtype=np.float32)
        image_utils.np_to_render_pass(pixels, render_pass, color_space="sRGB")
        np.testing.assert_allclose(
            render_pass.rect.pixels, [0.0, 0.21404114, 1.0, 0.75], atol=1e-7
        )

    def test_07_render_failure_releases_the_render_waiter(self):
        from dream_textures import render_pass

        failure = RuntimeError("simulated model load failure")
        backend = SimpleNamespace(generate=lambda args, **callbacks: callbacks["callback"](failure))
        prompt = SimpleNamespace(
            strength=0.5, get_backend=lambda: backend, generate_args=lambda context: SimpleNamespace()
        )
        scene = SimpleNamespace(
            dream_textures_render_properties_prompt=prompt,
            dream_textures_render_properties_pass_inputs="color",
        )
        layer = SimpleNamespace(passes={"Combined": object()})
        engine = SimpleNamespace(update_stats=lambda *args: None)
        raised = []

        def generate():
            try:
                render_pass._render_dream_textures_pass(engine, layer, (1, 1), scene, object(), object())
            except Exception as error:
                raised.append(error)

        with patch.object(render_pass.image_utils, "render_pass_to_np", return_value=np.zeros((1, 1, 4))):
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "Backend error left render waiting indefinitely")
        self.assertEqual(raised, [failure])

    def test_08_inpaint_preserves_selected_brush_settings(self):
        from dream_textures.operators.inpaint_area_brush import (
            configure_inpaint_brush, restore_inpaint_brush, unified_paint_settings,
        )

        brush = bpy.data.brushes.new("Compatibility Brush", mode='TEXTURE_PAINT')
        try:
            brush.image_brush_type = 'SOFTEN'
            brush.blend = 'MUL'
            brush.curve_distance_falloff_preset = 'SMOOTH'
            brush.strength = 0.25
            original = configure_inpaint_brush(brush)
            self.assertEqual(brush.image_brush_type, 'DRAW')
            self.assertEqual(brush.blend, 'ERASE_ALPHA')
            self.assertEqual(brush.curve_distance_falloff_preset, 'CONSTANT')
            self.assertEqual(brush.strength, 1.0)
            restore_inpaint_brush(brush, original)
            self.assertEqual(brush.image_brush_type, 'SOFTEN')
            self.assertEqual(brush.blend, 'MUL')
            self.assertEqual(brush.curve_distance_falloff_preset, 'SMOOTH')
            self.assertEqual(brush.strength, 0.25)
            self.assertEqual(
                unified_paint_settings(bpy.context),
                bpy.context.tool_settings.image_paint.unified_paint_settings,
            )
        finally:
            bpy.data.brushes.remove(brush)

    def test_09_ui_poll_does_not_start_uninstalled_runtime(self):
        from dream_textures.generator_process import Generator

        with patch.object(Generator, "runtime_available", return_value=False), patch.object(
            Generator, "shared", side_effect=AssertionError("Polling must not start the worker")
        ):
            self.assertFalse(bpy.ops.shade.dream_texture.poll())
            self.assertFalse(bpy.ops.shade.dream_texture_project.poll())
            self.assertFalse(bpy.ops.shade.dream_textures_upscale.poll())
            self.assertFalse(bpy.ops.dream_textures.control_net_bake.poll())
            bpy.context.scene.seamless_result.check(None)

    def test_10_builtin_presets_use_backend_properties(self):
        import runpy
        from dream_textures.ui.presets import PRESETS_PATH

        for name, expected_steps in (("Preview", 20), ("Final", 50), ("Debug", 20)):
            with self.subTest(preset=name):
                runpy.run_path(str(Path(PRESETS_PATH) / f"{name}.py"))
                prompt = bpy.context.scene.dream_textures_prompt
                self.assertEqual(prompt.steps, expected_steps)
                self.assertTrue(prompt.get_backend().half_precision)
                self.assertTrue(prompt.get_backend().sdp_attention)

    def test_90_disable_and_reenable(self):
        addon_utils.disable("dream_textures", default_set=True)
        self.assertFalse(hasattr(bpy.types.Scene, "dream_textures_prompt"))
        self.assertFalse(hasattr(bpy.types.Scene, "dream_textures_engine_prompt"))
        self.assertFalse(hasattr(bpy.types.Scene, "dream_textures_render_engine"))
        self.assertIs(cycles.CyclesRender.render, ORIGINAL_CYCLES_RENDER)
        self.assertIs(cycles.CyclesRender.update_render_passes, ORIGINAL_CYCLES_PASSES)
        self.__class__.addon = enable_addon()
        self.assertTrue(hasattr(bpy.types.Scene, "dream_textures_prompt"))

    @classmethod
    def tearDownClass(cls):
        addon_utils.disable("dream_textures", default_set=True)


if __name__ == "__main__":
    print(f"Blender {bpy.app.version_string}; Python {sys.version}; NumPy {np.__version__}")
    print(f"Package: {PACKAGE}")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BlenderCompatibilityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
