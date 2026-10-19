"""Run with Blender 5.2.2 and the private CPU test runtime; no network/model I/O."""

from pathlib import Path
import json
import os
import sys
import time
import unittest

import addon_utils
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ADDON = Path(os.environ.get("DREAM_TEXTURES_ADDON_SOURCE", ROOT / "package" / "dream_textures"))
if not ADDON.is_dir():
    ADDON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADDON.parent))

# The test caller points BLENDER_USER_CONFIG to runtime/test_config.
os.environ["DREAM_TEXTURES_TEST_ALLOW_CPU"] = "1"
FRONTEND_NUMPY = np.__file__
addon = addon_utils.enable("dream_textures", default_set=True, persistent=False)
if addon is None:
    raise RuntimeError("The add-on could not be enabled.")
from dream_textures.compatibility.tests import actor_probes
from dream_textures.generator_process import Generator
from dream_textures.generator_process import actor
from dream_textures.operators.install_dependencies import session_runtime
actor.main_thread_rendering = False


def wait(future, timeout=30):
    if not future._done_event.wait(timeout):
        Generator.shared_close()
        raise AssertionError("Worker operation did not complete within the bounded test timeout")
    return future.result()


class ActorIntegrationTests(unittest.TestCase):
    def test_01_embedded_python_and_private_dependencies(self):
        self.assertTrue(Generator.runtime_available())
        result = wait(Generator.shared().call(actor_probes.describe))
        print("ACTOR_RUNTIME " + json.dumps(result, sort_keys=True), flush=True)
        self.assertEqual(result["python"][:2], [3, 13])
        self.assertFalse(result["bpy_imported"])
        self.assertTrue(Path(result["torch_path"]).is_relative_to(session_runtime()))
        self.assertTrue(Path(result["numpy_path"]).is_relative_to(session_runtime()))
        self.assertEqual(np.__file__, FRONTEND_NUMPY)
        self.assertNotIn("torch", sys.modules)

    def test_02_numpy_array_round_trip(self):
        result = wait(Generator.shared().call(actor_probes.array_result))
        np.testing.assert_array_equal(result, np.arange(12, dtype=np.float32).reshape(3, 4))
        self.assertEqual(result.dtype, np.float32)

    def test_03_dependency_exception_reaches_frontend_without_dependency_import(self):
        with self.assertRaisesRegex(RuntimeError, "HFValidationError"):
            wait(Generator.shared().call(actor_probes.dependency_error))
        self.assertNotIn("huggingface_hub", sys.modules)
        self.assertEqual(wait(Generator.shared().call(sum, [1, 2, 3])), 6)

    def test_04_cancel_stream_then_accept_next_request(self):
        future = Generator.shared().call(actor_probes.progress)
        deadline = time.monotonic() + 5
        while not future._responses and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(future._responses)
        future.cancel()
        wait(future)
        self.assertLess(len(future._responses), 100)
        self.assertEqual(wait(Generator.shared().call(sum, [10, 20])), 30)

    def test_05_crash_completes_future_and_next_request_recovers(self):
        previous_pid = Generator.shared().process.pid
        with self.assertRaisesRegex(RuntimeError, r"exited \(code 42\)"):
            wait(Generator.shared().call(actor_probes.crash), timeout=10)
        self.assertEqual(wait(Generator.shared().call(sum, [20, 22])), 42)
        self.assertNotEqual(Generator.shared().process.pid, previous_pid)

    def test_06_unreadable_response_cannot_corrupt_the_next_request(self):
        with self.assertRaisesRegex(RuntimeError, "value Blender could not read"):
            wait(Generator.shared().call(actor_probes.unsupported_dependency_value))
        self.assertEqual(wait(Generator.shared().call(sum, [40, 2])), 42)

    @classmethod
    def tearDownClass(cls):
        Generator.shared_close()


suite = unittest.defaultTestLoader.loadTestsFromTestCase(ActorIntegrationTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
addon_utils.disable("dream_textures", default_set=True)
if not result.wasSuccessful():
    raise SystemExit(1)
print("ACTOR_INTEGRATION_PASS", flush=True)
