"""Meaningful invariants of the private runtime install/activation boundary."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ADDON = Path(__file__).resolve().parents[1] / "package" / "dream_textures"
if not ADDON.is_dir():
    ADDON = Path(__file__).resolve().parents[2]
MODULE = ADDON / "compatibility_installer.py"
spec = importlib.util.spec_from_file_location("dream_compat_installer", MODULE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="Dream Textures test ")
        self.root = Path(self.temporary.name) / "config" / "dream_textures_compat"
        (self.root / "runtimes").mkdir(parents=True)
        installer._SESSION_RUNTIMES.clear()

    def tearDown(self):
        installer._SESSION_RUNTIMES.clear()
        self.temporary.cleanup()

    def manifest(self, name, **updates):
        result = {
            "schema": 1, "validated": True, "runtime_id": name,
            "python": [sys.version_info.major, sys.version_info.minor],
            "platform": sys.platform, "bits": struct.calcsize("P") * 8,
            "backend": "cuda128", "cuda_available": True,
        }
        result.update(updates)
        return result

    def staged(self):
        path = Path(tempfile.mkdtemp(prefix="staging-", dir=self.root / "runtimes"))
        (path / "user-package.txt").write_text("preserve me", encoding="utf-8")
        return path

    def activated(self, name, **updates):
        return installer.activate_runtime(self.root, self.staged(), self.manifest(name, **updates))

    def input_files(self):
        requirements = self.root / "requirements file.txt"
        requirements.write_text("example==1.0\n", encoding="utf-8")
        torch_requirements = self.root / "torch requirements.txt"
        torch_requirements.write_text("torch==2.9.1+cu128\n", encoding="utf-8")
        wheels = self.root / "reviewed wheels"
        wheels.mkdir(exist_ok=True)
        return requirements, torch_requirements, wheels

    def test_install_selection_is_frozen_until_next_blender_process(self):
        self.assertIsNone(installer.session_runtime(self.root))
        installed = self.activated("runtime-one")
        self.assertEqual(installer.get_active_runtime(self.root), installed)
        self.assertIsNone(installer.session_runtime(self.root))
        self.assertTrue(installer.runtime_status(self.root, None)["restart_required"])
        installer._SESSION_RUNTIMES.clear()  # Simulate restarting Blender.
        self.assertEqual(installer.session_runtime(self.root), installed)
        self.assertTrue(installer.runtime_status(self.root, installed)["ready"])

    def test_addon_module_reload_does_not_hot_swap_native_libraries(self):
        self.assertIsNone(installer.session_runtime(self.root))
        self.activated("runtime-one")
        reloaded_spec = importlib.util.spec_from_file_location("dream_compat_installer_reloaded", MODULE)
        reloaded = importlib.util.module_from_spec(reloaded_spec)
        reloaded_spec.loader.exec_module(reloaded)
        self.assertIsNone(reloaded.session_runtime(self.root))
        self.assertTrue(reloaded.runtime_status(self.root, None)["restart_required"])

    def test_replacement_and_rollback_preserve_old_packages(self):
        original = self.activated("runtime-one")
        loaded = installer.session_runtime(self.root)
        replacement = self.activated("runtime-two")
        self.assertEqual(installer.read_runtime_state(self.root)["previous"], original.name)
        self.assertEqual(installer.session_runtime(self.root), loaded)
        self.assertTrue(installer.runtime_status(self.root, loaded)["restart_required"])
        self.assertEqual(installer.restore_previous_runtime(self.root), original)
        self.assertEqual(installer.get_active_runtime(self.root), original)
        self.assertTrue((original / "user-package.txt").is_file())
        self.assertTrue((replacement / "user-package.txt").is_file())

    def test_failed_pointer_commit_preserves_previous_selection(self):
        old = self.activated("runtime-one")
        staged = self.staged()
        replace = os.replace

        def fail_pointer(source, target):
            if Path(target) == self.root / "runtime.json":
                raise OSError("simulated disk failure")
            return replace(source, target)

        with mock.patch.object(installer.os, "replace", side_effect=fail_pointer):
            with self.assertRaises(OSError):
                installer.activate_runtime(self.root, staged, self.manifest("runtime-two"))
        self.assertEqual(installer.get_active_runtime(self.root), old)
        self.assertFalse((self.root / "runtimes" / "runtime-two").exists())
        self.assertTrue((old / "user-package.txt").exists())

    def test_cpu_and_wrong_abi_cannot_satisfy_cuda_readiness(self):
        cpu = self.activated("cpu-test", backend="cpu-test", cuda_available=False)
        self.assertIsNone(installer.get_active_runtime(self.root))
        self.assertEqual(installer.get_active_runtime(self.root, require_cuda=False), cpu)
        manifest = self.manifest("cpu-test", python=[3, 11] if sys.version_info[:2] != (3, 11) else [3, 13])
        installer.atomic_json(cpu / "runtime-manifest.json", manifest)
        self.assertIsNone(installer.get_active_runtime(self.root, require_cuda=False))

    def test_unvalidated_runtime_cannot_replace_working_runtime(self):
        old = self.activated("runtime-one")
        with self.assertRaises(installer.InstallerError):
            installer.activate_runtime(self.root, self.staged(), self.manifest("runtime-two", validated=False))
        self.assertEqual(installer.get_active_runtime(self.root), old)

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        for name in ("../outside", "/tmp/outside", "C:\\outside", "runtime/child", "..", "staging-uncommitted"):
            installer.atomic_json(self.root / "runtime.json", {"schema": 1, "active": name, "previous": None})
            self.assertEqual(installer.read_runtime_state(self.root), {})
            self.assertIsNone(installer.get_active_runtime(self.root))
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        installer.atomic_json(outside / "runtime-manifest.json", self.manifest("escape"))
        try:
            (self.root / "runtimes" / "escape").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("OS does not permit test symlinks")
        installer.atomic_json(self.root / "runtime.json", {"schema": 1, "active": "escape", "previous": None})
        self.assertIsNone(installer.get_active_runtime(self.root))

    def test_resolver_failure_removes_only_this_attempt(self):
        old = self.activated("runtime-one")
        files = self.input_files()
        with mock.patch.object(installer, "_bundled_pip_wheel", return_value=self.root / "pip.whl"), mock.patch.object(installer, "run_subprocess", side_effect=installer.InstallerError("resolver failed")):
            with self.assertRaisesRegex(installer.InstallerError, "resolver failed"):
                installer.install_runtime(self.root, *files, cpu_test_only=True)
        self.assertEqual(installer.get_active_runtime(self.root), old)
        self.assertEqual(list((self.root / "runtimes").glob("staging-*")), [])

    def test_cancel_between_validation_and_activation_keeps_previous(self):
        old = self.activated("runtime-one")
        files = self.input_files()
        cancel = self.root / "cancel"

        def fake_command(command, cancel_file=None):
            if "validate" in command:
                result = Path(command[command.index("--result") + 1])
                installer.atomic_json(result, self.manifest("temporary", backend="cpu-test", cuda_available=False))
                cancel.write_text("cancel", encoding="utf-8")

        with mock.patch.object(installer, "_bundled_pip_wheel", return_value=self.root / "pip.whl"), mock.patch.object(installer, "run_subprocess", side_effect=fake_command):
            with self.assertRaises(installer.InstallCancelled):
                installer.install_runtime(self.root, *files, cancel_file=cancel, cpu_test_only=True)
        self.assertEqual(installer.get_active_runtime(self.root), old)
        self.assertEqual(list((self.root / "runtimes").glob("staging-*")), [])

    def test_pip_command_has_private_target_fixed_indexes_and_no_shell(self):
        requirements, torch_requirements, wheels = self.input_files()
        target = self.root / "runtimes" / "staging-with spaces"
        command = installer.build_pip_command("C:/Blender 5.2/python.exe", "C:/Blender 5.2/pip.whl", target, requirements, torch_requirements, wheels)
        self.assertEqual(command[command.index("--target") + 1], str(target))
        self.assertEqual(command[command.index("--index-url") + 1], "https://pypi.org/simple")
        self.assertEqual(command[command.index("--extra-index-url") + 1], "https://download.pytorch.org/whl/cu128")
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("--ignore-installed", command)
        self.assertIn("-I", command)
        self.assertIn("-S", command)
        self.assertNotIn("--user", command)
        self.assertNotIn("uninstall", command)
        self.assertEqual(command.count("-r"), 2)

    def test_hermes_environment_cannot_redirect_pip_or_python(self):
        environment = installer.sanitized_environment({
            "PYTHONPATH": "C:/Hermes/site-packages", "PYTHONHOME": "C:/Hermes",
            "VIRTUAL_ENV": "C:/Hermes", "PIP_TARGET": "C:/Hermes",
            "PIP_INDEX_URL": "https://unexpected.example/simple", "PIP_USER": "true",
            "SystemRoot": "C:/Windows", "PATH": "C:/Windows/System32",
        })
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("VIRTUAL_ENV", environment)
        self.assertNotIn("PIP_TARGET", environment)
        self.assertNotIn("PIP_INDEX_URL", environment)
        self.assertNotIn("PIP_USER", environment)
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(environment["SystemRoot"], "C:/Windows")

    def test_embedded_python_selection_never_uses_blender_executable(self):
        prefix = Path(self.temporary.name) / "Blender 5.2" / "python"
        (prefix / "bin").mkdir(parents=True)
        python = prefix / "bin" / "python.exe"
        python.touch()
        self.assertEqual(installer.blender_python_path(prefix, prefix.parent / "blender.exe"), python.resolve())

    def test_windows_abi3_and_torch_dll_directories_keep_live_handles(self):
        blender = Path(self.temporary.name) / "Blender 5.2"
        binary = blender / "5.2" / "python" / "bin" / "python.exe"
        binary.parent.mkdir(parents=True)
        binary.touch()
        (blender / "python3.dll").touch()
        runtime = self.staged()
        (runtime / "torch" / "lib").mkdir(parents=True)
        handle = object()
        original_count = len(installer._DLL_HANDLES)
        try:
            with mock.patch.object(installer.sys, "platform", "win32"), mock.patch.object(installer.sys, "executable", str(binary)), mock.patch.object(installer.os, "add_dll_directory", return_value=handle, create=True) as add_directory:
                installer.configure_runtime_dlls(runtime)
            added = [call.args[0] for call in add_directory.call_args_list]
            self.assertIn(str(blender), added)
            self.assertIn(str(runtime / "torch" / "lib"), added)
            self.assertEqual(len(installer._DLL_HANDLES), original_count + 2)
            self.assertIs(installer._DLL_HANDLES[-1], handle)
        finally:
            del installer._DLL_HANDLES[original_count:]

    def test_pip_can_run_directly_from_bundled_wheel_without_installing_it(self):
        wheel = installer._bundled_pip_wheel()
        result = subprocess.run([sys.executable, "-I", "-S", "-c", installer._PIP_ENTRY, str(wheel), "--version"], env=installer.sanitized_environment(), text=True, capture_output=True, check=True)
        self.assertIn("pip ", result.stdout)
        self.assertIn(str(wheel), result.stdout)

    def test_runtime_lock_releases_after_error_and_rejects_overlap(self):
        with installer.install_lock(self.root):
            with self.assertRaises(installer.InstallerError):
                with installer.install_lock(self.root):
                    self.fail("Overlapping install lock accepted")
        with installer.install_lock(self.root):
            pass

    def test_cancel_stops_owned_subprocess_promptly(self):
        cancel = self.root / "cancel-subprocess"
        timer = threading.Timer(0.2, lambda: cancel.write_text("cancel", encoding="utf-8"))
        timer.start()
        start = time.monotonic()
        try:
            with self.assertRaises(installer.InstallCancelled):
                installer.run_subprocess([sys.executable, "-c", "import time; time.sleep(30)"], cancel)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - start, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
