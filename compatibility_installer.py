"""Private, transactional runtime installer for the Dream Textures compatibility build.

This module deliberately imports only the standard library. It is shared by the
Blender UI and an isolated subprocess, and can be tested without importing bpy.
Installed libraries never enter Blender's Python installation or a user's venv.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid

SCHEMA = 1
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_DLL_HANDLES = []
# Blender reloads add-on modules on disable/enable. Keep this cache for the
# interpreter lifetime, because already imported native libraries cannot unload.
_SESSION_RUNTIMES = getattr(sys, "_dream_textures_compat_session_runtimes", None)
if not isinstance(_SESSION_RUNTIMES, dict):
    _SESSION_RUNTIMES = {}
    sys._dream_textures_compat_session_runtimes = _SESSION_RUNTIMES


class InstallerError(RuntimeError):
    pass


class InstallCancelled(InstallerError):
    pass


def runtime_data_root(config_dir):
    return Path(config_dir).expanduser().resolve() / "dream_textures_compat"


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def atomic_json(path, value):
    """Replace a small state file on the same volume, leaving the old one on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def read_runtime_state(data_root):
    value = _read_json(Path(data_root) / "runtime.json")
    if type(value.get("schema")) is not int or value.get("schema") != SCHEMA:
        return {}
    active = value.get("active")
    previous = value.get("previous")
    if not isinstance(active, str) or not _RUNTIME_ID.fullmatch(active) or active.startswith("staging-"):
        return {}
    if previous is not None and (
        not isinstance(previous, str) or not _RUNTIME_ID.fullmatch(previous) or previous.startswith("staging-")
    ):
        return {}
    return {"schema": SCHEMA, "active": active, "previous": previous}


def validated_runtime_path(data_root, runtime_id, require_cuda=True):
    if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id) or runtime_id.startswith("staging-"):
        return None
    root = Path(data_root).resolve()
    runtime_folder = root / "runtimes"
    if runtime_folder.is_symlink():
        return None
    runtimes = runtime_folder.resolve()
    if runtimes.parent != root:
        return None
    candidate = runtimes / runtime_id
    try:
        resolved = candidate.resolve(strict=True)
        if resolved.parent != runtimes or candidate.is_symlink() or not resolved.is_dir():
            return None
    except (OSError, RuntimeError):
        return None
    if (resolved / "runtime-manifest.json").is_symlink():
        return None
    manifest = _read_json(resolved / "runtime-manifest.json")
    if (
        type(manifest.get("schema")) is not int
        or manifest.get("schema") != SCHEMA
        or manifest.get("validated") is not True
        or manifest.get("python") != [sys.version_info.major, sys.version_info.minor]
        or manifest.get("platform") != sys.platform
        or manifest.get("bits") != struct.calcsize("P") * 8
        or manifest.get("runtime_id") != runtime_id
    ):
        return None
    if require_cuda and (
        manifest.get("backend") != "cuda128"
        or manifest.get("cuda_available") is not True
    ):
        return None
    return resolved


def get_active_runtime(data_root, require_cuda=True):
    state = read_runtime_state(data_root)
    return validated_runtime_path(data_root, state.get("active"), require_cuda)


def session_runtime(data_root, require_cuda=True):
    """Cache selection once per Blender process; a completed install needs restart."""
    key = (str(Path(data_root).resolve()), bool(require_cuda))
    if key not in _SESSION_RUNTIMES:
        _SESSION_RUNTIMES[key] = get_active_runtime(data_root, require_cuda)
    return _SESSION_RUNTIMES[key]


def runtime_status(data_root, loaded_runtime=None, require_cuda=True):
    selected = get_active_runtime(data_root, require_cuda)
    loaded = Path(loaded_runtime).resolve() if loaded_runtime else None
    state = read_runtime_state(data_root)
    manifest = _read_json(selected / "runtime-manifest.json") if selected else {}
    previous = validated_runtime_path(data_root, state.get("previous"), require_cuda)
    return {
        "ready": selected is not None and loaded == selected,
        "restart_required": selected is not None and loaded != selected,
        "selected": str(selected) if selected else None,
        "loaded": str(loaded) if loaded else None,
        "previous": str(previous) if previous else None,
        "manifest": manifest,
    }


def sanitized_environment(source=None):
    source = os.environ if source is None else source
    env = {
        key: value for key, value in source.items()
        if not key.upper().startswith(("PYTHON", "PIP_"))
        and key.upper() not in {"VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"}
    }
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull
    return env


def blender_python_path(prefix=None, executable=None):
    """Resolve the embedded interpreter; sys.executable can be blender.exe in bpy."""
    prefix = Path(sys.prefix if prefix is None else prefix)
    executable = Path(sys.executable if executable is None else executable)
    candidates = [prefix / "bin" / "python.exe", prefix / "python.exe"]
    candidates += [prefix / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}", prefix / "bin" / "python3"]
    if executable.name.lower().startswith("python"):
        candidates.append(executable)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise InstallerError("Blender's bundled Python executable was not found. See the installation log for the expected Python folder: " + str(prefix))


def configure_runtime_dlls(runtime_dir):
    """Retain Windows DLL search directories, including Blender's ABI3 bridge."""
    if sys.platform != "win32":
        return
    runtime = Path(runtime_dir).resolve()
    directories = [runtime / "Library" / "bin", runtime / "torch" / "lib"]
    directories += [parent for parent in Path(sys.executable).resolve().parents if (parent / "python3.dll").is_file()]
    for directory in dict.fromkeys(directories):
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def _bundled_pip_wheel():
    import ensurepip
    bundled = Path(ensurepip.__file__).parent / "_bundled"
    wheels = sorted(bundled.glob("pip-*.whl"))
    if not wheels:
        raise InstallerError("This Blender Python does not include an ensurepip wheel. Reinstall Blender from its official Windows distribution; no global pip installation was attempted.")
    return wheels[-1]


_PIP_ENTRY = (
    "import runpy,sys; "
    "wheel=sys.argv.pop(1); sys.path.insert(0,wheel); "
    "sys.argv[0]='pip'; runpy.run_module('pip',run_name='__main__')"
)


def build_pip_command(python, pip_wheel, target, requirements, torch_requirements, wheel_dir, cpu_test_only=False):
    """One resolver, fixed indexes, binary wheels, and an explicit private target."""
    command = [
        str(python), "-I", "-S", "-c", _PIP_ENTRY, str(pip_wheel),
        "install", "--isolated", "--disable-pip-version-check", "--no-input",
        "--no-warn-script-location", "--no-cache-dir", "--ignore-installed",
        "--only-binary=:all:", "--target", str(target),
        "--index-url", PYPI_INDEX,
        "--extra-index-url", CPU_INDEX if cpu_test_only else CUDA_INDEX,
        "--find-links", str(wheel_dir),
        "-r", str(requirements), "-r", str(torch_requirements),
    ]
    return command


def _check_cancel(cancel_file):
    if cancel_file and Path(cancel_file).exists():
        raise InstallCancelled("Installation cancelled. The previous runtime is unchanged.")


def terminate_process_tree(process):
    """Stop only the child process tree owned by this installer."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            with contextlib.suppress(OSError):
                process.kill()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def run_subprocess(command, cancel_file=None):
    _check_cancel(cancel_file)
    options = {"env": sanitized_environment(), "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    process = subprocess.Popen(command, **options)
    try:
        while process.poll() is None:
            _check_cancel(cancel_file)
            time.sleep(0.2)
        if process.returncode:
            raise InstallerError(f"A runtime subprocess exited with code {process.returncode}. The preceding log lines contain the actual error.")
    except BaseException:
        terminate_process_tree(process)
        raise


@contextlib.contextmanager
def install_lock(data_root):
    """OS lock automatically releases after a crash; no stale PID deletion needed."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    with (data_root / ".install.lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise InstallerError("Another Dream Textures runtime installation is already running.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def activate_runtime(data_root, prepared_dir, manifest):
    """Commit a validated new runtime; never delete or rename the previous runtime."""
    root = Path(data_root).resolve()
    if (root / "runtimes").is_symlink():
        raise InstallerError("The private runtime directory must not be a symbolic link.")
    runtimes = (root / "runtimes").resolve()
    if runtimes.parent != root:
        raise InstallerError("The runtime directory is outside this installation.")
    prepared = Path(prepared_dir).resolve()
    if prepared.parent != runtimes or not prepared.name.startswith("staging-"):
        raise InstallerError("Refusing to activate a runtime outside this installation's staging directory.")
    if manifest.get("validated") is not True or manifest.get("schema") != SCHEMA:
        raise InstallerError("The runtime did not pass validation.")
    runtime_id = manifest.get("runtime_id")
    if not isinstance(runtime_id, str) or not _RUNTIME_ID.fullmatch(runtime_id) or runtime_id.startswith("staging-"):
        raise InstallerError("The runtime identifier is invalid.")
    final = runtimes / runtime_id
    if final.exists():
        raise InstallerError("A runtime with this identifier already exists.")
    atomic_json(prepared / "runtime-manifest.json", manifest)
    state = read_runtime_state(root)
    previous = state.get("active")
    prepared.rename(final)
    try:
        if validated_runtime_path(root, runtime_id, require_cuda=manifest.get("backend") != "cpu-test") is None:
            raise InstallerError("The prepared runtime manifest is not valid on this Python installation.")
        atomic_json(root / "runtime.json", {"schema": SCHEMA, "active": runtime_id, "previous": previous})
    except BaseException:
        # This newly created directory has never been selected by a running actor.
        shutil.rmtree(final, ignore_errors=True)
        raise
    return final


def restore_previous_runtime(data_root):
    with install_lock(data_root):
        state = read_runtime_state(data_root)
        previous = validated_runtime_path(data_root, state.get("previous"))
        if previous is None:
            raise InstallerError("No previously validated CUDA runtime is available.")
        atomic_json(Path(data_root) / "runtime.json", {"schema": SCHEMA, "active": previous.name, "previous": state.get("active")})
        return previous


def validate_runtime(runtime_dir, cpu_test_only=False):
    """Imports and tiny CUDA kernel checks only; does not download any AI model."""
    runtime_dir = Path(runtime_dir).resolve()
    sys.path.insert(0, str(runtime_dir))
    configure_runtime_dlls(runtime_dir)
    import importlib
    import importlib.metadata
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "PIL", "huggingface_hub", "cv2", "PyOpenColorIO", "controlnet_aux", "imwatermark"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(runtime_dir):
            raise InstallerError(f"{name} was imported from outside the private runtime: {origin}")
        versions[name] = getattr(module, "__version__", "installed")
    import torch
    from diffusers import (
        ControlNetModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,
        StableDiffusionUpscalePipeline,
    )
    from transformers import CLIPTextModel, CLIPTokenizer
    result = {
        "schema": SCHEMA,
        "validated": True,
        "python": [sys.version_info.major, sys.version_info.minor],
        "platform": sys.platform,
        "bits": struct.calcsize("P") * 8,
        "backend": "cpu-test" if cpu_test_only else "cuda128",
        "packages": versions,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Dependency imports and small device operations; model generation still requires a Blender smoke test.",
    }
    if not cpu_test_only:
        if not result["cuda_available"]:
            raise InstallerError("PyTorch could not initialize CUDA. Check the NVIDIA driver and the preceding CUDA error. The existing runtime was not replaced.")
        if not str(torch.version.cuda).startswith("12.8"):
            raise InstallerError(f"Expected CUDA 12.8 Torch libraries, received CUDA {torch.version.cuda}.")
        capability = torch.cuda.get_device_capability(0)
        architectures = torch.cuda.get_arch_list()
        if capability >= (12, 0) and "sm_120" not in architectures:
            raise InstallerError("The installed PyTorch wheel does not contain RTX 50-series (sm_120) support.")
        with torch.inference_mode():
            values = torch.ones((16, 16), device="cuda:0", dtype=torch.float16)
            product = values @ values
            sample = torch.ones((1, 4, 16, 16), device="cuda:0", dtype=torch.float16)
            kernel = torch.ones((4, 4, 3, 3), device="cuda:0", dtype=torch.float16)
            convolved = torch.nn.functional.conv2d(sample, kernel)
            torch.cuda.synchronize()
            if not bool(torch.isfinite(product).all()) or not bool(torch.isfinite(convolved).all()):
                raise InstallerError("The GPU validation produced a non-finite result.")
        result.update({"device": torch.cuda.get_device_name(0), "capability": list(capability), "architectures": architectures, "vram_bytes": torch.cuda.get_device_properties(0).total_memory})
    else:
        if not torch.equal(torch.ones((2, 2)) @ torch.ones((2, 2)), torch.full((2, 2), 2.0)):
            raise InstallerError("CPU test operation failed.")
        result["device"] = "CPU (development test only)"
    return result


def install_runtime(data_root, requirements, torch_requirements, wheel_dir, status_file=None, cancel_file=None, cpu_test_only=False):
    root = Path(data_root).resolve()
    if not cpu_test_only and (sys.platform != "win32" or sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8):
        raise InstallerError("This compatibility package targets 64-bit Blender Python 3.13 on Windows. The CPU lane is reserved for development tests.")
    for path in (requirements, torch_requirements):
        if not Path(path).is_file():
            raise InstallerError(f"The compatibility package is incomplete: requirements file missing: {path}")
    if not Path(wheel_dir).is_dir():
        raise InstallerError("The compatibility package is incomplete: its reviewed wheel directory is missing.")

    def status(phase, message, **extra):
        print(message, flush=True)
        if status_file:
            atomic_json(status_file, {"phase": phase, "message": message, "pid": os.getpid(), **extra})

    with install_lock(root):
        _check_cancel(cancel_file)
        runtime_id = datetime.datetime.now(datetime.timezone.utc).strftime("runtime-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:10]
        if (root / "runtimes").is_symlink():
            raise InstallerError("The private runtime directory must not be a symbolic link.")
        (root / "runtimes").mkdir(exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix="staging-", dir=root / "runtimes"))
        try:
            status("installing", "Downloading and installing the Dream Textures GPU libraries. Blender can remain open.")
            command = build_pip_command(sys.executable, _bundled_pip_wheel(), staged, requirements, torch_requirements, wheel_dir, cpu_test_only)
            run_subprocess(command, cancel_file)
            _check_cancel(cancel_file)
            status("validating", "Checking dependency imports and the graphics device.")
            result_file = root / "jobs" / (runtime_id + "-validation.json")
            command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "validate", "--runtime", str(staged), "--result", str(result_file)]
            if cpu_test_only:
                command.append("--cpu-test-only")
            run_subprocess(command, cancel_file)
            manifest = _read_json(result_file)
            manifest["runtime_id"] = runtime_id
            _check_cancel(cancel_file)
            final = activate_runtime(root, staged, manifest)
            status("complete", "Runtime installation passed its checks. Restart Blender before generating an image.", runtime=str(final), validation=manifest)
            return final
        except BaseException:
            # Only this attempt's staging directory is removed. Previous installations remain intact.
            shutil.rmtree(staged, ignore_errors=True)
            raise


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    install = commands.add_parser("install")
    install.add_argument("--root", required=True)
    install.add_argument("--requirements", required=True)
    install.add_argument("--torch-requirements", required=True)
    install.add_argument("--wheel-dir", required=True)
    install.add_argument("--status")
    install.add_argument("--cancel")
    install.add_argument("--cpu-test-only", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--runtime", required=True)
    validate.add_argument("--result", required=True)
    validate.add_argument("--status")
    validate.add_argument("--cpu-test-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            install_runtime(args.root, args.requirements, args.torch_requirements, args.wheel_dir, args.status, args.cancel, args.cpu_test_only)
        else:
            result = validate_runtime(args.runtime, args.cpu_test_only)
            atomic_json(args.result, result)
            message = "Runtime check passed: " + str(result.get("device", "device available"))
            print(message, flush=True)
            if args.status:
                atomic_json(args.status, {"phase": "complete", "message": message, "validation": result})
        return 0
    except BaseException as error:
        import traceback
        traceback.print_exc()
        phase = "cancelled" if isinstance(error, InstallCancelled) else "failed"
        if getattr(args, "status", None):
            atomic_json(args.status, {"phase": phase, "message": str(error) or type(error).__name__})
        return 2 if phase == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
