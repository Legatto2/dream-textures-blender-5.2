"""Non-blocking Blender controls for the private compatibility runtime."""

import contextlib
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import bpy

from ..absolute_path import absolute_path
from .. import compatibility_installer as installer
from ..generator_process import Generator

_JOB = None
_LAST_STATUS = {}
_LATEST_LOG = None


def data_root():
    return installer.runtime_data_root(bpy.utils.user_resource('CONFIG'))


def _require_cuda_runtime():
    return sys.platform == "win32" or os.environ.get("DREAM_TEXTURES_TEST_ALLOW_CPU") != "1"


def session_runtime():
    return installer.session_runtime(data_root(), require_cuda=_require_cuda_runtime())


def current_runtime_status():
    return installer.runtime_status(data_root(), session_runtime(), require_cuda=_require_cuda_runtime())


def current_job_status():
    if _JOB is not None:
        value = installer._read_json(_JOB["status"])
        return value or {"phase": "starting", "message": "Starting the runtime check." if _JOB["mode"] == "validate" else "Starting the private dependency installer."}
    return _LAST_STATUS


def job_running():
    return _JOB is not None


def _redraw():
    manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(manager, "windows", ()):
        for area in window.screen.areas:
            area.tag_redraw()


def _poll_job():
    global _JOB, _LAST_STATUS
    if _JOB is None:
        return None
    _redraw()
    process = _JOB["process"]
    if process.poll() is None:
        return 0.4
    _LAST_STATUS = installer._read_json(_JOB["status"])
    if not _LAST_STATUS or _LAST_STATUS.get("phase") not in {"complete", "failed", "cancelled"}:
        _LAST_STATUS = {
            "phase": "failed",
            "message": f"The installer stopped with exit code {process.returncode}. Open the installation log for details.",
        }
    _JOB["log_handle"].close()
    _JOB = None
    _redraw()
    return None


def _start_job(mode):
    global _JOB, _LAST_STATUS, _LATEST_LOG
    if job_running():
        raise installer.InstallerError("A Dream Textures runtime operation is already running.")
    if mode == "install" and not getattr(bpy.app, "online_access", True):
        raise installer.InstallerError("Enable Allow Online Access in Blender Preferences, then install the dependencies.")
    if mode == "install" and (sys.platform != "win32" or sys.version_info[:2] != (3, 13)):
        raise installer.InstallerError("This build's installer requires Blender's Windows Python 3.13.")
    instance = getattr(Generator, "_shared_instance", None)
    if instance is not None and instance.is_alive() and not instance.can_use():
        raise installer.InstallerError("Finish or cancel the current Dream Textures generation before changing its runtime.")
    Generator.shared_close()
    loaded = session_runtime()
    if mode == "validate" and not current_runtime_status()["ready"]:
        raise installer.InstallerError("Install the dependencies and restart Blender before checking the runtime.")
    python = installer.blender_python_path()
    root = data_root()
    job_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:8]
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    log_path = root / "logs" / f"{mode}-{job_id}.log"
    status_path = root / "jobs" / f"{mode}-{job_id}.json"
    cancel_path = root / "jobs" / f"{mode}-{job_id}.cancel"
    result_path = root / "jobs" / f"{mode}-{job_id}-result.json"
    command = [str(python), "-I", "-S", absolute_path("compatibility_installer.py"), mode]
    if mode == "install":
        command += [
            "--root", str(root),
            "--requirements", absolute_path("requirements/compat-runtime.txt"),
            "--torch-requirements", absolute_path("requirements/compat-torch-cuda128.txt"),
            "--wheel-dir", absolute_path("wheels"),
            "--cancel", str(cancel_path),
        ]
    else:
        command += ["--runtime", str(loaded), "--result", str(result_path)]
        if not _require_cuda_runtime():
            command.append("--cpu-test-only")
    command += ["--status", str(status_path)]
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    log_handle.write("Dream Textures compatibility runtime\n")
    log_handle.write(f"Blender: {bpy.app.version_string}\nPython: {python}\nRuntime data: {root}\nOperation: {mode}\n\n")
    log_handle.flush()
    options = {
        "stdin": subprocess.DEVNULL, "stdout": log_handle, "stderr": subprocess.STDOUT,
        "env": installer.sanitized_environment(),
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **options)
    except BaseException:
        log_handle.close()
        raise
    _LATEST_LOG = log_path
    _LAST_STATUS = {}
    _JOB = {"process": process, "log_handle": log_handle, "status": status_path, "cancel": cancel_path, "mode": mode}
    bpy.app.timers.register(_poll_job, first_interval=0.2)
    _redraw()


def shutdown_installer():
    """Called when disabling/reloading the add-on; no abandoned pip process."""
    global _JOB
    if _JOB is not None:
        _JOB["cancel"].write_text("cancel\n", encoding="utf-8")
        process = _JOB["process"]
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            installer.terminate_process_tree(process)
        _JOB["log_handle"].close()
        _JOB = None
    with contextlib.suppress(ValueError, RuntimeError):
        if bpy.app.timers.is_registered(_poll_job):
            bpy.app.timers.unregister(_poll_job)


class InstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.install_dependencies"
    bl_label = "Install Compatible GPU Dependencies"
    bl_description = "Install Dream Textures' private CUDA 12.8 runtime and check the GPU; restart Blender after completion"
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return not job_running()

    def execute(self, context):
        try:
            _start_job("install")
        except (OSError, installer.InstallerError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, "Dependency installation started. Progress and the log are available in Dream Textures preferences.")
        return {'FINISHED'}


class CancelDependencyInstall(bpy.types.Operator):
    bl_idname = "dream_textures.cancel_dependency_install"
    bl_label = "Cancel"
    bl_description = "Cancel this runtime operation and preserve the previous installation"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return job_running()

    def execute(self, context):
        if _JOB:
            if _JOB["process"].poll() is not None:
                _poll_job()
                return {'FINISHED'}
            _JOB["cancel"].write_text("cancel\n", encoding="utf-8")
            if _JOB["mode"] == "validate":
                installer.terminate_process_tree(_JOB["process"])
                installer.atomic_json(_JOB["status"], {"phase": "cancelled", "message": "Runtime check cancelled."})
            self.report({'INFO'}, "Cancellation requested.")
        return {'FINISHED'}


class VerifyRuntime(bpy.types.Operator):
    bl_idname = "dream_textures.verify_runtime"
    bl_label = "Check GPU Runtime"
    bl_description = "Check package imports and small CUDA operations without downloading a model"
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return not job_running() and current_runtime_status()["ready"]

    def execute(self, context):
        try:
            _start_job("validate")
        except (OSError, installer.InstallerError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class OpenRuntimeLog(bpy.types.Operator):
    bl_idname = "dream_textures.open_runtime_log"
    bl_label = "Open Installation Log"
    bl_description = "Open the latest Dream Textures dependency installation or runtime check log"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        log = _LATEST_LOG
        if log is None or not log.is_file():
            logs = list((data_root() / "logs").glob("*.log"))
            log = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
        if log is None:
            self.report({'INFO'}, "No runtime installation log exists yet.")
            return {'CANCELLED'}
        if os.name == "nt":
            os.startfile(str(log))
        else:
            bpy.ops.wm.path_open(filepath=str(log))
        return {'FINISHED'}


class RestorePreviousRuntime(bpy.types.Operator):
    bl_idname = "dream_textures.restore_previous_runtime"
    bl_label = "Restore Previous Runtime"
    bl_description = "Select the previously validated private runtime; restart Blender to use it"
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return not job_running() and current_runtime_status()["previous"] is not None

    def execute(self, context):
        try:
            installer.restore_previous_runtime(data_root())
        except (OSError, installer.InstallerError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, "Previous runtime selected. Restart Blender to use it.")
        _redraw()
        return {'FINISHED'}


class UninstallDependencies(bpy.types.Operator):
    """Keep the historical class name for registration; never touch global packages."""
    bl_idname = "stable_diffusion.uninstall_dependencies"
    bl_label = "Clean Incomplete Installations"
    bl_description = "Remove abandoned Dream Textures staging folders; retain every completed runtime"
    bl_options = {'REGISTER', 'INTERNAL'}

    conflicts: bpy.props.StringProperty(options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return not job_running()

    def execute(self, context):
        try:
            root = data_root()
            with installer.install_lock(root):
                runtimes = root / "runtimes"
                if runtimes.is_symlink():
                    raise installer.InstallerError("The private runtime directory is a symbolic link; cleanup was skipped.")
                for path in runtimes.glob("staging-*"):
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
        except (OSError, installer.InstallerError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self.report({'INFO'}, "Incomplete installations removed. Completed runtimes and models were retained.")
        return {'FINISHED'}
