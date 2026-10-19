"""Dream Textures background worker with an isolated, validated Python runtime."""

from multiprocessing import Queue, Lock, current_process, get_context, spawn
import multiprocessing.synchronize
from collections.abc import Generator
from pathlib import Path
from queue import Empty, Full
import enum
import traceback
import threading
from typing import Type, TypeVar
import site
import sys
import os
from .future import Future

_SPAWN_LOCK = threading.Lock()
_RUNTIME_ERROR = (
    "Dream Textures' compatibility runtime is not ready. Open Edit > Preferences > "
    "Add-ons > Dream Textures, install the GPU dependencies, then restart Blender."
)


def _require_cuda():
    # Explicit development-only lane; the Windows UI always installs CUDA.
    return sys.platform == 'win32' or os.environ.get("DREAM_TEXTURES_TEST_ALLOW_CPU") != "1"


def _frontend_runtime():
    import bpy
    from ..compatibility_installer import runtime_data_root, session_runtime
    root = runtime_data_root(bpy.utils.user_resource('CONFIG'))
    return root, session_runtime(root, require_cuda=_require_cuda())


def _load_dependencies():
    """Load the validated runtime selected by this Blender session."""
    from ..compatibility_installer import validated_runtime_path, configure_runtime_dlls
    root = os.environ.get("DREAM_TEXTURES_COMPAT_ROOT")
    runtime_id = os.environ.get("DREAM_TEXTURES_COMPAT_RUNTIME_ID")
    runtime = validated_runtime_path(root, runtime_id, require_cuda=_require_cuda()) if root else None
    if runtime is None:
        raise RuntimeError(_RUNTIME_ERROR)
    runtime_path = str(runtime)
    # multiprocessing copies Blender's sys.path, including paths injected by
    # unrelated add-ons. Keep the bundled standard library and this add-on's
    # import parent; obtain every third-party generation library from runtime.
    python_root = Path(sys.base_prefix).resolve()
    addon_parent = Path(__file__).resolve().parents[2]
    retained = []
    for path in sys.path:
        if not path:
            continue
        resolved = Path(path).resolve()
        is_stdlib = resolved.is_relative_to(python_root) and not {
            'site-packages', 'dist-packages'
        }.intersection(resolved.parts)
        if is_stdlib or resolved == addon_parent:
            retained.append(path)
    sys.path[:] = [runtime_path, *dict.fromkeys(retained)]
    site.addsitedir(runtime_path)
    configure_runtime_dlls(runtime)


main_thread_rendering = False
is_actor_process = current_process().name == "__actor__"
if is_actor_process:
    _load_dependencies()
elif {"-b", "-f", "-a", "--background"}.intersection(sys.argv):
    main_thread_rendering = True
    import bpy

    def main_thread_rendering_finished():
        global main_thread_rendering
        main_thread_rendering = False

    bpy.app.timers.register(main_thread_rendering_finished, persistent=True)


class ActorContext(enum.IntEnum):
    """The Blender frontend and its separate Python backend."""
    FRONTEND = 0
    BACKEND = 1


class Message:
    """A method call whose arguments must be picklable."""

    def __init__(self, method_name, args, kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    CANCEL = "__cancel__"
    END = "__end__"


def _start_backend(cls, message_queue, response_queue):
    cls(
        ActorContext.BACKEND,
        message_queue=message_queue,
        response_queue=response_queue,
    ).start()


class TracedError(BaseException):
    def __init__(self, base: BaseException, trace: str):
        self.base = base
        self.trace = trace


T = TypeVar('T', bound='Actor')


class Actor:
    """Dispatch specialized actors' public methods to a background process."""

    _message_queue: Queue
    _response_queue: Queue
    _lock: multiprocessing.synchronize.Lock
    _shared_instance = None
    _protected_methods = {
        "start", "close", "is_alive", "can_use", "runtime_available",
        "shared", "shared_close",
    }

    def __init__(self, context: ActorContext, message_queue: Queue = None, response_queue: Queue = None):
        self.context = context
        self._closed = False
        self._message_queue = message_queue if message_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._response_queue = response_queue if response_queue is not None else get_context('spawn').Queue(maxsize=1)
        self._setup()
        self.__class__._shared_instance = self

    def _setup(self):
        if self.context == ActorContext.FRONTEND:
            self._lock = Lock()
            for name in filter(lambda name: callable(getattr(self, name)) and not name.startswith("_") and name not in self._protected_methods, dir(self)):
                setattr(self, name, self._send(name))

    @classmethod
    def shared(cls: Type[T]) -> T:
        if cls._shared_instance is not None:
            if cls._shared_instance.is_alive():
                return cls._shared_instance
            cls.shared_close()
        try:
            return cls(ActorContext.FRONTEND).start()
        except BaseException:
            cls.shared_close()
            raise

    @staticmethod
    def runtime_available():
        if is_actor_process:
            return True
        return _frontend_runtime()[1] is not None

    def start(self: T) -> T:
        if self.context == ActorContext.FRONTEND:
            from ..compatibility_installer import blender_python_path
            root, runtime = _frontend_runtime()
            if runtime is None:
                raise RuntimeError(_RUNTIME_ERROR)
            self.process = get_context('spawn').Process(
                target=_start_backend,
                args=(self.__class__, self._message_queue, self._response_queue),
                name="__actor__", daemon=True,
            )
            main_module = sys.modules["__main__"]
            # Use Blender's own Python and avoid re-executing its startup/text
            # script in the worker, where bpy is intentionally unavailable.
            with _SPAWN_LOCK:
                missing = object()
                main_file = getattr(main_module, "__file__", missing)
                main_spec = getattr(main_module, "__spec__", missing)
                previous_executable = spawn.get_executable()
                changes = {
                    "DREAM_TEXTURES_COMPAT_ROOT": str(root),
                    "DREAM_TEXTURES_COMPAT_RUNTIME_ID": runtime.name,
                    "PYTHONNOUSERSITE": "1",
                }
                removed = {key for key in os.environ if key.upper().startswith('PYTHON')
                           or key.upper() in {'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV'}}
                previous_env = {key: os.environ.get(key) for key in removed | changes.keys()}
                try:
                    main_module.__file__ = None
                    main_module.__spec__ = None
                    get_context('spawn').set_executable(str(blender_python_path()))
                    for key in removed:
                        os.environ.pop(key, None)
                    os.environ.update(changes)
                    self.process.start()
                finally:
                    get_context('spawn').set_executable(previous_executable)
                    for name, value in (("__file__", main_file), ("__spec__", main_spec)):
                        if value is missing:
                            main_module.__dict__.pop(name, None)
                        else:
                            setattr(main_module, name, value)
                    for key, value in previous_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            self._backend_loop()
        return self

    def close(self):
        if self.context == ActorContext.FRONTEND:
            if self._closed:
                return
            self._closed = True
            process = getattr(self, 'process', None)
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
            for name in ('_message_queue', '_response_queue'):
                queue = getattr(self, name, None)
                if queue is not None:
                    queue.cancel_join_thread()
                    queue.close()

    @classmethod
    def shared_close(cls: Type[T]):
        if cls._shared_instance is not None:
            cls._shared_instance.close()
            cls._shared_instance = None

    def is_alive(self):
        if self.context == ActorContext.BACKEND:
            return True
        process = getattr(self, 'process', None)
        return not self._closed and process is not None and process.is_alive()

    def can_use(self):
        if not self.is_alive():
            return False
        if result := self._lock.acquire(block=False):
            self._lock.release()
        return result

    def _backend_loop(self):
        while True:
            message = self._message_queue.get()
            # A cancellation arriving after the final result belongs to the
            # previous operation, not the next request.
            if isinstance(message, Message):
                self._receive(message)

    def _receive(self, message: Message):
        try:
            response = getattr(self, message.method_name)(*message.args, **message.kwargs)
            if isinstance(response, Generator):
                for res in response:
                    try:
                        if self._message_queue.get_nowait() == Message.CANCEL:
                            break
                    except Empty:
                        pass
                    if isinstance(res, Future):
                        def check_cancelled():
                            try:
                                return self._message_queue.get_nowait() == Message.CANCEL
                            except Empty:
                                return False
                        res.check_cancelled = check_cancelled
                        res.add_response_callback(lambda _, value: self._response_queue.put(value))
                        res.add_exception_callback(lambda _, error: self._response_queue.put(RuntimeError(repr(error))))
                    else:
                        self._response_queue.put(res)
            else:
                self._response_queue.put(response)
        except Exception as error:
            trace = traceback.format_exc()
            # Blender must not import a private dependency just to unpickle its
            # error. Retain the original exception name and complete traceback.
            if type(error).__module__ != 'builtins':
                error = RuntimeError(f"{type(error).__module__}.{type(error).__qualname__}: {error}")
            self._response_queue.put(TracedError(error, trace))
        self._response_queue.put(Message.END)

    def _send(self, name):
        def send(*args, _block=False, **kwargs):
            if main_thread_rendering:
                _block = True
            future = Future()

            def send_thread():
                with self._lock:
                    try:
                        if not self.is_alive():
                            raise RuntimeError("Dream Textures' background process is not running. Retry the operation or restart Blender.")
                        self._message_queue.put(Message(name, args, kwargs), timeout=2)
                        cancel_sent = False
                        while not future.done:
                            if future.cancelled and not cancel_sent:
                                try:
                                    self._message_queue.put_nowait(Message.CANCEL)
                                    cancel_sent = True
                                except Full:
                                    pass
                            try:
                                response = self._response_queue.get(timeout=0.25)
                            except Empty:
                                if not self.is_alive():
                                    code = getattr(self.process, 'exitcode', None)
                                    raise RuntimeError(f"Dream Textures' background process exited (code {code}). Check the Blender console for its error, then retry.")
                                continue
                            except Exception as error:
                                raise RuntimeError(
                                    "The Dream Textures worker returned a value Blender could not read. "
                                    "Retry the operation; its private Python libraries remain separate."
                                ) from error
                            if isinstance(response, str) and response == Message.END:
                                future.set_done()
                            elif isinstance(response, TracedError):
                                response.base.__cause__ = Exception(response.trace)
                                future.set_exception(response.base)
                            elif isinstance(response, Exception):
                                future.set_exception(response)
                            else:
                                future.add_response(response)
                    except Exception as error:
                        # An interrupted/unreadable message can leave END in the
                        # queue. Replace the worker instead of pairing that stale
                        # terminator with the next generation request.
                        self.close()
                        future.set_exception(error)
                    finally:
                        if not future.done:
                            future.set_done()

            if _block:
                send_thread()
            else:
                threading.Thread(target=send_thread, daemon=True).start()
            return future
        return send

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Python teardown may have already disposed of multiprocessing.
            pass
