"""Picklable worker probes used only by the compatibility integration tests."""

def describe():
    import sys
    import numpy
    import torch
    return {
        "python": list(sys.version_info[:3]),
        "executable": sys.executable,
        "numpy_path": numpy.__file__,
        "numpy_version": numpy.__version__,
        "torch_path": torch.__file__,
        "torch_version": str(torch.__version__),
        "bpy_imported": "bpy" in sys.modules,
    }


def array_result():
    import numpy
    return numpy.arange(12, dtype=numpy.float32).reshape(3, 4)


def dependency_error():
    from huggingface_hub.errors import HFValidationError
    raise HFValidationError("Deliberate worker exception for transport validation")


def progress():
    import time
    for index in range(100):
        yield index
        time.sleep(0.025)


def crash():
    import os
    os._exit(42)


def unsupported_dependency_value():
    import torch
    return torch.__version__  # A TorchVersion subclass deliberately requiring torch to unpickle.
