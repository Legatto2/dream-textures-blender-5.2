"""Run the focused compatibility suites without Blender or model downloads.

Python 3.11+ and NumPy are required. Each suite runs in its own interpreter so
the simulated dependency modules used by one suite cannot affect another.
The Blender/real-runtime suites in tests/ are intentionally separate.
"""

from pathlib import Path
import subprocess
import sys


SUITES = (
    "test_installer.py",
    "test_offload_lifecycle.py",
    "test_backend_backports.py",
    "test_projection_inputs.py",
    "test_model_discovery.py",
    "test_model_dispatch.py",
    "test_model_selection.py",
)


def main():
    tests = Path(__file__).resolve().parent / "tests"
    failed = []
    for suite in SUITES:
        print(f"\nRunning {suite}", flush=True)
        result = subprocess.run([sys.executable, "-B", str(tests / suite)])
        if result.returncode:
            failed.append(suite)
    if failed:
        print("\nFailed suites: " + ", ".join(failed), flush=True)
        return 1
    print(f"\nAll {len(SUITES)} offline suites passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
