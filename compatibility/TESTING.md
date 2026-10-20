# Reproducing the checks

These commands are for development. Normal Windows setup uses the buttons described in the main README.

The historical execution logs describe the **0.4.2-compat.1** implementation. **0.4.2-compat.2** updates attribution, documentation, and build labels only; its package/source verification is described in [BUILD_REPORT.md](BUILD_REPORT.md). The original application/runtime test suite was not repeated for that revision.

The original application/runtime tests were run with the official Blender 5.2.2 Linux distribution and its embedded Python 3.13.13. Set the following paths to an extracted copy of the source ZIP and an installed test Blender:

~~~bash
DT_SOURCE=/absolute/path/to/dream_textures
DT_BLENDER=/absolute/path/to/blender-5.2.2-linux-x64/blender
DT_PYTHON=/absolute/path/to/blender-5.2.2-linux-x64/5.2/python/bin/python3.13
DT_PROFILE=/absolute/path/to/disposable-dream-profile
~~~

## Current offline compatibility checks

For **0.4.2-compat.3**, run the explicitly selected suites through [run_offline_tests.py](run_offline_tests.py). They use Python 3.10+ and NumPy, temporary checkpoint files, and simulated Blender/dependency/device boundaries. They do not run Blender, load a pretrained model, or install the production runtime. Each suite gets a separate interpreter so its module stubs cannot affect another suite.

~~~bash
python3 -m pip install numpy==2.3.4
python3 -B "$DT_SOURCE/compatibility/run_offline_tests.py"
~~~

The NumPy pin matches the historical Blender frontend version and the new Python 3.13 CI lane. The local combined check passed **83 tests across seven suites**, using Linux Python 3.12.14 and NumPy 2.3.5. Its output is retained in [offline-backports-2026-10-20.log](evidence/offline-backports-2026-10-20.log).

| Suite | Coverage |
|---|---|
| [Private installer](tests/test_installer.py) | Runtime isolation, activation, cancellation, rollback, and command invariants. |
| [Offload lifecycle](tests/test_offload_lifecycle.py) | Repeated cached calls, hook ownership/cleanup, ControlNet replacement, and recommended-model preference UI. |
| [Backend backports](tests/test_backend_backports.py) | Cancellation callback binding, supported non-empty ControlNet weight files and format/precision fallback, Hub symlinks, and DPT processor selection. |
| [Projection inputs](tests/test_projection_inputs.py) | Finite masked depth, empty/flat captures, correct input/task/orientation, invoking viewport, owned diagnostics, and failure cleanup. |
| [Model discovery](tests/test_model_discovery.py) | Bounded header validation and scanning, malformed files, supported/ambiguous architectures, identity/config precedence, catalog replacement, cancellation, and reload guards. |
| [Model dispatch](tests/test_model_dispatch.py) | Local and Hub ControlNet resolution for all supported task branches and separate display labels/identifiers. |
| [Model selection](tests/test_model_selection.py) | Full-identifier storage through menu reordering, removal, saved-state recreation, explicit-link aliases, and legacy numeric fallback. |

The [Compatibility smoke checks workflow](https://github.com/Legatto2/dream-textures-blender-5.2/blob/blender-5.2-compat/.github/workflows/compatibility-smoke.yml) runs the current offline suites on Python 3.13, compiles the Python source, builds the add-on ZIP, checks that the committed manifest matches, and retains the package as an Actions artifact. Its GPU dependencies are still installed separately by the private runtime installer. This workflow does not publish a GitHub release or perform Windows/Blender/CUDA acceptance.

Avoid broad unittest discovery across `compatibility/tests`: historical files in that directory require Blender or a real runtime. The focused runner deliberately names its supported suites.

### Optional live model-selection acceptance

[blender_model_selection.py](tests/blender_model_selection.py) checks the actual Blender enum transform/storage contract, including a temporary `.blend` save/reopen. It is separate from the offline runner and **has not been executed for this revision** because the current development environment has no Blender binary. Run it only in a disposable factory scene:

~~~bash
"$DT_BLENDER" --background --factory-startup --python-exit-code 1 \
  --python "$DT_SOURCE/compatibility/tests/blender_model_selection.py"
~~~

This script does not perform model inference or validate projection rendering. Broader Blender and Windows acceptance remains described in the current validation record.

## Focused offload lifecycle regression

This individual suite requires only Python 3.10 or newer and the standard library. It loads the current source by default and simulates dependency/device behavior; it does not run Blender or CUDA.

~~~bash
python3 -B "$DT_SOURCE/compatibility/tests/test_offload_lifecycle.py"
~~~

To compare against an extracted pre-hotfix source checkout, pass `--source /absolute/path/to/baseline`. Failures are expected for the original loader and preference UI. When using unittest discovery, restrict it to `test_offload_lifecycle.py`; other files in this directory require Blender.

The current run is recorded in [offload-lifecycle-2026-10-20.log](evidence/offload-lifecycle-2026-10-20.log). The [Windows validation record](VALIDATION_2026-10-20.md) separately documents the maintainer's actual GPU/projection tests. The full historical 47-test suite was not rerun for this update.

## Installer invariants

~~~bash
"$DT_PYTHON" "$DT_SOURCE/compatibility/tests/test_installer.py"
~~~

The tests create and remove their own temporary directories. They do not install a production GPU runtime. The current rerun is recorded in [installer-2026-10-20.log](evidence/installer-2026-10-20.log).

## Blender frontend without AI dependencies

Use a fresh disposable profile without a selected runtime:

~~~bash
BLENDER_USER_RESOURCES="$DT_PROFILE" \
"$DT_BLENDER" --background --factory-startup --offline-mode \
  --python-exit-code 1 \
  --python "$DT_SOURCE/compatibility/tests/test_blender_compat.py" \
  -- "$(dirname "$DT_SOURCE")"
~~~

## Install an isolated CPU test runtime

The following downloads the pinned CPU dependencies and exercises the real installer. It is a development-only lane.

~~~bash
"$DT_PYTHON" -I -S "$DT_SOURCE/compatibility_installer.py" install \
  --root "$DT_PROFILE/config/dream_textures_compat" \
  --requirements "$DT_SOURCE/requirements/compat-runtime.txt" \
  --torch-requirements "$DT_SOURCE/requirements/compat-torch-cpu-test.txt" \
  --wheel-dir "$DT_SOURCE/wheels" \
  --cpu-test-only
~~~

Read the active runtime ID from the printed installation result or from runtime.json and set DT_RUNTIME to the full resulting directory.

~~~bash
DT_RUNTIME=/absolute/path/to/disposable-dream-profile/config/dream_textures_compat/runtimes/runtime-ID
"$DT_PYTHON" -I -S "$DT_SOURCE/compatibility_installer.py" validate \
  --runtime "$DT_RUNTIME" \
  --result "$DT_PROFILE/runtime-check.json" \
  --cpu-test-only
"$DT_PYTHON" "$DT_SOURCE/compatibility/tests/backend_smoke.py" \
  --runtime "$DT_RUNTIME" --addon "$DT_SOURCE"
~~~

The backend suite uses random weights and offline mode. It does not download a pretrained model.

## Blender worker integration

Run this in Linux, where the explicit CPU development flag is allowed. Windows UI runtime selection always requires CUDA.

~~~bash
BLENDER_USER_RESOURCES="$DT_PROFILE" \
BLENDER_USER_CONFIG="$DT_PROFILE/config" \
HF_HOME="$DT_PROFILE/hf-cache" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
DREAM_TEXTURES_TEST_ALLOW_CPU=1 \
DREAM_TEXTURES_ADDON_SOURCE="$DT_SOURCE" \
"$DT_BLENDER" --background --factory-startup --offline-mode \
  --python-exit-code 1 \
  --python "$DT_SOURCE/compatibility/tests/test_actor_integration.py"
~~~

This suite deliberately crashes a worker and returns an unreadable dependency-specific value to check recovery. Those deliberate failures are successful tests when the suite reports OK.

## Packaging

~~~bash
"$DT_PYTHON" "$DT_SOURCE/compatibility/build_zip.py" \
  --source "$DT_SOURCE" --output /absolute/path/to/Dream-Textures-0.4.2-compat3-Blender52-Windows.zip
~~~

The build helper labels the source **0.4.2-compat.3**, regenerates the current file-hash manifest, excludes caches/old dependency placeholders, and writes one top-level dream_textures folder. It validates the resulting ZIP CRCs. The supplied add-on source remains GPL-3.0-or-later.

Windows wheel-resolution evidence and dependency hashes are included in evidence/final-windows-resolution.json. A target-platform resolver check does not replace running the Windows installer and CUDA smoke test.
