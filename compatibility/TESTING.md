# Reproducing the checks

These commands are for development. Normal Windows setup uses the buttons described in the main README.

The tests were run with the official Blender 5.2.2 Linux distribution and its embedded Python 3.13.13. Set the following paths to an extracted copy of the source ZIP and an installed test Blender:

~~~bash
DT_SOURCE=/absolute/path/to/dream_textures
DT_BLENDER=/absolute/path/to/blender-5.2.2-linux-x64/blender
DT_PYTHON=/absolute/path/to/blender-5.2.2-linux-x64/5.2/python/bin/python3.13
DT_PROFILE=/absolute/path/to/disposable-dream-profile
~~~

## Installer invariants

~~~bash
"$DT_PYTHON" "$DT_SOURCE/compatibility/tests/test_installer.py"
~~~

The tests create and remove their own temporary directories. They do not install a production GPU runtime.

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
  --source "$DT_SOURCE" --output /absolute/path/to/Dream-Textures.zip
~~~

The build helper creates a file-hash manifest, excludes caches/old dependency placeholders, and writes one top-level dream_textures folder. It validates the resulting ZIP CRCs. The supplied add-on source remains GPL-3.0-or-later.

Windows wheel-resolution evidence and dependency hashes are included in evidence/final-windows-resolution.json. A target-platform resolver check does not replace running the Windows installer and CUDA smoke test.
