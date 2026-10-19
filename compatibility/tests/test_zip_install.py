"""Install the distributable ZIP in an isolated Blender profile and verify it.

Run with BLENDER_USER_RESOURCES set to a fresh disposable directory:
    blender --background --factory-startup --offline-mode --python-exit-code 1 \
        --python tests/test_zip_install.py -- /absolute/path/to/addon.zip
"""

import hashlib
import os
from pathlib import Path
import sys
import zipfile

import bpy
import cycles


archive = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
profile = Path(os.environ["BLENDER_USER_RESOURCES"]).resolve()
if not zipfile.is_zipfile(archive):
    raise RuntimeError(f"Not a complete ZIP archive: {archive}")

original_render = cycles.CyclesRender.render
original_passes = cycles.CyclesRender.update_render_passes
print(f"Blender: {bpy.app.version_string}", flush=True)
print(f"Archive: {archive}", flush=True)
print(f"SHA256: {hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()}", flush=True)
print(f"Disposable profile: {profile}", flush=True)

result = bpy.ops.preferences.addon_install(filepath=str(archive), overwrite=False)
assert result == {'FINISHED'}, result
print("PASS: actual distributable ZIP installed through Blender's add-on installer", flush=True)

result = bpy.ops.preferences.addon_enable(module="dream_textures")
assert result == {'FINISHED'}, result
addon = sys.modules["dream_textures"]
installed_path = Path(addon.__file__).resolve()
assert installed_path.is_relative_to(profile), installed_path
assert "dream_textures" in bpy.context.preferences.addons
assert addon.bl_info["version"] == (0, 4, 2), addon.bl_info["version"]
assert hasattr(bpy.types.Scene, "dream_textures_prompt")
assert cycles.CyclesRender.render is not original_render
backend = bpy.context.scene.dream_textures_prompt.get_backend()
assert backend.cpu_offload == 'model', backend.cpu_offload
print(f"PASS: installed copy enabled from {installed_path}", flush=True)
print(f"PASS: version {addon.bl_info['version']}; CPU offload default {backend.cpu_offload!r}", flush=True)

result = bpy.ops.preferences.addon_disable(module="dream_textures")
assert result == {'FINISHED'}, result
assert not hasattr(bpy.types.Scene, "dream_textures_prompt")
assert not hasattr(bpy.types.Scene, "dream_textures_engine_prompt")
assert not hasattr(bpy.types.Scene, "dream_textures_render_engine")
assert cycles.CyclesRender.render is original_render
assert cycles.CyclesRender.update_render_passes is original_passes
assert "dream_textures" not in bpy.context.preferences.addons
print("PASS: installed copy disabled; RNA properties removed and Cycles restored", flush=True)
print("ZIP INSTALL SMOKE: PASS", flush=True)
