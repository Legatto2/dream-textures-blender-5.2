import bpy
from bpy.types import Panel, Operator, Menu
from bl_operators.presets import AddPresetBase
from bl_ui.utils import PresetPanel
import os
import shutil
from ..absolute_path import absolute_path
from ..generator_process.actions.prompt_to_image import Optimizations

class DreamTexturesPresetPanel(PresetPanel, Panel):
    preset_operator = "script.execute_preset"

class DREAM_PT_AdvancedPresets(DreamTexturesPresetPanel):
    bl_label = "Advanced Presets"
    preset_subdir = "dream_textures/advanced"
    preset_add_operator = "dream_textures.advanced_preset_add"

class DREAM_MT_AdvancedPresets(Menu):
    bl_label = 'Advanced Presets'
    preset_subdir = 'dream_textures/advanced'
    preset_operator = 'script.execute_preset'
    draw = Menu.draw_preset

class AddAdvancedPreset(AddPresetBase, Operator):
    bl_idname = 'dream_textures.advanced_preset_add'
    bl_label = 'Add Advanced Preset'
    preset_menu = 'DREAM_MT_AdvancedPresets'
    
    preset_subdir = 'dream_textures/advanced'
    
    preset_defines = [
        'prompt = bpy.context.scene.dream_textures_prompt',
        'backend = prompt.get_backend()',
    ]
    preset_values = [
        "prompt.steps",
        "prompt.cfg_scale",
        "prompt.scheduler",
        "prompt.step_preview_mode",

        "backend.attention_slicing",
        "backend.attention_slice_size_src",
        "backend.attention_slice_size",
        "backend.cudnn_benchmark",
        "backend.tf32",
        "backend.half_precision",
        "backend.cpu_offload",
        "backend.channels_last_memory_format",
        "backend.sdp_attention",
        "backend.batch_size",
        "backend.vae_slicing",
        "backend.cpu_only",
    ]

class RestoreDefaultPresets(Operator):
    bl_idname = "dream_textures.restore_default_presets"
    bl_label = "Restore Default Presets"
    bl_description = ("Restores all default presets provided by the addon.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        register_default_presets(force=True)
        return {"FINISHED"}

PRESETS_PATH = os.path.join(bpy.utils.user_resource('SCRIPTS'), 'presets/dream_textures/advanced')
DEFAULT_PRESETS_PATH = absolute_path('builtin_presets')
def register_default_presets(force=False):
    os.makedirs(PRESETS_PATH, exist_ok=True)
    for default_preset in os.listdir(DEFAULT_PRESETS_PATH):
        source = os.path.join(DEFAULT_PRESETS_PATH, default_preset)
        if not default_preset.endswith('.py') or not os.path.isfile(source):
            continue
        if force or not os.path.exists(os.path.join(PRESETS_PATH, default_preset)):
            shutil.copy(source, PRESETS_PATH)

def default_presets_missing():
    if not os.path.isdir(PRESETS_PATH):
        return True
    for default_preset in os.listdir(DEFAULT_PRESETS_PATH):
        if not default_preset.endswith('.py') or not os.path.isfile(os.path.join(DEFAULT_PRESETS_PATH, default_preset)):
            continue
        if not os.path.exists(os.path.join(PRESETS_PATH, default_preset)):
            return True
