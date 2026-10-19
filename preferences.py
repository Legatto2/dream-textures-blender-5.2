import bpy
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import webbrowser
import textwrap

from .absolute_path import absolute_path
from .operators.install_dependencies import (
    InstallDependencies, UninstallDependencies, CancelDependencyInstall,
    VerifyRuntime, OpenRuntimeLog, RestorePreviousRuntime,
    current_runtime_status, current_job_status, job_running,
)
from .ui.presets import RestoreDefaultPresets, default_presets_missing
from .generator_process import Generator
from .generator_process.actions.huggingface_hub import DownloadStatus, Model as HubModel
from .generator_process.models import Checkpoint, ModelConfig, ModelType

is_downloading = False
_preferences_generation = None
_initial_models_timer = None


def _preferences_for_callback(generation):
    """Old worker callbacks must not touch preferences after disable or reload."""
    if generation is None or generation is not _preferences_generation:
        return None
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        return addon.preferences if addon is not None else None
    except (AttributeError, ReferenceError, RuntimeError):
        return None

class OpenURL(bpy.types.Operator):
    bl_idname = "dream_textures.open_url"
    bl_label = "Get Access Token"
    bl_description = ("Opens huggingface.co to the tokens page")
    bl_options = {"REGISTER", "INTERNAL"}

    url: bpy.props.StringProperty(name="URL")

    def execute(self, context):
        webbrowser.open(self.url)
        return {"FINISHED"}

_model_config_options = [(m.name, m.value, '') for m in ModelConfig]
import_extensions = ['.ckpt', '.safetensors', '.pth']
import_extensions_glob = ";".join(import_extensions).replace(".", "*.")
class ImportWeights(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.import_weights"
    bl_label = "Import Checkpoint File"
    filename_ext = ".ckpt"
    filter_glob: bpy.props.StringProperty(
        default=import_extensions_glob,
        options={'HIDDEN'},
        maxlen=255,
    )
    model_config: bpy.props.EnumProperty(
        name="Model Config",
        items=_model_config_options
    )
    prefer_fp16_variant: bpy.props.BoolProperty(
        name="Save Half Precision Weights",
        default=True
    )

    def execute(self, context):
        global is_downloading
        is_downloading = True
        generation = _preferences_generation
        f = Generator.shared().convert_original_stable_diffusion_to_diffusers(self.filepath, ModelConfig[self.model_config], self.prefer_fp16_variant)
        def on_progress(_, response: DownloadStatus):
            pref = _preferences_for_callback(generation)
            if pref is not None:
                pref.download_file = response.file
                pref.download_progress = int((response.index / max(response.total, 1)) * 100)
        def on_done(future):
            global is_downloading
            if _preferences_for_callback(generation) is None:
                return
            is_downloading = False
            if future.exception() is None:
                fetch_installed_models()
        def on_exception(_, exception):
            if _preferences_for_callback(generation) is None:
                return
            self.report({"ERROR"}, str(exception))
            raise exception
        f.add_response_callback(on_progress)
        f.add_done_callback(on_done)
        f.add_exception_callback(on_exception)
        return {"FINISHED"}

class Model(bpy.types.PropertyGroup):
    bl_label = "Model"
    bl_idname = "dream_textures.Model"

    model: bpy.props.StringProperty(name="Model")
    model_base: bpy.props.StringProperty()
    downloads: bpy.props.IntProperty(name="Downloads")
    likes: bpy.props.IntProperty(name="Likes")
    model_type: bpy.props.EnumProperty(name="Model Type", items=[(t.name, t.name, '') for t in ModelType])

class PREFERENCES_UL_ModelList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        model_name = item.model
        is_installed = False
        if os.path.exists(item.model):
            model_name = os.path.basename(item.model).replace('models--', '').replace('--', '/')
            is_installed = True
        split = layout.split(factor=0.75)
        split.label(text=model_name)
        if item.downloads != -1:
            split.label(text=str(item.downloads), icon="IMPORT")
        if item.downloads != -1:
            split.label(text=str(item.likes), icon="HEART")
        if ModelType[item.model_type] != ModelType.UNKNOWN:
            split.label(text=item.model_type.replace('_', ' ').title())
        install_model = layout.operator(InstallModel.bl_idname, text="", icon="FILE_FOLDER" if is_installed else "IMPORT")
        install_model.model = item.model
        install_model.prefer_fp16_variant = data.prefer_fp16_variant
        install_model.resume_download = data.resume_download

def set_model_list(model_list: str, models: list):
    pref = _preferences_for_callback(_preferences_generation)
    if pref is None:
        return
    getattr(pref, model_list).clear()
    for model in models:
        m = getattr(pref, model_list).add()
        m.model = model.id
        m.model_base = os.path.basename(model.id)
        m.downloads = model.downloads
        m.likes = model.likes
        try:
            m.model_type = model.model_type.name
        except:
            pass

class checkpoint_lookup:
    _checkpoints = {}

    @classmethod
    def get(cls, item):
        return cls._checkpoints.get(item, item)

class model_lookup:
    _models = {}

    @classmethod
    def get(cls, item):
        return cls._models.get(item, None)

def fetch_installed_models(blocking=True):
    if not current_runtime_status()["ready"]:
        return
    generation = _preferences_generation

    def on_done(future):
        pref = _preferences_for_callback(generation)
        if pref is None:
            return
        model_list = future.result()

        model_lookup._models = { os.path.basename(model.id).replace('models--', '').replace('--', '/'): model for model in model_list }

        checkpoint_links = ((link.path, ModelConfig[link.model_config]) for link in pref.linked_checkpoints)
        checkpoints = {}
        for path, config in checkpoint_links:
            if not os.path.exists(path):
                continue
            if os.path.isfile(path):
                checkpoints[os.path.basename(path)] = (path, config)
                continue
            for name in os.listdir(path):
                if os.path.splitext(name)[1] not in import_extensions:
                    continue
                if name in checkpoints:
                    # file linked config takes precedence over folder linked config
                    continue
                checkpoints[name] = (os.path.join(path, name), config)
        checkpoint_lookup._checkpoints.clear()
        for path, config in checkpoints.values():
            model = HubModel(path, "", [], -1, -1, ModelType.from_config(config))
            model_list.append(model)
            checkpoint_lookup._checkpoints[os.path.basename(path)] = Checkpoint(path, config)
            model_lookup._models[os.path.basename(path)] = model

        set_model_list('installed_models', model_list)

    future = Generator.shared().hf_list_installed_models()
    if blocking:
        on_done(future)
    else:
        future.add_done_callback(on_done)


class ModelSearch(bpy.types.Operator):
    bl_idname = "dream_textures.model_search"
    bl_label = "Search"
    bl_description = ("Searches Hugging Face Hub for models")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        
        return {"FINISHED"}

class InstallModel(bpy.types.Operator):
    bl_idname = "dream_textures.install_model"
    bl_label = "Install or Open"
    bl_description = ("Install or open a model from the cache")
    bl_options = {"REGISTER", "INTERNAL"}

    model: StringProperty(name="Model ID")
    prefer_fp16_variant: bpy.props.BoolProperty(name="", default=True)
    resume_download: bpy.props.BoolProperty(name="", default=True)

    def execute(self, context):
        if os.path.exists(self.model):
            if os.path.isfile(self.model):
                webbrowser.open(f"file://{os.path.dirname(self.model)}")
            else:
                webbrowser.open(f"file://{self.model}")
        else:
            global is_downloading
            is_downloading = True
            generation = _preferences_generation
            f = Generator.shared().hf_snapshot_download(
                self.model,
                bpy.context.preferences.addons[__package__].preferences.hf_token,
                "fp16" if self.prefer_fp16_variant else None,
                self.resume_download
            )
            def on_progress(_, response: DownloadStatus):
                pref = _preferences_for_callback(generation)
                if pref is not None:
                    pref.download_file = response.file
                    pref.download_progress = int((response.index / max(response.total, 1)) * 100)
            def on_done(future):
                global is_downloading
                if _preferences_for_callback(generation) is None:
                    return
                is_downloading = False
                if future.exception() is None:
                    fetch_installed_models()
            def on_exception(_, exception):
                if _preferences_for_callback(generation) is None:
                    return
                self.report({"ERROR"}, str(exception))
                raise exception
            f.add_response_callback(on_progress)
            f.add_done_callback(on_done)
            f.add_exception_callback(on_exception)
        return {"FINISHED"}

def _model_search(self, context):
    if not current_runtime_status()["ready"]:
        return
    generation = _preferences_generation
    def on_done(future):
        if _preferences_for_callback(generation) is None:
            return
        set_model_list('model_results', future.result())
    Generator.shared().hf_list_models(self.model_query, self.hf_token).add_done_callback(on_done)

def _update_ui(self, context):
    if hasattr(context.area, "regions"):
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()
    return None

def _template_model_download_progress(context, layout):
    global is_downloading
    preferences = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences
    if is_downloading:
        progress_col = layout.column()
        progress_col.label(text=preferences.download_file)
        progress_col.prop(preferences, "download_progress", slider=True)
        progress_col.enabled = False
    return is_downloading

class CheckpointGroup(bpy.types.PropertyGroup):
    bl_label = "Model"
    bl_idname = "dream_textures.checkpoint"

    path: bpy.props.StringProperty(name="Checkpoint")
    model_config: bpy.props.EnumProperty(
        name="Model Config",
        items=_model_config_options
    )

class LinkCheckpoint(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.link_checkpoint"
    bl_label = "Link Checkpoint File or Folder"
    filename_ext = ".ckpt"
    files: CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'}
    )
    filter_glob: bpy.props.StringProperty(
        default=import_extensions_glob,
        options={'HIDDEN'},
        maxlen=255,
    )
    model_config: bpy.props.EnumProperty(
        name="Model Config",
        items=_model_config_options
    )

    def invoke(self, context, _event):
        if os.path.isfile(self.filepath):
            # Reset to a directory, otherwise the filename remains populated and can cause issues to select a directory if gone unnoticed.
            self.filepath = os.path.dirname(self.filepath) + os.path.sep
        return super().invoke(context, _event)

    def execute(self, context):
        pref = context.preferences.addons[__package__].preferences
        for file in self.files:
            path = self.filepath
            if file.name != "":
                path = os.path.join(os.path.dirname(path), file.name)

            if not os.path.exists(path):
                self.report({"ERROR"}, f"{path} does not exist")
                continue
            if os.path.isfile(path) and os.path.splitext(path)[1] not in import_extensions:
                self.report({"ERROR"}, f"{os.path.basename(path)} is not a checkpoint")
                continue

            link = next((link for link in pref.linked_checkpoints if link.path == path), None)
            if link is None:
                link = pref.linked_checkpoints.add()
                link.path = path
            link.model_config = self.model_config

        fetch_installed_models()

        return {"FINISHED"}

class UnlinkCheckpoint(bpy.types.Operator):
    bl_idname = "dream_textures.unlink_checkpoint"
    bl_label = "Unlink Checkpoint File"

    path: bpy.props.StringProperty()
    def execute(self, context):
        pref = context.preferences.addons[__package__].preferences
        index = next((i for i, link in enumerate(pref.linked_checkpoints) if link.path == self.path), -1)
        if index != -1:
            pref.linked_checkpoints.remove(index)

        fetch_installed_models()

        return {"FINISHED"}

class PREFERENCES_UL_CheckpointList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        split = layout.split(factor=0.75)
        split.label(text=item.path)
        split.label(text=ModelConfig[item.model_config].value)
        install_model = layout.operator(InstallModel.bl_idname, text="", icon="FILE_FOLDER")
        install_model.model = item.path
        unlink = layout.operator(UnlinkCheckpoint.bl_idname, text="", icon="TRASH")
        unlink.path = item.path

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    dream_studio_key: StringProperty(name="DreamStudio Key", subtype='PASSWORD')

    model_query: StringProperty(name="Search", update=_model_search)
    model_results: CollectionProperty(type=Model)
    active_model_result: bpy.props.IntProperty(name="Active Model", default=0)
    hf_token: StringProperty(name="HuggingFace Token", subtype='PASSWORD')
    prefer_fp16_variant: bpy.props.BoolProperty(name="Prefer Half Precision Weights", description="Download fp16 weights if available for smaller file size. If you run with 'Half Precision' disabled, you should not use this setting", default=True)
    resume_download: bpy.props.BoolProperty(name="Resume Incomplete Download", description="Continue an in-progress download in case if Blender was closed or connection was interrupted, otherwise incomplete files will be entirely redownloaded", default=True)

    installed_models: CollectionProperty(type=Model)
    active_installed_model: bpy.props.IntProperty(name="Active Model", default=0)

    linked_checkpoints: CollectionProperty(type=CheckpointGroup)
    active_linked_checkpoint: bpy.props.IntProperty(name="Active Checkpoint", default=0)

    download_file: bpy.props.StringProperty(name="")
    download_progress: bpy.props.IntProperty(name="", min=0, max=100, subtype="PERCENTAGE", update=_update_ui)

    model_cache = []

    @staticmethod
    def register():
        # The preferences entry is added after class registration. Even an
        # already installed runtime must wait until Blender finishes enabling.
        global _preferences_generation, _initial_models_timer
        generation = _preferences_generation = object()

        def initial_refresh():
            global _initial_models_timer
            _initial_models_timer = None
            if _preferences_for_callback(generation) is not None and current_runtime_status()["ready"]:
                fetch_installed_models(False)
            return None

        _initial_models_timer = initial_refresh
        bpy.app.timers.register(initial_refresh, first_interval=0.1)

    @staticmethod
    def unregister():
        global _preferences_generation, _initial_models_timer, is_downloading
        _preferences_generation = None
        is_downloading = False
        if _initial_models_timer is not None:
            if bpy.app.timers.is_registered(_initial_models_timer):
                bpy.app.timers.unregister(_initial_models_timer)
            _initial_models_timer = None

    def draw(self, context):
        layout = self.layout

        weights_installed = len(self.installed_models) > 0

        if not weights_installed:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        runtime = current_runtime_status()
        runtime_box = layout.box()
        runtime_box.label(text="Dream Textures — Blender 5.2 Compatibility", icon="PREFERENCES")
        if runtime["restart_required"]:
            runtime_box.label(text="Dependencies installed. Restart Blender to continue.", icon="FILE_REFRESH")
        elif runtime["ready"]:
            manifest = runtime["manifest"]
            runtime_box.label(text="GPU runtime checked: " + manifest.get("device", "CUDA available"), icon="CHECKMARK")
            versions = manifest.get("packages", {})
            runtime_box.label(text="PyTorch " + versions.get("torch", "installed") + " · CUDA " + str(manifest.get("cuda_version", "test runtime")))
        else:
            runtime_box.label(text="Install the GPU dependencies to finish setup.", icon="INFO")
            runtime_box.label(text="Uses Blender's Python 3.13 and a private CUDA 12.8 runtime.")

        status = current_job_status()
        if status:
            phase = status.get("phase", "")
            icon = "ERROR" if phase == "failed" else "INFO"
            for line in textwrap.wrap(status.get("message", ""), width=85):
                runtime_box.label(text=line, icon=icon)
        row = runtime_box.row(align=True)
        row.operator(InstallDependencies.bl_idname, text="Reinstall GPU Dependencies" if runtime["selected"] else "Install GPU Dependencies", icon="IMPORT")
        if job_running():
            row.operator(CancelDependencyInstall.bl_idname, icon="CANCEL")
        else:
            row.operator(VerifyRuntime.bl_idname, icon="CHECKMARK")
        runtime_box.operator(OpenRuntimeLog.bl_idname, icon="TEXT")
        if runtime["previous"]:
            runtime_box.operator(RestorePreviousRuntime.bl_idname, icon="RECOVER_LAST")

        has_dependencies = runtime["ready"]
        if has_dependencies:
            if not _template_model_download_progress(context, layout):
                if not weights_installed:
                    default_weights_box = layout.box()
                    default_weights_box.label(text="You need to download at least one model.")
                    install_model = default_weights_box.operator(InstallModel.bl_idname, text="Download Stable Diffusion 1.5", icon="IMPORT")
                    install_model.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
                    install_model.prefer_fp16_variant = self.prefer_fp16_variant
                    install_model.resume_download = self.resume_download

                search_box = layout.box()
                search_box.label(text="Find Models", icon="SETTINGS")
                search_box.label(text="Search Hugging Face Hub for more compatible models.")

                search_box.prop(self, "model_query", text="", icon="VIEWZOOM")
                
                if len(self.model_results) > 0:
                    search_box.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_model_results", self, "model_results", self, "active_model_result")

                search_box.label(text="Some models require authentication. Provide a token to download gated models.")

                auth_row = search_box.row()
                auth_row.prop(self, "hf_token", text="Token")
                auth_row.operator(OpenURL.bl_idname, text="Get Your Token", icon="KEYINGSET").url = "https://huggingface.co/settings/tokens"
                
                search_box.prop(self, "prefer_fp16_variant")
                search_box.prop(self, "resume_download")

            layout.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_installed_models", self, "installed_models", self, "active_installed_model")
            import_weights = layout.operator(ImportWeights.bl_idname, icon='IMPORT')
            import_weights.prefer_fp16_variant = self.prefer_fp16_variant
            layout.template_list(PREFERENCES_UL_CheckpointList.__name__, "dream_textures_linked_checkpoints", self, "linked_checkpoints", self, "active_linked_checkpoint")
            layout.operator(LinkCheckpoint.bl_idname, icon='FOLDER_REDIRECT')

            if weights_installed or len(self.dream_studio_key) > 0:
                complete_box = layout.box()
                complete_box.label(text="Addon Setup Complete", icon="CHECKMARK")
                complete_box.label(text="To locate the interface:")
                complete_box.label(text="1. Open an Image Editor or Shader Editor space")
                complete_box.label(text="2. Enable 'View' > 'Sidebar'")
                complete_box.label(text="3. Select the 'Dream' tab")
            
            if default_presets_missing():
                presets_box = layout.box()
                presets_box.label(text="Default Presets", icon="PRESET")
                presets_box.label(text="It looks like you removed some of the default presets.")
                presets_box.label(text="You can restore them here.")
                presets_box.operator(RestoreDefaultPresets.bl_idname, icon="RECOVER_LAST")
        elif not runtime["restart_required"]:
            layout.label(text="After installation, restart Blender and download or link a model.")

        contributors_box = layout.box()
        contributors_box.label(text="Contributors", icon="COMMUNITY")
        contributors_box.label(text="Dream Textures is made possible by the contributors on GitHub.")
        contributors_box.operator(OpenURL.bl_idname, text="See All Contributors", icon="URL").url = "https://github.com/carson-katri/dream-textures/graphs/contributors"

        if context.preferences.view.show_developer_ui:
            developer_box = layout.box()
            developer_box.label(text="Runtime Maintenance", icon="CONSOLE")
            developer_box.operator(UninstallDependencies.bl_idname, icon="TRASH")
