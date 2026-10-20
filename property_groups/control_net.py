import bpy
from bpy.props import FloatProperty, EnumProperty, PointerProperty, IntProperty, BoolProperty

from .. import api, image_utils
from ..diffusers_backend import DiffusersBackend
from ..generator_process import Generator
from ..generator_process.models.optimizations import Optimizations
from ..model_selection import ModelSelection

_control_net_selection = ModelSelection("_dream_textures_control_net_identifier")


def _control_net_selection_data(context):
    context = context or bpy.context
    try:
        backend = context.scene.dream_textures_prompt.get_backend()
        return backend.list_controlnet_models(context), getattr(backend, "resolve_model_identifier", None)
    except (AttributeError, KeyError, ReferenceError, RuntimeError):
        return [], None

def control_net_options(self, context):
    models, resolve = _control_net_selection_data(context)
    return _control_net_selection.items(self, models, resolve)


def _control_net_get(self, current_value, is_set):
    models, resolve = _control_net_selection_data(bpy.context)
    return _control_net_selection.get(self, current_value, is_set, models, resolve)


def _control_net_set(self, new_value, current_value, is_set):
    models, resolve = _control_net_selection_data(bpy.context)
    return _control_net_selection.set(self, new_value, current_value, is_set, models, resolve)

PROCESSOR_IDS = [
    ("none", "None", "No pre-processing"),
    None,
    ("depth_leres", "Depth (LeRes)", ""),
    ("depth_leres++", "Depth (LeRes++)", ""),
    ("depth_midas", "Depth (MiDaS)", ""),
    ("depth_zoe", "Depth (Zoe)", ""),
    None,
    ("canny", "Canny", "Canny edge detection"),
    ("mlsd", "M-LSD", ""),
    ("softedge_hed", "Soft Edge (HED)", ""),
    ("softedge_hedsafe", "Soft Edge (HED-Safe)", ""),
    ("softedge_pidinet", "Soft Edge (PidiNet)", ""),
    ("softedge_pidsafe", "Soft Edge (Pidsafe)", ""),
    None,
    ("lineart_anime", "Lineart (Anime)", ""),
    ("lineart_coarse", "Lineart (Coarse)", ""),
    ("lineart_realistic", "Lineart (Realistic)", ""),
    None,
    ("normal_bae", "Normal (BAE)", ""),
    None,
    ("openpose", "OpenPose", ""),
    ("openpose_face", "OpenPose (Face)", ""),
    ("openpose_faceonly", "OpenPose (Face Only)", ""),
    ("openpose_full", "OpenPose (Full)", ""),
    ("openpose_hand", "OpenPose (Hand)", ""),
    # ("dwpose", "DWPose", ""), # requires additional dependencies
    # ("mediapipe_face", "MediaPipe Face", ""), # requires additional dependencies
    None,
    ("scribble_hed", "Scribble (HED)", ""),
    ("scribble_pidinet", "Scribble (PidiNet)", ""),
    None,
    ("shuffle", "Shuffle", ""),
]

class ControlNet(bpy.types.PropertyGroup):
    control_net: EnumProperty(name="ControlNet", items=control_net_options,
                             get_transform=_control_net_get, set_transform=_control_net_set,
                             description="Specify which ControlNet to use")
    conditioning_scale: FloatProperty(name="Conditioning Scale", default=1.0, description="Increases the strength of the ControlNet's effect")
    control_image: PointerProperty(type=bpy.types.Image)
    processor_id: EnumProperty(
        name="Processor",
        items=PROCESSOR_IDS,
        description="Pre-process the control image"
    )
    enabled: BoolProperty(name="Enabled", default=True)

class ControlNetsAddMenu(bpy.types.Menu):
    bl_idname = "DREAM_MT_control_nets_add"
    bl_label = "Add ControlNet"

    def draw(self, context):
        layout = self.layout

        for model in control_net_options(None, context):
            if model is None:
                layout.separator()
            else:
                layout.operator("dream_textures.control_nets_add", text=model[1]).control_net = model[0]

class ControlNetsAdd(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_add"
    bl_label = "Add ControlNet"

    # The transient operator uses ordinary RNA storage and stable session enum
    # numbers. Only the resulting persistent PropertyGroup stores a full ID.
    control_net: EnumProperty(name="ControlNet", items=lambda self, context: control_net_options(None, context))

    def execute(self, context):
        if not any(item is not None and item[0] == self.control_net
                   for item in control_net_options(None, context)):
            self.report({'ERROR'}, "Select an available ControlNet model.")
            return {'CANCELLED'}
        net = context.scene.dream_textures_prompt.control_nets.add()
        net.control_net = self.control_net
        return {'FINISHED'}
class ControlNetsRemove(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_remove"
    bl_label = "Remove ControlNet"

    index: IntProperty(name="Index")

    def execute(self, context):
        context.scene.dream_textures_prompt.control_nets.remove(self.index)
        return {'FINISHED'}

class BakeControlNetImage(bpy.types.Operator):
    bl_idname = "dream_textures.control_net_bake"
    bl_label = "Bake Control Image"
    bl_description = "Runs the selected processor, and bakes the result to an image datablock"

    index: IntProperty(name="Index")

    @classmethod
    def poll(cls, context):
        if not Generator.runtime_available():
            cls.poll_message_set("Install Dream Textures dependencies in Preferences, then restart Blender.")
            return False
        return Generator.shared().can_use()

    def execute(self, context):
        prompt = context.scene.dream_textures_prompt

        net = prompt.control_nets[self.index]

        gen = Generator.shared()
        backend: api.Backend = prompt.get_backend()
        optimizations = backend.optimizations() if isinstance(backend, DiffusersBackend) else Optimizations()

        future = gen.controlnet_aux(
            processor_id=net.processor_id,
            image=image_utils.bpy_to_np(net.control_image, color_space=None),
            optimizations=optimizations
        )
        
        control_image = image_utils.np_to_bpy(
            future.result(last_only=True),
            f"{net.control_image.name} ({next(processor[1] for processor in PROCESSOR_IDS if processor != None and processor[0] == net.processor_id)})",
        )
        
        net.control_image = control_image
        net.processor_id = "none"

        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                area.spaces.active.image = control_image

        return {'FINISHED'}
