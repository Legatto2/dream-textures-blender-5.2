import bpy

def unified_paint_settings(context):
    # Blender 5 stores these settings per paint mode.
    paint = context.tool_settings.image_paint
    return getattr(paint, "unified_paint_settings", getattr(context.tool_settings, "unified_paint_settings", None))


def configure_inpaint_brush(brush):
    """Use the selected brush asset and return its settings for restoration."""
    brush_type = "image_brush_type" if hasattr(brush, "image_brush_type") else "image_tool"
    curve_preset = "curve_distance_falloff_preset" if hasattr(brush, "curve_distance_falloff_preset") else "curve_preset"
    values = {name: getattr(brush, name) for name in ("blend", curve_preset, "strength", brush_type)}
    setattr(brush, brush_type, 'DRAW')
    brush.blend = 'ERASE_ALPHA'
    setattr(brush, curve_preset, 'CONSTANT')
    brush.strength = 1.0
    return values


def restore_inpaint_brush(brush, values):
    if brush is not None and values is not None:
        try:
            for name, value in values.items():
                setattr(brush, name, value)
        except ReferenceError:
            # The image or blend file may have been closed before the tool.
            pass


class InpaintAreaBrushActivated(bpy.types.GizmoGroup):
    bl_idname = "dream_textures.inpaint_area_brush_activated"
    bl_label = "Inpaint Area Brush Activated"
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_region_type = 'WINDOW'

    def setup(self, context):
        self._brush = None
        self._settings = None
        self._timer = None
        self.refresh(context)

    def refresh(self, context):
        brush = context.tool_settings.image_paint.brush
        if brush == self._brush or self._timer is not None:
            return
        restore_inpaint_brush(self._brush, self._settings)
        self._brush = None
        self._settings = None
        if brush is None:
            return

        def set_blend():
            self._timer = None
            self._brush = brush
            self._settings = configure_inpaint_brush(brush)
        self._timer = set_blend
        bpy.app.timers.register(set_blend)

    def __del__(self):
        timer = getattr(self, "_timer", None)
        if timer is not None and bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
        restore_inpaint_brush(getattr(self, "_brush", None), getattr(self, "_settings", None))

class InpaintAreaBrush(bpy.types.WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'

    bl_idname = "dream_textures.inpaint_area_brush"
    bl_label = "Mark Inpaint Area"
    bl_description = "Mark an area for inpainting"
    bl_icon = "brush.paint_texture.mask"
    bl_widget = InpaintAreaBrushActivated.bl_idname
    bl_options = {'USE_BRUSHES'}
    bl_keymap = (
        ("paint.image_paint", {"type": 'LEFTMOUSE', "value": 'PRESS'}, {"properties": [("mode", 'NORMAL')]}),
    )

    def draw_settings(context, layout, tool):
        brush = context.tool_settings.image_paint.brush
        if brush is None:
            layout.label(text="Select a Paint brush from the asset shelf")
            return
        settings = unified_paint_settings(context)
        layout.prop(settings if settings is not None and settings.use_unified_size else brush, 'size')
