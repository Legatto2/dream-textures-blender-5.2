import bpy


def input_sockets(node_tree):
    """Iterate group inputs across the Blender 4.0 interface API change."""
    if hasattr(node_tree, "interface"):
        return (
            item for item in node_tree.interface.items_tree
            if item.item_type == 'SOCKET' and item.in_out == 'INPUT'
        )
    return iter(node_tree.inputs)


class DreamTexturesNodeTree(bpy.types.NodeTree):
    bl_idname = "DreamTexturesNodeTree"
    bl_label = "Dream Textures Node Editor"
    bl_description = "Nodes for the Dream Textures Render Engine"
    bl_icon = 'NODETREE'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'DREAM_TEXTURES'
