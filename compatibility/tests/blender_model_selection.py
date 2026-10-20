"""Optional real Blender 5.2.2 enum/storage acceptance; no model runtime.

Run only in a disposable factory scene:
  blender --background --factory-startup --python-exit-code 1 --python compatibility/tests/blender_model_selection.py

This script saves/reopens its own temporary .blend file. It is intentionally
separate from the offline runner and requires no model or runtime download.
"""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import bpy


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("selection_acceptance", ROOT / "model_selection.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)
helper = selection.ModelSelection("selected_identifier")
legacy_items = [None, ('owner/a', 'A', ''), ('owner/b', 'B', '')]
models = [None, NS(id='local:new', name='New', description=''),
          NS(id='owner/a', name='A', description=''), NS(id='owner/b', name='B', description='')]


class DTSelectionAcceptance(bpy.types.PropertyGroup):
    selected: bpy.props.EnumProperty(items=legacy_items)


def items(self, context):
    return helper.items(self, models)


def get_value(self, value, is_set):
    return helper.get(self, value, is_set, models)


def set_value(self, value, current, is_set):
    return helper.set(self, value, current, is_set, models)


bpy.utils.register_class(DTSelectionAcceptance)
bpy.types.Scene.dt_selection_acceptance = bpy.props.PointerProperty(type=DTSelectionAcceptance)
try:
    owner = bpy.context.scene.dt_selection_acceptance
    owner.selected = 'owner/b'
    # Replace only the RNA definition, retaining the legacy internally stored
    # numeric value. Inserting an automatic row must not shift that old value.
    del DTSelectionAcceptance.selected
    DTSelectionAcceptance.selected = bpy.props.EnumProperty(
        items=items, get_transform=get_value, set_transform=set_value)
    assert owner.selected == 'owner/b', owner.selected
    assert owner.get('selected_identifier') is None

    owner.selected = 'local:new'
    assert owner['selected_identifier'] == 'local:new'
    models.insert(0, NS(id='local:earlier', name='Earlier', description=''))
    assert owner.selected == 'local:new'
    models[:] = [model for model in models if model is None or model.id != 'local:new']
    assert owner.selected == 'local:new'  # Explicit unavailable row, never A/B.
    models.append(NS(id='local:new', name='New', description=''))

    with tempfile.TemporaryDirectory(prefix='dream-textures-selection-') as directory:
        filepath = str(Path(directory) / 'selection.blend')
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        helper = selection.ModelSelection('selected_identifier')
        models.reverse()
        bpy.ops.wm.open_mainfile(filepath=filepath)
        owner = bpy.context.scene.dt_selection_acceptance
        assert owner['selected_identifier'] == 'local:new'
        assert owner.selected == 'local:new'
    print(f'Model enum/storage acceptance passed in Blender {bpy.app.version_string}.')
finally:
    del bpy.types.Scene.dt_selection_acceptance
    bpy.utils.unregister_class(DTSelectionAcceptance)
