"""Exercise actual backend dispatch/selector methods without Blender or models."""
import ast
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[2]


def backend_namespace():
    ns = {'__name__': __name__, 'NDArray': object}
    task_tree = ast.parse((ROOT / 'api/models/task.py').read_text())
    task_tree.body = [node for node in task_tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'numpy.typing')]
    exec(compile(task_tree, 'task.py', 'exec'), ns)
    tree = ast.parse((ROOT / 'diffusers_backend.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DiffusersBackend')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {'generate', 'list_models', 'list_controlnet_models', 'resolve_model_identifier'}]
    ns.update({
        'GenerationArguments': object, 'GenerationResult': object, 'StepCallback': object,
        'Callback': object, 'Future': object, 'Model': lambda **kwargs: NS(**kwargs), 'reduce': reduce,
        'ModelType': Enum('ModelType', ['PROMPT_TO_IMAGE', 'CONTROL_NET', 'UNKNOWN']),
        'StableDiffusionPreferences': NS(bl_idname='test_addon'),
        'Scheduler': lambda name: name,
    })
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'diffusers_backend.py', 'exec'), ns)
    return ns


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.ns = backend_namespace()
        self.calls = []
        self.local_base = NS(path='/models/base.safetensors', config='explicit-base-config')
        self.local_control = NS(path='/models/depth--local.safetensors', config='explicit-control-config')
        lookup = {'base': self.local_base, 'local:depth': self.local_control, 'legacy-control.ckpt': self.local_control}
        self.ns['checkpoint_lookup'] = NS(get=lambda key: lookup.get(key, key))
        future = NS(add_response_callback=lambda cb: None, add_done_callback=lambda cb: None,
                    add_exception_callback=lambda cb: None)

        def control_net(**kwargs):
            self.calls.append(kwargs)
            return future

        self.ns['Generator'] = NS(shared=lambda: NS(control_net=control_net))
        self.backend = NS(optimizations=lambda: NS(), use_sdxl_refiner=False)

    def arguments(self, task, control_id):
        return NS(model=NS(id='base'), scheduler='DDIM', prompt=NS(positive='brick', negative=None),
                  steps=20, size=(512, 512), seed=1, guidance_scale=7.5, seamless_axes=None,
                  iterations=1, step_preview_mode=None, task=task,
                  control_nets=[NS(model=control_id, image='depth image', strength=1.0)])

    def test_all_controlnet_branches_resolve_local_and_legacy_checkpoint_ids(self):
        tasks = [self.ns['PromptToImage'](), self.ns['ImageToImage']('image', 0.5, False),
                 self.ns['Inpaint']('image', 0.5, False, self.ns['Inpaint'].MaskSource.ALPHA, '', 1.0)]
        for task in tasks:
            for identifier in ['local:depth', 'legacy-control.ckpt']:
                with self.subTest(task=type(task).__name__, identifier=identifier):
                    self.ns['generate'](self.backend, self.arguments(task, identifier), lambda *args: True, lambda *args: None)
                    call = self.calls[-1]
                    self.assertIs(call['model'], self.local_base)
                    self.assertIs(call['control_net'][0], self.local_control)
                    self.assertEqual(call['control'], ['depth image'])
                    self.assertEqual(call['controlnet_conditioning_scale'], [1.0])
                    self.assertEqual(call['inpaint'], isinstance(task, self.ns['Inpaint']))

    def test_hub_controlnet_identifiers_are_unchanged(self):
        args = self.arguments(self.ns['ImageToImage']('image', 0.75, True), 'lllyasviel/control_v11f1p_sd15_depth')
        self.ns['generate'](self.backend, args, lambda *args: True, lambda *args: None)
        self.assertEqual(self.calls[-1]['control_net'], ['lllyasviel/control_v11f1p_sd15_depth'])

    def test_selector_labels_are_independent_from_stable_ids(self):
        rows = [
            NS(model_base='base--version.safetensors', model_label='Folder A/base--version.safetensors',
               model_identifier='local:base-one', model_type='PROMPT_TO_IMAGE'),
            NS(model_base='depth--version.safetensors', model_label='Folder B/depth--version.safetensors',
               model_identifier='local:depth-one', model_type='CONTROL_NET'),
            NS(model_base='models--owner--repo', model_label='', model_identifier='', model_type='PROMPT_TO_IMAGE'),
        ]
        context = NS(preferences=NS(addons={'test_addon': NS(preferences=NS(installed_models=rows))}))
        base = [model for model in self.ns['list_models'](self.backend, context) if model is not None]
        control = self.ns['list_controlnet_models'](self.backend, context)
        self.assertEqual({m.id for m in base}, {'local:base-one', 'owner/repo'})
        self.assertIn('Folder A/base--version.safetensors', {m.name for m in base})
        self.assertEqual([(m.id, m.name) for m in control],
                         [('local:depth-one', 'Folder B/depth--version.safetensors')])

    def test_model_alias_resolves_to_explicit_entry_and_unknown_id_is_retained(self):
        linked = NS(identifier='name/one.safetensors')
        models = {'local:previous-scan': linked, 'name--one.safetensors': linked,
                  'owner/repo': NS(id='/cache/models--owner--repo')}
        self.ns['model_lookup'] = NS(get=models.get)
        for alias in ['local:previous-scan', 'name--one.safetensors']:
            self.assertEqual(self.ns['resolve_model_identifier'](self.backend, alias), 'name/one.safetensors')
        for unchanged in ['owner/repo', 'local:removed-model']:
            self.assertEqual(self.ns['resolve_model_identifier'](self.backend, unchanged), unchanged)

    def test_refiner_enum_callbacks_preserve_selection_through_reorder_and_link(self):
        spec = importlib.util.spec_from_file_location('dispatch_model_selection', ROOT / 'model_selection.py')
        selection_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = selection_module
        spec.loader.exec_module(selection_module)
        tree = ast.parse((ROOT / 'diffusers_backend.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DiffusersBackend')
        prop = next(node for node in cls.body if isinstance(node, ast.AnnAssign)
                    and node.target.id == 'sdxl_refiner_model')
        key = '_dream_textures_refiner_identifier'
        context = object()
        callbacks = eval(compile(ast.Expression(prop.annotation), 'diffusers_backend.py', 'eval'), {
            'EnumProperty': lambda **kwargs: kwargs,
            '_refiner_selection': selection_module.ModelSelection(key),
            'bpy': NS(context=context),
        })
        models = [None, NS(id='owner/base', name='Base', description=''),
                  NS(id='local:selected', name='Refiner', description='')]
        aliases = {}

        class Owner(dict):
            def list_models(self, context):
                return models

            def resolve_model_identifier(self, identifier):
                return aliases.get(identifier, identifier)

        owner = Owner()
        number = lambda identifier: next(row[-1] for row in callbacks['items'](owner, context)
                                         if row is not None and row[0] == identifier)
        value = callbacks['set_transform'](owner, number('local:selected'), 0, False)
        self.assertEqual(owner[key], 'local:selected')
        models.insert(1, NS(id='owner/another', name='Another', description=''))
        self.assertEqual(callbacks['get_transform'](owner, value, True), number('local:selected'))
        models[-1] = NS(id='explicit-refiner.safetensors', name='Refiner', description='')
        aliases['local:selected'] = 'explicit-refiner.safetensors'
        self.assertEqual(callbacks['get_transform'](owner, value, True), number('explicit-refiner.safetensors'))
        self.assertEqual(owner[key], 'local:selected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
