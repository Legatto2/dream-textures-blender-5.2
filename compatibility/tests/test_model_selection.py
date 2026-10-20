"""Model selection identity tests without Blender or the AI runtime.

Exercises the real pure helper and the real property callback definitions with
small simulated RNA owners. Live EnumProperty acceptance is a separate script.
Run: python3 -B compatibility/tests/test_model_selection.py
"""

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("selection_under_test", ROOT / "model_selection.py")
selection = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = selection
spec.loader.exec_module(selection)


def model(identifier, name=None):
    return NS(id=identifier, name=name or identifier, description="Test checkpoint")


def value_for(rows, identifier):
    return next(item[3] for item in rows if item is not None and item[0] == identifier)


def id_for(rows, number):
    return next(item[0] for item in rows if item is not None and item[3] == number)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.helper = selection.ModelSelection("selected_identifier")
        self.owner = {}
        self.a, self.b = model("owner/a"), model("local:" + "b" * 64)
        self.models = [None, self.a, self.b]

    def choose(self, identifier):
        number = value_for(self.helper.items(self.owner, self.models), identifier)
        return self.helper.set(self.owner, number, 0, False, self.models)

    def selected(self, current_value=None, is_set=False):
        number = self.helper.get(self.owner, current_value, is_set, self.models)
        return id_for(self.helper.items(self.owner, self.models), number)

    def test_add_remove_and_reorder_keep_full_selected_identifier(self):
        self.choose(self.b.id)
        self.models[:] = [model("local:" + "a" * 64), self.b, self.a]
        self.assertEqual(self.selected(), self.b.id)
        self.models.remove(self.a)
        self.assertEqual(self.selected(), self.b.id)
        self.assertEqual(self.owner["selected_identifier"], self.b.id)

    def test_missing_selection_is_unavailable_and_returns_when_model_returns(self):
        old_number = self.choose(self.b.id)
        self.models[:] = [self.a]
        self.assertEqual(self.selected(), self.b.id)
        row = next(item for item in self.helper.items(self.owner, self.models) if item is not None and item[0] == self.b.id)
        self.assertEqual(row[1], "Unavailable model")
        self.assertEqual(row[3], old_number)
        self.models.insert(0, self.b)
        self.assertEqual(self.selected(), self.b.id)
        self.assertEqual(value_for(self.helper.items(self.owner, self.models), self.b.id), old_number)

    def test_serialized_full_string_works_without_any_stored_numeric_value(self):
        self.choose(self.b.id)
        restored = json.loads(json.dumps(self.owner))
        fresh = selection.ModelSelection("selected_identifier")
        rows = fresh.items(restored, [self.b, self.a])
        number = fresh.get(restored, None, False, [self.b, self.a])
        self.assertEqual(id_for(rows, number), self.b.id)
        self.assertEqual(restored, {"selected_identifier": self.b.id})

    def test_registry_never_reuses_removed_values_or_truncates_identifiers(self):
        first = model("local:" + "0" * 63 + "1")
        second = model("local:" + "0" * 63 + "2")
        first_number = value_for(self.helper.items(self.owner, [first]), first.id)
        second_number = value_for(self.helper.items(self.owner, [second]), second.id)
        self.assertNotEqual(first_number, second_number)
        # A model disappears while its old menu is open. Its event still means
        # that exact model and becomes unavailable, never the replacement row.
        self.helper.set(self.owner, first_number, None, False, [second])
        number = self.helper.get(self.owner, second_number, True, [second])
        self.assertEqual(id_for(self.helper.items(self.owner, [second]), number), first.id)

    def test_legacy_numeric_positions_exclude_auto_only_groups(self):
        legacy_a, legacy_b = model("owner/depth"), model("owner/prompt")
        models = [None, model("local:auto-group"), None, legacy_a,
                  None, model("local:inside-group"), legacy_b]
        self.assertEqual(selection.legacy_models(models), [None, legacy_a, None, legacy_b])
        result = self.helper.get({}, 3, True, models)
        self.assertEqual(id_for(self.helper.items({}, models), result), legacy_b.id)
        self.assertEqual(selection.legacy_models([legacy_a, None, None, legacy_b]),
                         [legacy_a, None, None, legacy_b])

    def test_legacy_control_default_and_missing_numeric_do_not_choose_auto_model(self):
        models = [self.b, self.a]
        default = self.helper.get({}, None, False, models)
        self.assertEqual(id_for(self.helper.items({}, models), default), self.a.id)
        for raw in (None, -1, 99):
            result = self.helper.get({}, raw, True, models)
            self.assertEqual(id_for(self.helper.items({}, models), result), selection.UNSELECTED)
        self.assertEqual(self.helper.get({}, 0, True, [self.b]), 0)

    def test_getter_is_read_only_and_explicit_unselection_does_not_fallback(self):
        class ReadOnly(dict):
            def __setitem__(self, key, value):
                raise AssertionError("an enum getter must not mutate RNA")
        owner = ReadOnly({"selected_identifier": self.b.id})
        self.helper.get(owner, 0, True, self.models)
        self.helper.get(ReadOnly(), 0, True, [self.a])
        self.helper.set(self.owner, 0, 0, False, [self.a])
        self.assertEqual(self.helper.get(self.owner, 0, True, [self.a]), 0)

    def test_alias_resolution_retains_canonical_storage_and_explicit_row(self):
        self.choose(self.b.id)
        linked = model("manual--name.safetensors")
        resolve = lambda identifier: linked.id if identifier == self.b.id else identifier
        number = self.helper.get(self.owner, None, False, [linked], resolve)
        rows = self.helper.items(self.owner, [linked], resolve)
        self.assertEqual(id_for(rows, number), linked.id)
        self.assertEqual(self.owner["selected_identifier"], self.b.id)
        self.assertEqual([row[0] for row in rows if row is not None], [linked.id, selection.UNSELECTED])

    def test_invalid_numeric_set_preserves_previous_selection(self):
        self.choose(self.b.id)
        before = self.owner.copy()
        with self.assertRaisesRegex(ValueError, "no longer available"):
            self.helper.set(self.owner, 1234567, 0, True, self.models)
        self.assertEqual(self.owner, before)

    def test_dynamic_enum_strings_remain_referenced_across_callbacks(self):
        first = self.helper.items(self.owner, [model("owner/" + "a" * 64, "A model")])[0]
        second = self.helper.items(self.owner, [model("owner/" + "a" * 64, "A model")])[0]
        for index in range(3):
            self.assertIs(first[index], second[index])


def callback_namespace(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return tree


class CallbackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.models = [None, model("owner/base"), model("local:canonical")]
        self.aliases = {}
        backend = NS(list_models=lambda context: self.models,
                     list_controlnet_models=lambda context: [m for m in self.models if m is not None],
                     resolve_model_identifier=lambda identifier: self.aliases.get(identifier, identifier))
        class Owner(dict):
            def get_backend(self):
                return backend
        self.Owner = Owner
        self.backend = backend
        self.context = NS(scene=NS(dream_textures_prompt=Owner()))
        self.ns = {"bpy": NS(context=self.context), "EnumProperty": lambda **kwargs: kwargs,
                   "_model_selection": selection.ModelSelection("_dream_textures_model_identifier"),
                   "_control_net_selection": selection.ModelSelection("_dream_textures_control_net_identifier")}
        base_tree = callback_namespace(ROOT / "property_groups/dream_prompt.py", {
            '_model_selection_data', 'model_options', '_model_get', '_model_set'
        }, self.ns)
        control_tree = callback_namespace(ROOT / "property_groups/control_net.py", {
            '_control_net_selection_data', 'control_net_options', '_control_net_get', '_control_net_set'
        }, self.ns)
        attributes = next(node.value for node in base_tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'attributes' for target in node.targets))
        base_call = next(value for key, value in zip(attributes.keys, attributes.values) if key.value == 'model')
        self.base = eval(compile(ast.Expression(base_call), "dream_prompt.py", "eval"), self.ns)
        self.control_tree = control_tree
        self.control = self.annotation('ControlNet')

    def annotation(self, class_name):
        cls = next(node for node in self.control_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        call = next(node.annotation for node in cls.body if isinstance(node, ast.AnnAssign) and node.target.id == 'control_net')
        return eval(compile(ast.Expression(call), "control_net.py", "eval"), self.ns)

    def test_base_and_control_property_callbacks_save_restore_and_resolve_aliases(self):
        for descriptor, key in ((self.base, '_dream_textures_model_identifier'),
                                (self.control, '_dream_textures_control_net_identifier')):
            with self.subTest(property=key):
                self.models[:] = [None, model('owner/base'), model('local:canonical')]
                self.aliases.clear()
                owner = self.Owner()
                number = value_for(descriptor['items'](owner, self.context), 'local:canonical')
                descriptor['set_transform'](owner, number, 0, False)
                self.assertEqual(owner[key], 'local:canonical')
                self.models.insert(0, model('local:earlier'))
                value = descriptor['get_transform'](owner, None, False)
                self.assertEqual(id_for(descriptor['items'](owner, None), value), 'local:canonical')
                self.models[:] = [model('manual.safetensors')]
                self.aliases['local:canonical'] = 'manual.safetensors'
                value = descriptor['get_transform'](owner, number, True)
                self.assertEqual(id_for(descriptor['items'](owner, self.context), value), 'manual.safetensors')
                self.assertEqual(owner[key], 'local:canonical')

    def test_transient_add_operator_uses_available_only_items_without_custom_storage(self):
        descriptor = self.annotation('ControlNetsAdd')
        self.assertNotIn('get_transform', descriptor)
        self.assertNotIn('set_transform', descriptor)
        rows = descriptor['items'](object(), self.context)
        self.assertEqual([row[0] for row in rows], ['owner/base', 'local:canonical'])

    def test_transient_add_rejects_missing_selection_before_creating_a_net(self):
        cls = next(node for node in self.control_tree.body if isinstance(node, ast.ClassDef) and node.name == 'ControlNetsAdd')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
        exec(compile(ast.Module(body=[method], type_ignores=[]), "control_net.py", "exec"), self.ns)
        add = Mock()
        self.context.scene.dream_textures_prompt.control_nets = NS(add=add)
        operator = NS(control_net='', report=Mock())
        self.assertEqual(self.ns['execute'](operator, self.context), {'CANCELLED'})
        add.assert_not_called()

    def test_generation_matches_resolved_base_and_control_ids_without_retargeting_missing(self):
        args_api = NS(GenerationArguments=lambda **kwargs: NS(**kwargs), PromptToImage=NS,
                      Prompt=lambda *args: args,
                      models=NS(control_net=NS(ControlNet=lambda identifier, image, strength: NS(model=identifier))))
        ns = {"api": args_api, "file_batch_structure": NS(id='file-batch'),
              "SeamlessAxes": lambda value: value, "StepPreviewMode": lambda value: value}
        callback_namespace(ROOT / "property_groups/dream_prompt.py", {'generate_args'}, ns)
        self.backend.get_batch_size = lambda context: 1
        self.models[:] = [model('manual-base.safetensors')]
        self.aliases.update({'local:base': 'manual-base.safetensors', 'local:control': 'manual-control.safetensors'})
        prompt = NS(get_backend=lambda: self.backend, prompt_structure='plain', iterations=1,
                    use_init_img=False, model='local:base', generate_prompt=lambda: 'brick',
                    use_negative_prompt=False, use_size=False, get_seed=lambda: 42, steps=20,
                    cfg_scale=7.5, scheduler='test', seamless_axes='off', step_preview_mode='none',
                    control_nets=[NS(control_net='local:control', conditioning_scale=1, enabled=True)])
        result = ns['generate_args'](prompt, self.context)
        self.assertIs(result.model, self.models[0])
        self.assertEqual(result.control_nets[0].model, 'manual-control.safetensors')
        prompt.model = 'local:missing'
        self.assertIsNone(ns['generate_args'](prompt, self.context).model)


if __name__ == '__main__':
    unittest.main(verbosity=2)
