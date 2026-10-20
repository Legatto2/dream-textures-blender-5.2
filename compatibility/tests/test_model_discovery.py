"""Local model discovery checks; no Blender, AI dependencies, models or network.

Fixtures contain realistic tensor metadata and sparse payloads. They are never
loaded as model weights. Preferences tests execute the actual integration
functions against a main-thread-only RNA stand-in.
"""
import ast
from dataclasses import replace
import enum
import importlib.util
import json
import math
import ntpath
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("discovery_under_test", ROOT / "model_scanner.py")
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)


def sd_shapes(family="sd1", inputs=4):
    channels = 384 if family == "refiner" else 320
    shapes = {
        "model.diffusion_model.input_blocks.0.0.weight": [channels, inputs, 3, 3],
        "model.diffusion_model.out.2.weight": [4, channels, 3, 3],
        "first_stage_model.encoder.conv_in.weight": [128, 3, 3, 3],
        "first_stage_model.decoder.conv_out.weight": [3, 128, 3, 3],
    }
    if family == "sd1":
        shapes["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = [49408, 768]
    elif family == "sd2":
        shapes["cond_stage_model.model.token_embedding.weight"] = [49408, 1024]
    elif family == "xl":
        shapes["conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"] = [49408, 768]
        shapes["conditioner.embedders.1.model.token_embedding.weight"] = [49408, 1280]
    elif family == "refiner":
        shapes["conditioner.embedders.0.model.token_embedding.weight"] = [49408, 1280]
    return shapes


def control_shapes(width=768):
    return {
        "control_model.input_blocks.0.0.weight": [320, 4, 3, 3],
        "control_model.input_hint_block.0.weight": [16, 3, 3, 3],
        "control_model.zero_convs.0.0.weight": [320, 320, 1, 1],
        "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [320, width],
    }


def tensor_header(shapes, metadata=None):
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = math.prod(shape) * 2
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    return header, offset


def write_header(path, header, payload=0):
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(data).to_bytes(8, "little"))
        handle.write(data)
        handle.truncate(8 + len(data) + payload)


def write_model(path, shapes=None, metadata=None):
    header, payload = tensor_header(sd_shapes() if shapes is None else shapes, metadata)
    write_header(path, header, payload)
    return path


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_supported_architectures_use_explicit_configs(self):
        fixtures = [
            ("sd1", sd_shapes(), None, "STABLE_DIFFUSION_1"),
            ("sd2-eps", sd_shapes("sd2"), {"modelspec.prediction_type": "epsilon"}, "STABLE_DIFFUSION_2_BASE"),
            ("sd2-v", sd_shapes("sd2"), {"modelspec.prediction_type": "v"}, "STABLE_DIFFUSION_2"),
            ("depth", sd_shapes("sd2", 5), None, "STABLE_DIFFUSION_2_DEPTH"),
            ("inpaint", sd_shapes("sd2", 9), None, "STABLE_DIFFUSION_2_INPAINTING"),
            ("xl", sd_shapes("xl"), None, "STABLE_DIFFUSION_XL_BASE"),
            ("refiner", sd_shapes("refiner"), None, "STABLE_DIFFUSION_XL_REFINER"),
            ("control1", control_shapes(), None, "CONTROL_NET_1_5"),
            ("control2", control_shapes(1024), None, "CONTROL_NET_2_1"),
        ]
        for name, shapes, metadata, expected in fixtures:
            with self.subTest(name=name):
                folder = self.root / name
                write_model(folder / (name + ".safetensors"), shapes, metadata)
                result = scanner.scan_model_folders([str(folder)])
                self.assertFalse(result.limited)
                self.assertEqual([m.classification.config for m in result.models], [expected])

    def test_malformed_file_does_not_hide_valid_neighbors(self):
        key = "model.diffusion_model.input_blocks.0.0.weight"
        for index, header in enumerate([[], {key: 4}, {key: {"shape": None}}, {key: {}}, {"x": {"dtype": []}}]):
            write_header(self.root / f"bad{index}.safetensors", header)
        write_model(self.root / "good.safetensors")
        result = scanner.scan_model_folders([str(self.root)])
        self.assertEqual([m.display for m in result.models], ["good.safetensors"])
        self.assertEqual(result.skipped, 5)

    def test_truncation_oversized_headers_and_overlapping_tensors_are_skipped(self):
        (self.root / "prefix.safetensors").write_bytes(b"\x10\x00")
        (self.root / "truncated.safetensors").write_bytes((1000).to_bytes(8, "little") + b"{}")
        with (self.root / "oversized.safetensors").open("wb") as handle:
            length = scanner.ScanLimits().header_bytes + 1
            handle.write(length.to_bytes(8, "little"))
            handle.truncate(length + 8)
        header, payload = tensor_header(sd_shapes())
        header["model.diffusion_model.out.2.weight"]["data_offsets"] = header["model.diffusion_model.input_blocks.0.0.weight"]["data_offsets"]
        write_header(self.root / "overlap.safetensors", header, payload)
        result = scanner.scan_model_folders([str(self.root)])
        self.assertEqual(len(result.models), 0)
        self.assertEqual(result.skipped, 4)

    def test_components_modern_models_and_ambiguous_sd2_are_not_models(self):
        examples = [
            {"cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": [49408, 768]},
            {"encoder.conv_in.weight": [128, 3, 3, 3]},
            {"lora_unet_down_blocks.0.lora_down.weight": [4, 320]},
            {"double_blocks.0.img_attn.qkv.weight": [12, 12]},
            sd_shapes("sd2"), sd_shapes(inputs=11), control_shapes(2048),
        ]
        for index, shapes in enumerate(examples):
            write_model(self.root / f"unknown{index}.safetensors", shapes)
        (self.root / "untrusted.ckpt").write_bytes(b"never deserialize this")
        (self.root / "unrelated.pth").write_bytes(b"never deserialize this either")
        result = scanner.scan_model_folders([str(self.root)])
        self.assertEqual(result.models, ())
        self.assertEqual(result.skipped, len(examples))

    def test_identity_survives_duplicate_roots_and_label_collisions(self):
        a, b = self.root / "a" / "models", self.root / "b" / "models"
        write_model(a / "same--name.safetensors")
        write_model(b / "same--name.safetensors")
        result = scanner.scan_model_folders([str(a), str(a / "."), str(b)])
        self.assertEqual(len(result.models), 2)
        self.assertEqual(len({m.display for m in result.models}), 2)
        self.assertEqual(len({m.identifier for m in result.models}), 2)
        self.assertTrue(all("same--name" in m.display for m in result.models))
        reverse = scanner.scan_model_folders([str(b), str(a)])
        self.assertEqual({m.identifier for m in reverse.models}, {m.identifier for m in result.models})
        self.assertEqual(scanner.normalize_path(r"C:\MODELS\..\Models\a.safetensors", ntpath),
                         scanner.normalize_path(r"c:\models\a.safetensors", ntpath))

    def test_explicit_link_config_and_basename_take_precedence(self):
        model = write_model(self.root / "choice.safetensors")
        manual = self.root / "manual.ckpt"
        manual.write_bytes(b"not read")
        result = scanner.scan_model_folders([str(self.root)], [
            (str(self.root), "AUTO_DETECT"),
            (str(model), "STABLE_DIFFUSION_2"),
        ])
        self.assertEqual(result.models, ())
        self.assertEqual({Path(m.path).name: m.config for m in result.linked},
                         {"choice.safetensors": "STABLE_DIFFUSION_2", "manual.ckpt": "AUTO_DETECT"})

    def test_cached_headers_are_not_reread_and_deletions_disappear(self):
        model = write_model(self.root / "cached.safetensors")
        cache = {}
        with patch.object(scanner, "_read_safetensors_header", wraps=scanner._read_safetensors_header) as read:
            first = scanner.scan_model_folders([str(self.root)], cache=cache)
            second = scanner.scan_model_folders([str(self.root)], cache=cache)
            self.assertEqual(first.models, second.models)
            self.assertEqual(read.call_count, 1)
            model.unlink()
            self.assertEqual(scanner.scan_model_folders([str(self.root)], cache=cache).models, ())
            self.assertEqual(cache, {})

    def test_scan_limits_and_pre_cancelled_work(self):
        for index in range(3):
            write_model(self.root / f"m{index}.safetensors")
        result = scanner.scan_model_folders([str(self.root)], limits=replace(scanner.ScanLimits(), entries=1))
        self.assertTrue(result.limited)
        self.assertTrue(result.linked_complete)
        self.assertLessEqual(len(result.models), 1)
        cancel = threading.Event()
        cancel.set()
        result = scanner.scan_model_folders([str(self.root)], cancel=cancel)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.models, ())

    def test_directory_symlink_cycle_is_not_followed(self):
        write_model(self.root / "good.safetensors")
        try:
            (self.root / "cycle").symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks unavailable")
        result = scanner.scan_model_folders([str(self.root)])
        self.assertFalse(result.limited)
        self.assertEqual(len(result.models), 1)


class WorkerTests(unittest.TestCase):
    def test_stale_jobs_and_cancelled_jobs_cannot_publish(self):
        started, release = threading.Event(), threading.Event()
        calls = []

        def work(folders, linked, *, cancel, cache):
            self.assertIsNot(threading.current_thread(), threading.main_thread())
            calls.append((folders, cancel))
            if folders == ("old",):
                started.set()
                self.assertTrue(release.wait(2))
            return scanner.ScanResult(error=folders[0])

        service = scanner.ModelScanService(work)
        self.addCleanup(service.close)
        old = service.request(["old"])
        self.assertTrue(started.wait(2))
        new = service.request(["new"])
        self.assertTrue(calls[0][1].is_set())
        release.set()
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = service.take_result(new)
            time.sleep(0.005)
        self.assertIsNotNone(result)
        self.assertEqual(result.error, "new")
        self.assertIsNone(service.take_result(old))
        cancelled = service.request(["cancelled"])
        service.cancel()
        self.assertIsNone(service.take_result(cancelled))
        service.close()
        with self.assertRaises(RuntimeError):
            service.request(["closed"])


def integration_namespace():
    source = ast.parse((ROOT / "preferences.py").read_text())
    functions = {
        '_preferences_for_callback', '_replace_model_rows', 'set_model_list',
        '_legacy_model_identifier', '_build_model_catalog', '_cancel_model_refresh',
        'shutdown_model_discovery', 'fetch_installed_models',
    }
    nodes = [node for node in source.body if (
        isinstance(node, ast.FunctionDef) and node.name in functions
        or isinstance(node, ast.ClassDef) and node.name in {'checkpoint_lookup', 'model_lookup'}
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_MODEL_FIELDS' for t in node.targets)
    )]
    model_ns = {'enum': enum, 'absolute_path': lambda p: p, 'Task': object}
    for name, filename in [('ModelConfig', 'model_config.py'), ('ModelType', 'model_type.py')]:
        tree = ast.parse((ROOT / 'generator_process/models' / filename).read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), model_ns)
    ns = dict(model_ns, os=os, __package__='discovery_test')
    ns.update({
        'HubModel': lambda id, author, tags, likes, downloads, model_type: NS(id=id, likes=likes, downloads=downloads, model_type=model_type),
        'Checkpoint': lambda path, config: NS(path=path, config=config),
        '_preferences_generation': object(), '_model_scan_service': None,
        '_model_refresh_timer': None, '_model_refresh_token': None,
        '_published_model_scan': scanner.ScanResult(), 'ScanResult': scanner.ScanResult,
        'current_runtime_status': lambda: {'ready': True},
        'model_identifier': scanner.model_identifier,
        'reconcile_scan_result': scanner.reconcile_scan_result,
    })
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'preferences.py', 'exec'), ns)
    return ns


class FakeCollection(list):
    def clear(self):
        assert threading.current_thread() is threading.main_thread()
        super().clear()

    def add(self):
        assert threading.current_thread() is threading.main_thread()

        class Item(NS):
            def __setattr__(self, key, value):
                if key == 'model' and value == 'fail-assignment':
                    raise RuntimeError('simulated RNA assignment failure')
                super().__setattr__(key, value)

        item = Item(model='', model_base='', model_identifier='', model_label='', model_is_local=False,
                    downloads=0, likes=0, model_type='UNKNOWN')
        self.append(item)
        return item


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self.ns = integration_namespace()
        self.pref = NS(model_folders=[], linked_checkpoints=[], installed_models=FakeCollection(),
                       model_scan_running=False, model_scan_status='')
        callbacks = set()
        self.timers = NS(register=lambda cb, **kw: callbacks.add(cb), unregister=lambda cb: callbacks.discard(cb),
                         is_registered=lambda cb: cb in callbacks)
        self.ns['bpy'] = NS(app=NS(timers=self.timers), context=NS(preferences=NS(addons={'discovery_test': NS(preferences=self.pref)})))
        self.future = NS(done=True, result=lambda: [])
        self.ns['Generator'] = NS(shared=lambda: NS(hf_list_installed_models=lambda: self.future))
        self.service = NS(request=lambda *args: object(), cancel=lambda: None, close=lambda: None,
                          take_result=lambda token: scanner.ScanResult())
        self.ns['ModelScanService'] = lambda: self.service

    def refresh(self, scan, hub_models=()):
        self.service.take_result = lambda token: scan
        self.future.result = lambda: hub_models
        self.ns['fetch_installed_models']()
        self.ns['_model_refresh_timer']()

    def hub_model(self, identifier):
        return NS(id='/cache/models--' + identifier.replace('/', '--'), downloads=0, likes=0,
                  model_type=self.ns['ModelType'].PROMPT_TO_IMAGE)

    def test_refresh_is_nonblocking_and_stale_timer_is_ignored(self):
        self.future.done = False
        self.ns['fetch_installed_models']()
        old = self.ns['_model_refresh_timer']
        self.assertEqual(old(), 0.1)
        self.ns['fetch_installed_models']()
        current = self.ns['_model_refresh_timer']
        self.assertIsNone(old())
        self.assertIs(self.ns['_model_refresh_timer'], current)
        self.assertTrue(self.pref.model_scan_running)
        self.future.done = True
        self.assertIsNone(current())
        self.assertFalse(self.pref.model_scan_running)

    def test_shutdown_invalidates_callbacks_before_reload(self):
        self.ns['_published_model_scan'] = scanner.ScanResult(linked_settings=(('/old/link', 'config'),))
        self.ns['fetch_installed_models']()
        old = self.ns['_model_refresh_timer']
        self.ns['shutdown_model_discovery']()
        self.ns['_preferences_generation'] = object()
        self.pref.model_scan_status = 'new generation'
        self.assertIsNone(old())
        self.assertEqual(self.pref.model_scan_status, 'new generation')
        self.assertIsNone(self.ns['_model_scan_service'])
        self.assertEqual(self.ns['_published_model_scan'], scanner.ScanResult())

    def test_catalog_keeps_explicit_ids_and_separates_folder_labels(self):
        scan = scanner.ScanResult(
            models=(scanner.ScannedModel('/models/name--one.safetensors', '/models', 'name--one.safetensors',
                                        '/models/name--one.safetensors', 'local:canonical',
                                        scanner.ModelClassification('CONTROL_NET_1_5', 'CONTROL_NET')),),
            linked=(scanner.LinkedCheckpoint('/other/manual.ckpt', 'STABLE_DIFFUSION_2', '/other/manual.ckpt'),),
        )
        self.service.take_result = lambda token: scan
        self.ns['fetch_installed_models']()
        self.ns['_model_refresh_timer']()
        links = self.ns['checkpoint_lookup']._checkpoints
        self.assertEqual(links['manual.ckpt'].config.name, 'STABLE_DIFFUSION_2')
        self.assertEqual(links['local:canonical'].path, '/models/name--one.safetensors')
        self.assertEqual(self.pref.installed_models[-1].model_label, 'name--one.safetensors')
        self.assertEqual(self.pref.installed_models[-1].model_identifier, 'local:canonical')

    def test_explicit_link_preserves_prior_folder_selection_without_duplicate_row(self):
        path = '/models/name--one.safetensors'
        identity = scanner.normalize_path(path)
        prior_identifier = scanner.model_identifier(identity)
        scan = scanner.ScanResult(linked=(
            scanner.LinkedCheckpoint(path, 'STABLE_DIFFUSION_2_BASE', identity),
        ))
        rows, checkpoints, models = self.ns['_build_model_catalog']([], scan)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].identifier, 'name/one.safetensors')
        checkpoint = checkpoints['name/one.safetensors']
        self.assertIs(checkpoints['name--one.safetensors'], checkpoint)
        self.assertIs(checkpoints[prior_identifier], checkpoint)
        self.assertEqual(checkpoint.config.name, 'STABLE_DIFFUSION_2_BASE')
        self.assertIs(models[prior_identifier], rows[0])

    def test_capped_startup_scan_publishes_downloads_conversions_and_complete_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = root / 'folders'
            write_model(folders / 'discovered.safetensors')
            manual = root / 'manual.ckpt'
            manual.write_bytes(b'not deserialized')
            links = [(str(manual), 'STABLE_DIFFUSION_1')]
            self.pref.model_folders = [NS(path=str(folders))]
            self.pref.linked_checkpoints = [NS(path=path, model_config=config) for path, config in links]
            scan = scanner.scan_model_folders([str(folders)], links,
                                             limits=replace(scanner.ScanLimits(), entries=1))
            self.assertTrue(scan.limited)
            self.assertTrue(scan.linked_complete)
            downloaded, converted = self.hub_model('owner/new-download'), self.hub_model('local/converted')
            self.refresh(scan, [downloaded, converted])
            self.assertEqual({row.model for row in self.pref.installed_models},
                             {downloaded.id, converted.id, str(manual)})
            self.assertIs(self.ns['model_lookup'].get('owner/new-download'), downloaded)
            self.assertIs(self.ns['model_lookup'].get('local/converted'), converted)
            self.assertEqual(self.ns['checkpoint_lookup'].get('manual.ckpt').config.name, 'STABLE_DIFFUSION_1')
            self.assertIn('Hub models and checkpoint links refreshed', self.pref.model_scan_status)
            self.assertNotIn('failed', self.pref.model_scan_status)

    def test_capped_folder_refresh_keeps_registered_rows_and_honors_fresh_explicit_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep, removed = root / 'keep', root / 'removed'
            chosen = write_model(keep / 'chosen.safetensors')
            retained = write_model(keep / 'retained.safetensors')
            forgotten = write_model(removed / 'forgotten.safetensors')
            self.pref.model_folders = [NS(path=str(keep)), NS(path=str(removed))]
            self.refresh(scanner.scan_model_folders([str(keep), str(removed)]), [self.hub_model('owner/old')])
            identifiers = {row.path: row.identifier for row in self.ns['_published_model_scan'].models}
            manual = root / 'new-link.ckpt'
            manual.write_bytes(b'not deserialized')
            links = [(str(chosen), 'STABLE_DIFFUSION_2_BASE'), (str(manual), 'STABLE_DIFFUSION_1')]
            self.pref.model_folders = [NS(path=str(keep))]
            self.pref.linked_checkpoints = [NS(path=path, model_config=config) for path, config in links]
            scan = scanner.scan_model_folders([str(keep)], links,
                                             limits=replace(scanner.ScanLimits(), entries=2))
            self.assertTrue(scan.limited)
            self.assertTrue(scan.linked_complete)
            self.refresh(scan, [self.hub_model('owner/new')])
            rows = [row.model for row in self.pref.installed_models]
            self.assertIn(str(retained), rows)
            self.assertNotIn(str(forgotten), rows)
            self.assertEqual(rows.count(str(chosen)), 1)
            self.assertIn(str(manual), rows)
            self.assertIsNone(self.ns['model_lookup'].get('owner/old'))
            self.assertIsNotNone(self.ns['model_lookup'].get('owner/new'))
            checkpoints = self.ns['checkpoint_lookup']._checkpoints
            self.assertEqual(checkpoints[identifiers[str(chosen)]].config.name, 'STABLE_DIFFUSION_2_BASE')
            self.assertIs(checkpoints[identifiers[str(chosen)]], checkpoints['chosen.safetensors'])
            self.assertNotIn(identifiers[str(forgotten)], checkpoints)

    def test_capped_manual_stage_keeps_unchanged_complete_links_instead_of_partial_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen = write_model(root / 'chosen.safetensors')
            links = [(str(root), 'STABLE_DIFFUSION_1'), (str(chosen), 'STABLE_DIFFUSION_2_BASE')]
            self.pref.linked_checkpoints = [NS(path=path, model_config=config) for path, config in links]
            self.refresh(scanner.scan_model_folders([], links))
            scan = scanner.scan_model_folders([], links, limits=replace(scanner.ScanLimits(), entries=2))
            self.assertTrue(scan.limited)
            self.assertFalse(scan.linked_complete)
            self.assertEqual(scan.linked[0].config, 'STABLE_DIFFUSION_1')
            self.refresh(scan, [self.hub_model('owner/new')])
            self.assertIsNotNone(self.ns['model_lookup'].get('owner/new'))
            self.assertEqual(self.ns['checkpoint_lookup'].get('chosen.safetensors').config.name,
                             'STABLE_DIFFUSION_2_BASE')
            self.assertIn('unchanged link settings', self.pref.model_scan_status)

    def test_capped_manual_stage_does_not_restore_removed_links_or_changed_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen = write_model(root / 'linked' / 'chosen.safetensors')
            discovered = write_model(root / 'discovered' / 'model.safetensors')
            links = [(str(chosen.parent), 'STABLE_DIFFUSION_1'), (str(chosen), 'STABLE_DIFFUSION_2_BASE')]
            self.pref.model_folders = [NS(path=str(discovered.parent))]
            original = scanner.scan_model_folders([str(discovered.parent)], links)
            for changed in [links[:1], [links[0], (str(chosen), 'STABLE_DIFFUSION_2')]]:
                with self.subTest(links=changed):
                    self.refresh(original)
                    self.pref.linked_checkpoints = [NS(path=path, model_config=config) for path, config in changed]
                    scan = scanner.scan_model_folders([str(discovered.parent)], changed,
                                                     limits=replace(scanner.ScanLimits(), entries=1))
                    self.assertTrue(scan.limited)
                    self.assertFalse(scan.linked_complete)
                    fresh = self.hub_model('owner/new')
                    self.refresh(scan, [fresh])
                    self.assertEqual([row.model for row in self.pref.installed_models], [fresh.id])
                    self.assertEqual(self.ns['checkpoint_lookup']._checkpoints, {})
                    self.assertIn('Local models need a complete checkpoint-link scan', self.pref.model_scan_status)

    def test_failed_cancelled_or_unpublished_refresh_preserves_catalog_and_snapshot(self):
        previous = scanner.ScanResult(linked=(scanner.LinkedCheckpoint('/models/keep.ckpt', 'STABLE_DIFFUSION_1',
                                                                      '/models/keep.ckpt'),))
        self.refresh(previous, [self.hub_model('owner/keep')])
        rows = [vars(row).copy() for row in self.pref.installed_models]
        checkpoints, models = self.ns['checkpoint_lookup']._checkpoints, self.ns['model_lookup']._models
        for failure in ['scanner', 'cancelled', 'hub', 'assignment', 'cancel-refresh']:
            with self.subTest(failure=failure):
                scan = scanner.ScanResult(error='scan failure') if failure == 'scanner' else scanner.ScanResult(
                    cancelled=failure == 'cancelled', limited=failure == 'assignment')
                self.service.take_result = lambda token: scan
                self.future.result = lambda: [self.hub_model('owner/new')]
                if failure == 'hub':
                    def fail():
                        raise RuntimeError('hub failure')
                    self.future.result = fail
                elif failure == 'assignment':
                    self.future.result = lambda: [NS(id='fail-assignment', downloads=0, likes=0,
                                                    model_type=self.ns['ModelType'].PROMPT_TO_IMAGE)]
                self.ns['fetch_installed_models']()
                poll = self.ns['_model_refresh_timer']
                if failure == 'cancel-refresh':
                    self.ns['_cancel_model_refresh']()
                poll()
                self.assertEqual([vars(row) for row in self.pref.installed_models], rows)
                self.assertIs(self.ns['checkpoint_lookup']._checkpoints, checkpoints)
                self.assertIs(self.ns['model_lookup']._models, models)
                self.assertIs(self.ns['_published_model_scan'], previous)
                if failure != 'cancel-refresh':
                    self.assertIn('failed', self.pref.model_scan_status)


if __name__ == '__main__':
    unittest.main(verbosity=2)
