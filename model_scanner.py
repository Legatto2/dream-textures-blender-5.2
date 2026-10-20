"""Bounded, read-only discovery of supported local checkpoint files.

Model-folder discovery is adapted from Jason Hoku's dream-textures commit
63c19fd. The validation, identities, budgets and cancellable worker below are
specific to this compatibility fork. GPL-3.0-or-later; see ATTRIBUTION.md.

This module deliberately imports neither bpy nor the AI runtime. Workers return
plain data; Blender preferences are updated only by the main-thread caller.
"""

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
import os
import stat
import threading
import time


@dataclass(frozen=True)
class ScanLimits:
    header_bytes: int = 8 * 1024 * 1024
    total_header_bytes: int = 64 * 1024 * 1024
    entries: int = 10000
    directories: int = 1024
    depth: int = 12
    seconds: float = 15.0


@dataclass(frozen=True)
class ModelClassification:
    config: str
    model_type: str


@dataclass(frozen=True)
class ScannedModel:
    path: str
    root: str
    display: str
    identity: str
    identifier: str
    classification: ModelClassification


@dataclass(frozen=True)
class LinkedCheckpoint:
    path: str
    config: str
    identity: str


@dataclass(frozen=True)
class ScanResult:
    models: tuple = ()
    linked: tuple = ()
    skipped: int = 0
    errors: int = 0
    limited: bool = False
    cancelled: bool = False
    error: str = ""
    linked_complete: bool = True
    linked_settings: tuple = ()


def normalize_path(path, path_module=os.path):
    """Normalize an absolute path without filesystem access (including Windows)."""
    return path_module.normcase(path_module.normpath(path))


def canonical_path(path):
    # May access the filesystem to resolve aliases; call from the worker only.
    return normalize_path(os.path.realpath(os.path.abspath(path)))


def model_identifier(identity):
    # Labels, root order, and characters such as '--' never determine identity.
    return "local:" + sha256(os.fsencode(identity)).hexdigest()


def reconcile_scan_result(scan, previous, folders):
    """Retain complete local results when a capped scan refreshes the Hub list.

    The manual-link stage shares the scan budget. Its partial prefix cannot
    replace a complete snapshot, and a snapshot with different link settings
    could restore a removed link or override a newly selected configuration.
    """
    if not scan.limited:
        return scan

    def settings_key(settings):
        return tuple((normalize_path(os.path.abspath(path)), config) for path, config in settings)

    if not scan.linked_complete and settings_key(scan.linked_settings) != settings_key(previous.linked_settings):
        return replace(scan, models=(), linked=())
    roots = {normalize_path(os.path.abspath(path)) for path in folders}
    models = tuple(model for model in previous.models if normalize_path(model.root) in roots)
    linked = scan.linked if scan.linked_complete else previous.linked
    return replace(scan, models=models, linked=linked)


class _Stopped(Exception):
    pass


class _Budget:
    def __init__(self, limits, cancel):
        self.limits = limits
        self.cancel = cancel
        self.deadline = time.monotonic() + limits.seconds
        self.entries = self.directories = self.header_bytes = 0
        self.limited = False

    def check(self):
        if self.cancel.is_set():
            raise _Stopped()
        if time.monotonic() >= self.deadline:
            self.limited = True
            raise _Stopped()

    def take(self, kind, count=1):
        self.check()
        limit = self.limits.total_header_bytes if kind == "header_bytes" else getattr(self.limits, kind)
        if getattr(self, kind) + count > limit:
            self.limited = True
            raise _Stopped()
        setattr(self, kind, getattr(self, kind) + count)


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_safetensors_header(path, budget):
    """Validate the header and declared tensor layout without reading weights."""
    try:
        with open(path, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            length = int.from_bytes(prefix, "little")
            if not 2 <= length <= budget.limits.header_bytes or length > before.st_size - 8:
                return None
            budget.take("header_bytes", length)
            data = handle.read(length)
            if len(data) != length or _fingerprint(before) != _fingerprint(os.fstat(handle.fileno())):
                return None
        header = json.loads(data, object_pairs_hook=_unique_object)
        if not isinstance(header, dict) or not 1 <= len(header) <= 20000:
            return None
        payload = before.st_size - 8 - length
        intervals = []
        for key, tensor in header.items():
            if key == "__metadata__":
                if not isinstance(tensor, dict) or not all(isinstance(v, str) for v in tensor.values()):
                    return None
                continue
            if not key or not isinstance(tensor, dict):
                return None
            shape = tensor.get("shape")
            offsets = tensor.get("data_offsets")
            width = _DTYPE_BYTES.get(tensor.get("dtype"))
            if (not isinstance(shape, list) or len(shape) > 8
                    or not all(type(n) is int and n >= 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or not all(type(n) is int for n in offsets) or width is None):
                return None
            start, end = offsets
            if not 0 <= start <= end <= payload or end - start != math.prod(shape) * width:
                return None
            intervals.append((start, end))
        # Safetensors describes the entire payload without overlaps or holes.
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                return None
            cursor = end
        if not intervals or cursor != payload:
            return None
        return header
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def classify_header(header):
    """Recognize complete LDM checkpoints that have a supported explicit config.

    Ambiguous checkpoints, separate diffusers components, LoRAs, and DiT models
    remain available through the existing manual linking workflow, not this list.
    """
    if not isinstance(header, dict):
        return None
    keys = tuple(key for key in header if key != "__metadata__")
    if any(any(marker in key for marker in (
            "lora_", ".lora_A.", ".lora_B.", ".hada_w", ".lokr_w",
            "double_blocks.", "single_blocks.", "joint_blocks.",
            "transformer_blocks.0.attn.add_q_proj")) for key in keys):
        return None

    def shape(key):
        value = header.get(key)
        if not isinstance(value, dict) or value.get("dtype") not in {"F16", "BF16", "F32", "F64"}:
            return ()
        value = value.get("shape")
        return tuple(value) if isinstance(value, list) and all(type(n) is int and n > 0 for n in value) else ()

    # Original single-file ControlNets. Separate diffusers ControlNet weights
    # require their config.json and are discovered through the existing hub path.
    if shape("control_model.input_blocks.0.0.weight") == (320, 4, 3, 3):
        if (shape("control_model.input_hint_block.0.weight") != (16, 3, 3, 3)
                or shape("control_model.zero_convs.0.0.weight") != (320, 320, 1, 1)):
            return None
        context = shape("control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight")
        if context == (320, 768):
            return ModelClassification("CONTROL_NET_1_5", "CONTROL_NET")
        if context == (320, 1024):
            return ModelClassification("CONTROL_NET_2_1", "CONTROL_NET")
        return None

    unet = shape("model.diffusion_model.input_blocks.0.0.weight")
    if len(unet) != 4 or unet[2:] != (3, 3):
        return None
    channels, inputs = unet[:2]
    if (shape("model.diffusion_model.out.2.weight") != (4, channels, 3, 3)
            or shape("first_stage_model.encoder.conv_in.weight") != (128, 3, 3, 3)
            or shape("first_stage_model.decoder.conv_out.weight") != (3, 128, 3, 3)):
        return None

    if channels == 320 and inputs == 4 and shape(
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight") == (49408, 768):
        return ModelClassification("STABLE_DIFFUSION_1", "PROMPT_TO_IMAGE")

    if channels == 320 and shape("cond_stage_model.model.token_embedding.weight") == (49408, 1024):
        if inputs == 5:
            return ModelClassification("STABLE_DIFFUSION_2_DEPTH", "DEPTH")
        if inputs == 9:
            return ModelClassification("STABLE_DIFFUSION_2_INPAINTING", "INPAINTING")
        metadata = header.get("__metadata__", {})
        prediction = metadata.get("modelspec.prediction_type") if isinstance(metadata, dict) else None
        if inputs == 4 and prediction == "epsilon":
            return ModelClassification("STABLE_DIFFUSION_2_BASE", "PROMPT_TO_IMAGE")
        if inputs == 4 and prediction in {"v", "v_prediction"}:
            return ModelClassification("STABLE_DIFFUSION_2", "PROMPT_TO_IMAGE")
        return None  # Tensor dimensions cannot distinguish SD2 epsilon and v-prediction.

    if channels == 320 and inputs == 4 and (
            shape("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") == (49408, 768)
            and shape("conditioner.embedders.1.model.token_embedding.weight") == (49408, 1280)):
        return ModelClassification("STABLE_DIFFUSION_XL_BASE", "PROMPT_TO_IMAGE")
    if channels == 384 and inputs == 4 and (
            shape("conditioner.embedders.0.model.token_embedding.weight") == (49408, 1280)
            and not any(key.startswith("conditioner.embedders.1.") for key in keys)):
        return ModelClassification("STABLE_DIFFUSION_XL_REFINER", "PROMPT_TO_IMAGE")
    return None


def scan_model_folders(folders, linked_checkpoints=(), *, limits=None, cancel=None, cache=None):
    """Scan requested roots only; limits are checked between OS operations.

    Network filesystems can block an individual OS call. The caller therefore
    uses a single daemon worker and never joins it from Blender's UI thread.
    """
    budget = _Budget(limits or ScanLimits(), cancel or threading.Event())
    linked_checkpoints = tuple(linked_checkpoints)
    linked_complete = False
    cache = {} if cache is None else cache
    models, linked = [], {}
    seen_files, seen_roots, seen_dirs = set(), set(), set()
    skipped = errors = 0
    try:
        # Keep the existing basename identifiers and explicit config choices.
        # Direct file links override a same-name file found in a linked folder.
        for raw_path, config in linked_checkpoints:
            budget.take("entries")
            path = os.path.abspath(raw_path)
            try:
                if os.path.isfile(path):
                    linked[os.path.basename(path)] = LinkedCheckpoint(path, config, canonical_path(path))
                elif os.path.isdir(path):
                    budget.take("directories")
                    with os.scandir(path) as entries:
                        for entry in entries:
                            budget.take("entries")
                            if os.path.splitext(entry.name)[1] in {".safetensors", ".ckpt", ".pth"} and entry.is_file():
                                linked.setdefault(entry.name, LinkedCheckpoint(entry.path, config, canonical_path(entry.path)))
            except OSError:
                errors += 1

        linked_complete = True
        linked_paths = {item.identity for item in linked.values()}
        for raw_root in folders:
            budget.check()
            root = os.path.abspath(raw_root)
            identity = canonical_path(root)
            if identity in seen_roots:
                continue
            seen_roots.add(identity)
            stack = [(root, 0)]
            while stack:
                directory, depth = stack.pop()
                budget.check()
                directory_id = canonical_path(directory)
                if directory_id in seen_dirs:
                    continue
                seen_dirs.add(directory_id)
                budget.take("directories")
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            budget.take("entries")
                            if entry.name.startswith("."):
                                continue
                            # Do not follow directory symlinks or Windows junctions.
                            if entry.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(entry.path)):
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if depth < budget.limits.depth:
                                    stack.append((entry.path, depth + 1))
                                else:
                                    budget.limited = True
                                continue
                            if os.path.splitext(entry.name)[1] != ".safetensors" or not entry.is_file(follow_symlinks=False):
                                continue
                            path_id = canonical_path(entry.path)
                            if path_id in seen_files or path_id in linked_paths:
                                continue
                            seen_files.add(path_id)
                            try:
                                fingerprint = _fingerprint(entry.stat(follow_symlinks=False))
                                cached = cache.get(path_id)
                                if cached is not None and cached[0] == fingerprint:
                                    classification = cached[1]
                                else:
                                    header = _read_safetensors_header(entry.path, budget)
                                    classification = classify_header(header)
                                    cache[path_id] = (fingerprint, classification)
                            except OSError:
                                errors += 1
                                continue
                            if classification is None:
                                skipped += 1
                                continue
                            models.append(ScannedModel(
                                entry.path, root, os.path.relpath(entry.path, root).replace(os.sep, "/"),
                                path_id, model_identifier(path_id), classification,
                            ))
                except OSError:
                    errors += 1
    except _Stopped:
        pass

    counts = {}
    for model in models:
        counts[model.display] = counts.get(model.display, 0) + 1
    models = [replace(model, display=f"{os.path.basename(model.root)}/{model.display} [{model.identifier[-8:]}]")
              if counts[model.display] > 1 else model for model in models]
    models.sort(key=lambda model: (model.display.casefold(), model.identifier))
    # Cache only compact classifications from the current bounded visit.
    for key in list(cache):
        if key not in seen_files:
            del cache[key]
    return ScanResult(
        models=tuple(models), linked=tuple(linked.values()), skipped=skipped, errors=errors,
        limited=budget.limited, cancelled=budget.cancel.is_set(),
        linked_complete=linked_complete, linked_settings=linked_checkpoints,
    )


@dataclass
class _Job:
    folders: tuple
    linked: tuple
    token: object = field(default_factory=object)
    cancel: threading.Event = field(default_factory=threading.Event)


class ModelScanService:
    """One worker; rescan replaces pending work and invalidates older results."""
    def __init__(self, scanner=scan_model_folders):
        self._scanner = scanner
        self._condition = threading.Condition()
        self._latest = self._pending = self._completed = None
        self._closed = False
        self._cache = {}
        self._worker = threading.Thread(target=self._run, name="DreamTexturesModelScan", daemon=True)
        self._worker.start()

    def request(self, folders, linked=()):
        with self._condition:
            if self._closed:
                raise RuntimeError("Model scanner is closed")
            if self._latest is not None:
                self._latest.cancel.set()
            self._latest = self._pending = _Job(tuple(folders), tuple(linked))
            self._completed = None
            self._condition.notify()
            return self._latest.token

    def take_result(self, token):
        with self._condition:
            if self._latest is None or token is not self._latest.token:
                return None
            result = self._completed
            self._completed = None
            return result

    def cancel(self):
        with self._condition:
            if self._latest is not None:
                self._latest.cancel.set()
            self._latest = self._pending = self._completed = None

    def close(self):
        with self._condition:
            self._closed = True
            if self._latest is not None:
                self._latest.cancel.set()
            self._latest = self._pending = self._completed = None
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job, self._pending = self._pending, None
            try:
                result = self._scanner(job.folders, job.linked, cancel=job.cancel, cache=self._cache)
            except Exception as exception:
                result = ScanResult(error=f"{type(exception).__name__}: {exception}")
            with self._condition:
                if self._latest is job and not job.cancel.is_set() and not self._closed:
                    self._completed = result
