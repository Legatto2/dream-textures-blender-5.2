"""Persistent model identities behind Blender's integer-valued enum UI.

Blender 5.2.2's ``get_transform``/``set_transform`` callbacks expose the old
internal enum number while letting a separate custom string be authoritative.
See blender/blender v5.2.2, doc/python_api/examples/bpy.props.5.py and
source/blender/python/intern/bpy_props.cc. No Blender or model runtime imports
are needed here. Runtime enum numbers are never persisted as model identities.
"""

import threading


UNSELECTED = "dream-textures:unselected"


def legacy_models(models):
    """Reconstruct the old rows, including separator positions, without scans.

    The Diffusers backend grouped legacy rows with a leading None per group.
    An automatic-only group must not leave an extra separator in that list.
    Other backends' lists are left intact when there are no automatic rows.
    """
    models = list(models)
    if not any(model is not None and model.id.startswith("local:") for model in models):
        return models
    legacy = []
    for model in models:
        if model is not None and model.id.startswith("local:"):
            continue
        if model is None and legacy and legacy[-1] is None:
            continue
        legacy.append(model)
    while legacy and legacy[-1] is None:
        legacy.pop()
    return legacy


class ModelSelection:
    """Keep full per-instance identifiers and non-reused, session enum values.

    One instance belongs to one public enum property. The strong string pool
    also satisfies Blender's dynamic enum callback string-lifetime requirement.
    Getters do not write RNA data, including when resolving legacy selections.
    """

    def __init__(self, storage_key):
        self.storage_key = storage_key
        self._numbers = {UNSELECTED: 0}
        self._identifiers = {0: UNSELECTED}
        self._strings = {UNSELECTED: UNSELECTED}
        self._lock = threading.Lock()

    def _string(self, value):
        value = str(value)
        return self._strings.setdefault(value, value)

    def _number(self, identifier):
        identifier = self._string(identifier)
        # Blender can query property callbacks from a threaded context.
        with self._lock:
            if identifier not in self._numbers:
                number = len(self._numbers)
                if number > 2**31 - 1:
                    raise OverflowError("Too many model menu entries in this session.")
                self._numbers[identifier] = number
                self._identifiers[number] = identifier
            return self._numbers[identifier]

    def _stored(self, owner):
        if owner is None:
            return None
        value = owner.get(self.storage_key)
        return value if isinstance(value, str) else None

    @staticmethod
    def _resolve(identifier, resolve):
        return resolve(identifier) if resolve is not None and identifier else identifier

    def items(self, owner, models, resolve=None):
        """Return enum rows; None owner is useful for menus of available models."""
        models = list(models)
        rows = []
        available = set()
        for model in models:
            if model is None:
                rows.append(None)
            elif model.id not in available:
                available.add(model.id)
                rows.append((self._string(model.id), self._string(model.name),
                             self._string(model.description), self._number(model.id)))
        if owner is not None:
            stored = self._stored(owner)
            selected = self._resolve(stored, resolve)
            if selected and selected not in available:
                if rows and rows[-1] is not None:
                    rows.append(None)
                rows.append((self._string(selected), self._string("Unavailable model"),
                             self._string(f"The selected model is unavailable: {stored}. Refresh or select another model."),
                             self._number(selected)))
            # Zero has one explicit meaning; it can never become a model row.
            rows.append((self._string(UNSELECTED), self._string("Select a model"),
                         self._string("No model is selected"), 0))
        return rows

    def get(self, owner, current_value, is_set, models, resolve=None):
        """Translate full storage, or the old positional value, for Blender."""
        stored = self._stored(owner)
        if stored is not None:
            selected = self._resolve(stored, resolve)
            return self._number(selected) if selected else 0

        legacy = legacy_models(models)
        # An unset old EnumProperty used number zero. Preserve that default only
        # when an actual legacy row occupies it; automatic models cannot fill it.
        index = current_value if is_set else 0
        if type(index) is int and 0 <= index < len(legacy) and legacy[index] is not None:
            return self._number(self._resolve(legacy[index].id, resolve))
        return 0

    def set(self, owner, new_value, current_value, is_set, models, resolve=None):
        """Store a full identifier; an old open menu keeps its original mapping."""
        if type(new_value) is not int or new_value not in self._identifiers:
            raise ValueError("The selected model menu value is no longer available.")
        identifier = self._identifiers[new_value]
        owner[self.storage_key] = "" if identifier == UNSELECTED else identifier
        return new_value
