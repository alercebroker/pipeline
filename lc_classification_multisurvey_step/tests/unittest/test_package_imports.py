"""The package layout itself is a contract: the unit suite must import the pure
modules with no alerce_classifiers and no apf on the path (spec: the model lives
in the submodule, the step is the only module that needs it).

The four modules parametrised below are the ones that actually carry that
constraint — they are duck-typed over the OutputDTO precisely so they need no
alerce_classifiers, and they hold the logic the unit suite tests. Importing the
two package `__init__.py` files instead proves nothing: both are empty.
"""
import importlib
import sys
from contextlib import contextmanager

import pytest

PURE_MODULES = [
    "lc_classification_multisurvey_step.probabilities",
    "lc_classification_multisurvey_step.input_dto",
    "lc_classification_multisurvey_step.output_parser",
    "lc_classification_multisurvey_step.db.db",
]

FORBIDDEN = ("alerce_classifiers", "apf")


class _Blocker:
    """A meta_path finder that refuses `names` and everything under them."""

    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        for name in self.names:
            if fullname == name or fullname.startswith(name + "."):
                raise ImportError(
                    f"{fullname} is not importable here: this module must not need it"
                )
        return None


def _submodules_of(*prefixes):
    return [
        name
        for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    ]


@contextmanager
def _forbidden_and_uncached(module):
    """Block FORBIDDEN and evict it, plus `module`'s package, from sys.modules.

    Blocking alone is not enough: `import apf` is satisfied from sys.modules
    before any finder is consulted, and another test in the same session
    legitimately imports both (step.py needs them). Evicting the modules under
    test too forces the import to actually re-execute rather than hit the cache.
    """
    evicted = _submodules_of(*FORBIDDEN, "lc_classification_multisurvey_step")
    saved = {name: sys.modules.pop(name) for name in evicted}
    blocker = _Blocker(FORBIDDEN)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        # Drop whatever the blocked import left behind, then restore the
        # originals so the rest of the session sees the session's own objects.
        for name in _submodules_of(*FORBIDDEN, "lc_classification_multisurvey_step"):
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.mark.parametrize("module", PURE_MODULES)
def test_pure_modules_import_without_alerce_classifiers_or_apf(module):
    with _forbidden_and_uncached(module):
        assert importlib.import_module(module) is not None
        leaked = _submodules_of(*FORBIDDEN)
        assert not leaked, f"{module} pulled in {sorted(leaked)}"


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_the_blocker_actually_blocks(forbidden):
    """Guard against the tests above passing because the blocker does nothing.

    Matched on the blocker's own message, not merely on ImportError: a finder is
    consulted before the module is located, so this holds whether or not the
    dependency happens to be installed in the running environment — and it will
    not be satisfied by some unrelated missing import.
    """
    with _forbidden_and_uncached(forbidden):
        with pytest.raises(ImportError, match="must not need it"):
            importlib.import_module(forbidden)
