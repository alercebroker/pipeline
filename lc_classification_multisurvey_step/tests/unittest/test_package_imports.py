"""The package layout itself is a contract: the unit suite must import the pure
modules with no alerce_classifiers and no apf on the path (spec: the model lives
in the submodule, the step is the only module that needs it)."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "lc_classification_multisurvey_step",
        "lc_classification_multisurvey_step.db",
    ],
)
def test_package_modules_import(module):
    assert importlib.import_module(module) is not None
