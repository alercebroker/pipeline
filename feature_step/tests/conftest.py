"""Pytest configuration for feature_step tests.

Adds lc_classifier, idmapper, and apf to sys.path so that offline tests
(and any test importing features.offline.*) can resolve those packages
without requiring a full pip install of the pipeline dependencies.
"""
import sys
from pathlib import Path

_PIPE = Path(__file__).resolve().parents[3]   # .../pipeline
for _p in (
    _PIPE / "lc_classifier",
    _PIPE / "libs" / "idmapper",
    _PIPE / "libs" / "apf",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
