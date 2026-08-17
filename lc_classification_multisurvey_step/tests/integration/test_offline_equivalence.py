"""Equivalence against the offline reference implementation.

The step's batched row builder and offline `probability_writer.build_probability_rows`
must agree. Offline is strictly per-oid and pins CLASSIFIER_IDS = [5..9]; the step
melts by oid and resolves ids from the DB. So this feeds one OutputDTO through
both and compares row sets modulo ordering.

Opt-in: needs RUN_EQUIVALENCE_TEST=1 and the offline checkout present at
`~/desktop/pipeline/feature_step` — note the `feature_step` subdirectory, since
the checkout root alone does not resolve `features.offline`. It needs neither the
`alerce_classifiers` submodule nor MODEL_PATH: both row builders are pure and no
classifier is run. Not part of the default unit run.

What this cannot cover: the multi-oid melt path. The offline reference is strictly
per-oid and raises on a multi-row frame by design, so there is nothing to compare
against there; multi-oid behaviour is covered by the unit tests only.
"""
import importlib.util
import os
import sys

import pytest

# The offline reference lives in the `feature_step` subtree of the offline
# checkout: <root>/features/offline/probability_writer.py resolves only with
# `feature_step` on sys.path, not the checkout root.
OFFLINE_ROOT = os.path.expanduser("~/desktop/pipeline/feature_step")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_EQUIVALENCE_TEST"),
    reason="opt-in: set RUN_EQUIVALENCE_TEST=1 (needs the offline checkout)",
)


@pytest.fixture(scope="module")
def offline_writer():
    if not os.path.isdir(OFFLINE_ROOT):
        pytest.skip(f"offline checkout not found at {OFFLINE_ROOT}")
    if OFFLINE_ROOT not in sys.path:
        sys.path.insert(0, OFFLINE_ROOT)
    if importlib.util.find_spec("features.offline.probability_writer") is None:
        pytest.skip("features.offline.probability_writer not importable")
    from features.offline import probability_writer

    return probability_writer


@pytest.fixture
def dto():
    """A single-oid BHRF-shaped OutputDTO built from the real class names."""
    import pandas as pd
    from types import SimpleNamespace

    def frame(data):
        df = pd.DataFrame(data, index=[123456789])
        df.index.name = "oid"
        return df

    return SimpleNamespace(
        probabilities=frame({"SNIa": [0.6], "AGN": [0.3], "LPV": [0.1]}),
        hierarchical={
            "top": frame({"Transient": [0.7], "Stochastic": [0.2], "Periodic": [0.1]}),
            "children": {
                "Transient": frame({"SNIa": [0.8], "SLSN": [0.2]}),
                "Stochastic": frame({"AGN": [0.9], "QSO": [0.1]}),
                "Periodic": frame({"LPV": [0.5], "EA": [0.5]}),
            },
        },
    )


@pytest.fixture
def taxonomy_maps():
    """Mirrors the offline classifier_taxonomy_lut ids 5-9 for the classes above."""
    return {
        5: {"SNIa": 0, "AGN": 1, "LPV": 2},
        6: {"Transient": 0, "Stochastic": 1, "Periodic": 2},
        7: {"SNIa": 0, "SLSN": 1},
        8: {"AGN": 0, "QSO": 1},
        9: {"LPV": 0, "EA": 1},
    }


def test_row_sets_match_offline(offline_writer, dto, taxonomy_maps):
    from lc_classification_multisurvey_step.probabilities import (
        build_probability_rows,
        head_names,
    )

    oid, lastmjd = 123456789, 60123.5
    # Offline pins ids 5-9; the step resolves them. Bind the step's head names to
    # those same ids so the comparison isolates the row-building logic.
    classifier_ids = dict(zip(head_names(), [5, 6, 7, 8, 9]))

    offline_rows = offline_writer.build_probability_rows(
        dto, oid, lastmjd, taxonomy_maps, version="2.1.0", sid=0
    )
    step_rows = build_probability_rows(
        dto, {oid: lastmjd}, classifier_ids, taxonomy_maps, version="2.1.0", sid=0
    )

    def key(row):
        return tuple(sorted(row.items()))

    assert sorted(map(key, step_rows)) == sorted(map(key, offline_rows))
    assert len(step_rows) == 12  # 3 + 3 + 2 + 2 + 2
