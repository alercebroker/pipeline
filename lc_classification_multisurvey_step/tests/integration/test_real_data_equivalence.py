"""992 real objects, replayed through the step, against production's own rows.

The synthetic harness proves the wiring but says nothing about whether the port
classifies *correctly*: its features are random floats. This takes objects the
production pipeline has already classified, recomputes their probabilities
through the step's own path, and compares every row with what is stored in
`multisurvey_ztf.probability`.

The objects live in `data/real_examples.json.gz`, dumped by
`scripts/dump_real_examples.py`, so this needs neither the VPN nor database
credentials -- only `MODEL_PATH`.

They are taken as they come. The only condition for inclusion is that the object
has BHRF probability rows to compare against; there is no filter on feature
completeness, so the set spans objects carrying 19 to 189 of the model's 199
features and exercises the NaN handling on real sparsity.

This is also the only test that melts a realistic batch -- 992 oids in one call.
The offline reference the code was ported from raises on a multi-row frame, and
the synthetic harness only reaches five.

2026-09-03: 44640 rows. Row sets identical. 989 of 992 objects match to 1.1e-16
(one float64 epsilon); 4959 of 4960 (oid, head) pairs pick the same top class.
The three exceptions are recorded in KNOWN_DIVERGENT below.
"""
import gzip
import json
import os
import pathlib

import pandas as pd
import pytest

from lc_classification_multisurvey_step.input_dto import create_input_dto
from lc_classification_multisurvey_step.probabilities import (
    build_probability_rows,
    classifier_version_to_smallint,
)

from . import taxonomy_seed

FIXTURE = pathlib.Path(__file__).parent / "data" / "real_examples.json.gz"
BASE_NAME = taxonomy_seed.DEFAULT_CLASSIFIER_NAME
MODEL_VERSION = taxonomy_seed.CLASSIFIER_VERSION
ZTF_SID = 0

# `probability.probability` is REAL, so the stored value is float32 while the
# recomputation is float64; anything this small is the cast, not a difference.
PROBABILITY_TOLERANCE = 1e-7

# Three objects whose stored probabilities differ from a recomputation on their
# current features, by 0.010, 0.008 and 0.006.
#
# Not sparsity: objects with 28 of 199 features present match to 1e-16, while
# these carry 124, 180 and 184. Not stale features either, in the sense that is
# checkable -- `probability.lastmjd` equals the object's last detection mjd, so
# no new photometry arrived after the classification -- and not duplicate
# feature rows, which no object in the fixture has.
#
# What they look like is a handful of trees voting differently: the deltas are
# 5, 4 and 3 times 1/500, the vote quantum of a 500-estimator forest. That is
# what a marginally different input does, not what different code does, which
# would move every object. `feature.updated_date` is a DATE, so features
# recomputed later on the same day as the classification are indistinguishable
# from ones written before it, and that remains the untested explanation.
#
# They are kept in the fixture rather than filtered out, and the tests assert
# the deviating set is exactly this one: a new divergence fails, and so does one
# of these disappearing, either of which is worth looking at.
KNOWN_DIVERGENT = {
    36028933559737010,
    36028933559737043,
    36028933559737056,
}

# The single (oid, head) pair where the two sides name a different top class.
# Head 9 is `_periodic`; the object is one of KNOWN_DIVERGENT.
KNOWN_TOP_CLASS_DISAGREEMENT = {(36028933559737056, 9)}

pytestmark = pytest.mark.skipif(
    not os.getenv("MODEL_PATH"),
    reason=(
        "MODEL_PATH is unset; this replays real objects through the real BHRF "
        "2.1.0 model, which is not in this repo"
    ),
)


def _model_column(lut_name: str, band: int) -> str:
    """`feature_name_lut` spelling -> the column name the model's feature_list uses.

    The LUT writes colours with hyphens (`W1-W2`, `g-r_mean`) and ratios with
    slashes (`Power_rate_1/2`); the model uses underscores throughout. `band` 0
    means band-agnostic, 1 is g, 2 is r, and 12 is the (g,r) pair -- so
    `g-r_mean` at band 12 is the model's `g_r_mean_12`.

    Kept here, applied to the fixture's raw database spellings, rather than baked
    into the fixture: getting it wrong is silent -- around 31 of the 199 features
    land as NaN and the model returns plausible but different probabilities -- so
    it is worth having under test.
    """
    name = lut_name.replace("-", "_").replace("/", "_")
    return f"{name}_{band}" if band else name


@pytest.fixture(scope="module")
def examples() -> dict:
    if not FIXTURE.exists():
        pytest.skip(
            f"{FIXTURE.name} is missing; regenerate it with "
            "REAL_DB_CONFIG=... python scripts/dump_real_examples.py"
        )
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def replayed(examples) -> tuple:
    """(computed rows, stored rows keyed by (oid, classifier_id, class_id)).

    One model load and one batch for the whole fixture -- the step classifies a
    batch in a single call, and loading the pickle per test would cost 1.6 GB
    each time.
    """
    objects = examples["objects"]

    model_features = list(
        pd.read_pickle(
            os.path.join(
                os.environ["MODEL_PATH"], "hierarchical_random_forest_model.pkl"
            )
        )["feature_list"]
    )

    collapsed = {}
    for entry in objects:
        values = {
            _model_column(name, band): value for name, band, value in entry["features"]
        }
        # Exactly the frame a message would produce: every column the model asks
        # for, NaN where the object has no row -- which is the null the Kafka
        # message carries for a feature that was not computed.
        collapsed[int(entry["oid"])] = {
            "features": {name: values.get(name) for name in model_features}
        }

    lastmjd = {int(entry["oid"]): entry["lastmjd"] for entry in objects}
    stored = {
        (int(entry["oid"]), classifier_id, class_id): (probability, ranking)
        for entry in objects
        for classifier_id, class_id, probability, ranking in entry["probabilities"]
    }

    from alerce_classifiers.squidward.mapper import SquidwardMapper
    from alerce_classifiers.squidward.model import SquidwardFeaturesClassifier

    model = SquidwardFeaturesClassifier(
        model_path=os.environ["MODEL_PATH"], mapper=SquidwardMapper()
    )
    output_dto = model.predict(create_input_dto(collapsed))

    rows = build_probability_rows(
        output_dto,
        lastmjd,
        taxonomy_seed.classifier_ids(BASE_NAME),
        taxonomy_seed.taxonomy_maps(BASE_NAME),
        base_name=BASE_NAME,
        version=MODEL_VERSION,
        sid=ZTF_SID,
    )
    return rows, stored


def test_the_fixture_is_what_it_claims(examples):
    """Guard the inputs, so a thin dump cannot look like a passing comparison."""
    objects = examples["objects"]
    oids = [int(entry["oid"]) for entry in objects]

    assert len(oids) == len(set(oids))
    assert len(objects) > 500, "too few objects to say much"

    seeded_heads = set(taxonomy_seed.classifier_ids().values())
    for entry in objects:
        assert entry["probabilities"], f"oid {entry['oid']} has nothing to compare"
        heads = {classifier_id for classifier_id, _, _, _ in entry["probabilities"]}
        assert heads == seeded_heads
        assert entry["classifier_version"] == classifier_version_to_smallint(
            MODEL_VERSION
        )

    # The set must keep spanning sparse objects, or it stops covering the NaN
    # handling and quietly becomes the filtered set this fixture replaced.
    present = [len(entry["features"]) for entry in objects]
    assert min(present) < 100, "no feature-sparse objects left in the fixture"


def test_every_recomputed_row_exists_in_production(replayed):
    """The step writes exactly the rows production wrote -- no more, no fewer."""
    rows, stored = replayed

    computed_keys = {
        (row["oid"], row["classifier_id"], row["class_id"]) for row in rows
    }
    assert computed_keys - set(stored) == set(), "rows the step invents"
    assert set(stored) - computed_keys == set(), "rows the step would fail to write"
    assert len(rows) == len(stored)


def test_probabilities_and_rankings_match_production(replayed):
    """Row-for-row equality, except for the objects documented above."""
    rows, stored = replayed

    deviating = {
        row["oid"]
        for row in rows
        if abs(stored[(row["oid"], row["classifier_id"], row["class_id"])][0] - row["probability"])
        > PROBABILITY_TOLERANCE
    }
    assert deviating == KNOWN_DIVERGENT

    # Ranking is dense *within a head*, so one differing probability reshuffles
    # the ranks of every other class in that head -- including rows whose own
    # value still matches to 1e-7. A divergent object is therefore excluded
    # whole, not row by row. For every other object a ranking difference has no
    # such excuse and would be a real bug in how `build_probability_rows` ranks.
    rank_mismatches = [
        ((row["oid"], row["classifier_id"], row["class_id"]), stored_ranking, row["ranking"])
        for row in rows
        if row["oid"] not in KNOWN_DIVERGENT
        for stored_ranking in [stored[(row["oid"], row["classifier_id"], row["class_id"])][1]]
        if stored_ranking != row["ranking"]
    ]
    assert rank_mismatches == []


def test_top_class_matches_production(replayed):
    """The classification itself agrees, for every object and every head."""
    rows, stored = replayed

    def top_by_head(triples):
        best: dict = {}
        for (oid, classifier_id, class_id), probability in triples:
            key = (oid, classifier_id)
            if key not in best or probability > best[key][1]:
                best[key] = (class_id, probability)
        return {key: value[0] for key, value in best.items()}

    computed = top_by_head(
        ((row["oid"], row["classifier_id"], row["class_id"]), row["probability"])
        for row in rows
    )
    expected = top_by_head(
        (key, probability) for key, (probability, _) in stored.items()
    )

    assert computed.keys() == expected.keys()
    disagreements = {key for key in computed if computed[key] != expected[key]}
    assert disagreements == KNOWN_TOP_CLASS_DISAGREEMENT
