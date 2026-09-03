"""100 real objects, replayed through the step, against production's own rows.

The synthetic harness proves the wiring but says nothing about whether the port
classifies *correctly*: its features are random floats. This takes objects the
production pipeline has already classified, recomputes their probabilities
through the step's own path, and requires every row to match what is stored in
`multisurvey_ztf.probability`.

The objects live in `data/real_examples.json.gz`, dumped by
`scripts/dump_real_examples.py`, so this needs neither the VPN nor database
credentials -- only `MODEL_PATH`. Regenerate the fixture after a model bump or a
change to the feature set.

This is also the only test that exercises the batch path at a realistic size:
`build_probability_rows` melts 100 oids at once, which the offline reference it
was ported from cannot do at all (it raises on a multi-row frame), and which the
synthetic harness only reaches with five.

First run, 2026-09-03: 4500 rows over 100 objects, max |Δp| = 0, every ranking
and every top-1 class identical.
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
    """Guard the inputs, so a bad dump cannot look like a passing comparison."""
    objects = examples["objects"]
    assert len(objects) == 100

    # All five heads for every object, or the row-count assertions below would
    # pass while silently covering less than they claim.
    for entry in objects:
        assert len(entry["probabilities"]) == taxonomy_seed.CLASSES_PER_OID
        heads = {classifier_id for classifier_id, _, _, _ in entry["probabilities"]}
        assert heads == set(taxonomy_seed.classifier_ids().values())
        assert entry["classifier_version"] == classifier_version_to_smallint(
            MODEL_VERSION
        )

    assert len({entry["oid"] for entry in objects}) == 100


def test_every_recomputed_row_exists_in_production(replayed):
    rows, stored = replayed

    orphans = [
        (row["oid"], row["classifier_id"], row["class_id"])
        for row in rows
        if (row["oid"], row["classifier_id"], row["class_id"]) not in stored
    ]
    assert orphans == []
    assert len(rows) == len(stored) == 100 * taxonomy_seed.CLASSES_PER_OID


def test_probabilities_and_rankings_match_production(replayed):
    rows, stored = replayed

    worst = 0.0
    worst_key = None
    rank_mismatches = []
    for row in rows:
        key = (row["oid"], row["classifier_id"], row["class_id"])
        stored_probability, stored_ranking = stored[key]
        difference = abs(stored_probability - row["probability"])
        if difference > worst:
            worst, worst_key = difference, key
        if stored_ranking != row["ranking"]:
            rank_mismatches.append((key, stored_ranking, row["ranking"]))

    assert worst <= PROBABILITY_TOLERANCE, f"max |Δp| = {worst:.3e} at {worst_key}"
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

    assert computed == expected
    assert len(computed) == 100 * len(taxonomy_seed.classifier_ids())
