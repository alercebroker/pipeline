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

2026-09-03: 44640 rows over 992 objects, all matching exactly -- same
probabilities, same rankings, same top class everywhere. Same features and the
same model, so nothing less than exact would be acceptable.

Getting there needed one thing beyond reading the rows: `_current_features`
below drops feature rows superseded by a later computation. Three objects
carried a handful of rows a day older than their siblings, and those three were
the only ones whose probabilities disagreed.
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


def _current_features(entry: dict) -> list:
    """(name, band, value) for the rows the object's newest computation produced.

    `feature` is upserted with ON CONFLICT (oid, sid, feature_id, band) DO UPDATE
    ... updated_date = now(), so a pass only touches the rows it computed. A
    feature produced in an earlier pass but not in the latest one keeps its old
    row, old value and old date while every sibling row moves on -- and the
    classifier, working from that latest computation, saw NaN for it. Reading
    such a row back as a value feeds the model something production never had.

    Three of the 992 objects in this fixture carry exactly that: 3 or 4 rows
    dated a day before the other ~190, and they were the only three whose
    recomputed probabilities disagreed with the stored ones. Dropping the
    superseded rows is what makes the comparison a like-for-like one.
    """
    latest = max(updated for _, _, _, updated in entry["features"])
    return [
        (name, band, value)
        for name, band, value, updated in entry["features"]
        if updated == latest
    ]


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
            _model_column(name, band): value
            for name, band, value in _current_features(entry)
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
    """Row-for-row equality. Same features and same model, so nothing less."""
    rows, stored = replayed

    deviating = {
        row["oid"]
        for row in rows
        if abs(stored[(row["oid"], row["classifier_id"], row["class_id"])][0] - row["probability"])
        > PROBABILITY_TOLERANCE
    }
    assert deviating == set()

    rank_mismatches = [
        ((row["oid"], row["classifier_id"], row["class_id"]), stored_ranking, row["ranking"])
        for row in rows
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

    assert computed == expected
    assert len(computed) == len(examples_objects := set(row["oid"] for row in rows)) * len(
        taxonomy_seed.classifier_ids()
    ), f"expected one call per head for each of {len(examples_objects)} objects"
