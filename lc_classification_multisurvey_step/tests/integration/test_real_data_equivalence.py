"""Real features from the live database, through the step, against the stored rows.

The local harness proves the wiring but cannot say the port classifies
*correctly*: its features are random floats and its seed is a local copy. This
test closes that gap by taking objects that production has already classified,
recomputing their probabilities through the step's own path, and requiring the
result to match what is in `multisurvey_ztf.probability` exactly.

Opt-in, because it needs three things CI does not have: the VPN, read access to
the live database, and the BHRF pickle.

    REAL_DB_CONFIG=$(pwd)/local_config.yaml \\
    MODEL_PATH=/path/to/model/2.1.0 \\
    python -m pytest tests/integration/test_real_data_equivalence.py -v

`REAL_DB_CONFIG` is a yaml with a `PSQL_CONFIG` block -- the local run config
already has one -- so no credentials live in this repo.

First run, 2026-09-03: 225 rows over 5 objects, max |Δp| = 1.1e-16 (one float64
epsilon), every `ranking` and every top-1 class identical.
"""
import os

import pandas as pd
import pytest
import yaml
from sqlalchemy import bindparam, text

from lc_classification_multisurvey_step.db.db import PSQLConnection, resolve_classifiers
from lc_classification_multisurvey_step.input_dto import create_input_dto
from lc_classification_multisurvey_step.probabilities import (
    build_probability_rows,
    head_names,
)

CONFIG_ENV = "REAL_DB_CONFIG"
BASE_NAME = "lc_classifier_BHRF_forced_phot"
MODEL_VERSION = "2.1.0"
ZTF_SID = 0

# Picked because each has a full feature vector and all five heads already
# written. They are only a starting point: if production stops carrying them the
# test skips and names them, rather than failing as though the code broke.
OIDS = [
    36028933559740357,
    36028933559743251,
    36028933559736997,
    36028933559739832,
    36028933559741025,
]

# One float64 epsilon is what an exact recomputation costs; anything above this
# is a real difference, not arithmetic.
PROBABILITY_TOLERANCE = 1e-12

pytestmark = pytest.mark.skipif(
    not (os.getenv(CONFIG_ENV) and os.getenv("MODEL_PATH")),
    reason=(
        f"needs {CONFIG_ENV} (a yaml with a PSQL_CONFIG block, e.g. local_config.yaml) "
        "and MODEL_PATH, plus the VPN for the live database"
    ),
)


def _model_column(lut_name: str, band: int) -> str:
    """`feature_name_lut` spelling -> the column name the model's feature_list uses.

    The LUT writes colours with hyphens (`W1-W2`, `g-r_mean`) and ratios with
    slashes (`Power_rate_1/2`); the model uses underscores throughout. `band` 0
    means band-agnostic, 1 is g, 2 is r, and 12 is the (g,r) pair -- so
    `g-r_mean` at band 12 is the model's `g_r_mean_12`.

    Getting this wrong is silent: ~31 of the 199 features land as NaN and the
    model returns plausible but different probabilities.
    """
    name = lut_name.replace("-", "_").replace("/", "_")
    return f"{name}_{band}" if band else name


def _features_by_oid(connection, oids: list) -> dict:
    """{oid: {model column: value}} for the ZTF features stored for these objects."""
    statement = text(
        "SELECT f.oid, l.feature_name, f.band, f.value "
        "FROM feature f JOIN feature_name_lut l "
        "  ON l.feature_id = f.feature_id AND l.sid = f.sid "
        "WHERE f.sid = :sid AND f.oid IN :oids"
    ).bindparams(bindparam("oids", expanding=True))

    features: dict = {}
    with connection.session() as session:
        rows = session.execute(statement, {"sid": ZTF_SID, "oids": oids}).mappings()
        for row in rows:
            column = _model_column(row["feature_name"], row["band"])
            features.setdefault(int(row["oid"]), {})[column] = row["value"]
    return features


def _stored_probabilities(connection, oids: list, classifier_ids: list) -> tuple:
    """({(oid, classifier_id, class_id): (probability, ranking)}, {oid: lastmjd})."""
    statement = text(
        "SELECT oid, classifier_id, class_id, probability, ranking, lastmjd "
        "FROM probability WHERE oid IN :oids AND classifier_id IN :classifier_ids"
    ).bindparams(
        bindparam("oids", expanding=True),
        bindparam("classifier_ids", expanding=True),
    )

    stored: dict = {}
    lastmjd: dict = {}
    with connection.session() as session:
        rows = session.execute(
            statement, {"oids": oids, "classifier_ids": classifier_ids}
        ).mappings()
        for row in rows:
            oid = int(row["oid"])
            key = (oid, int(row["classifier_id"]), int(row["class_id"]))
            stored[key] = (float(row["probability"]), int(row["ranking"]))
            lastmjd[oid] = float(row["lastmjd"])
    return stored, lastmjd


@pytest.fixture(scope="module")
def live_db():
    with open(os.environ[CONFIG_ENV]) as handle:
        config = yaml.safe_load(handle)
    return PSQLConnection(config["PSQL_CONFIG"], poolclass="NullPool")


@pytest.fixture(scope="module")
def recomputed(live_db):
    """The step's own path over real features: (rows, stored, classifier_ids)."""
    classifier_ids, taxonomy_maps = resolve_classifiers(
        head_names(BASE_NAME), MODEL_VERSION, live_db
    )
    stored, lastmjd = _stored_probabilities(
        live_db, OIDS, list(classifier_ids.values())
    )
    features = _features_by_oid(live_db, OIDS)

    starved = [oid for oid in OIDS if oid not in features or oid not in lastmjd]
    if starved:
        pytest.skip(
            f"objects {starved} no longer have both features and BHRF probability "
            "rows in the live database; refresh OIDS with objects that do"
        )

    # Exactly the frame `build_features_frame` would produce from a message: one
    # row per oid, one column per feature the model asks for, and NaN where the
    # object has no row -- which is the null the Kafka message would carry.
    model_features = list(
        pd.read_pickle(
            os.path.join(
                os.environ["MODEL_PATH"], "hierarchical_random_forest_model.pkl"
            )
        )["feature_list"]
    )
    collapsed = {
        oid: {"features": {name: features[oid].get(name) for name in model_features}}
        for oid in OIDS
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
        classifier_ids,
        taxonomy_maps,
        base_name=BASE_NAME,
        version=MODEL_VERSION,
        sid=ZTF_SID,
    )
    return rows, stored, classifier_ids


def test_every_recomputed_row_exists_in_the_database(recomputed):
    """No row the step would write is absent from what production already wrote."""
    rows, stored, classifier_ids = recomputed

    assert rows, "the step produced no rows for objects the database has classified"

    orphans = [
        (row["oid"], row["classifier_id"], row["class_id"])
        for row in rows
        if (row["oid"], row["classifier_id"], row["class_id"]) not in stored
    ]
    assert orphans == []

    # Both sides cover the same ground: every stored row was also recomputed.
    assert len(rows) == len(stored)
    assert {row["classifier_id"] for row in rows} == set(classifier_ids.values())


def test_probabilities_and_rankings_match_the_database(recomputed):
    """The port reproduces production's numbers, not merely plausible ones."""
    rows, stored, _ = recomputed

    worst = 0.0
    rank_mismatches = []
    for row in rows:
        key = (row["oid"], row["classifier_id"], row["class_id"])
        stored_probability, stored_ranking = stored[key]
        worst = max(worst, abs(stored_probability - row["probability"]))
        if stored_ranking != row["ranking"]:
            rank_mismatches.append((key, stored_ranking, row["ranking"]))

    assert worst <= PROBABILITY_TOLERANCE, f"max |Δp| = {worst:.3e}"
    assert rank_mismatches == []


def test_top_class_matches_the_database(recomputed):
    """The classification itself agrees, per object and per head."""
    rows, stored, _ = recomputed

    def top_by_head(items):
        best: dict = {}
        for (oid, classifier_id, class_id), probability in items:
            key = (oid, classifier_id)
            if key not in best or probability > best[key][1]:
                best[key] = (class_id, probability)
        return {key: value[0] for key, value in best.items()}

    computed = top_by_head(
        (
            (row["oid"], row["classifier_id"], row["class_id"]),
            row["probability"],
        )
        for row in rows
    )
    expected = top_by_head(
        (key, probability) for key, (probability, _) in stored.items()
    )

    assert computed == expected
