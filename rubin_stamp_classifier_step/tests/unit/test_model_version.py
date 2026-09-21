"""Which classifier version reaches the database and scribe.

By default it is the version the model derives from its artifact path.
A non-empty top-level MODEL_VERSION overrides it, so a deployment can pin
the version it reports without depending on the URL layout. This is the
same key the lc classification step reads.
"""
from unittest import mock

import pytest

from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import CLASSES, MODEL_VERSION

TAXONOMY = {name: idx + 10 for idx, name in enumerate(CLASSES)}


def build_step(**extra):
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": {},
        "MODEL_CONFIG": {"CLASS": "tests.unit.stub_model.StubModel", "CLS_ID": 3},
        **extra,
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):
        return StampClassifierStep(config=config)


def prediction():
    return {
        "oid": 123,
        "sid": 1,
        "probabilities": {"AGN": 0.0, "SN": 1.0, "VS": 0.0, "asteroid": 0.0, "bogus": 0.0},
        "midpointMjdTai": 60000.5,
    }


@pytest.mark.parametrize("extra", [{}, {"MODEL_VERSION": ""}])
def test_version_defaults_to_the_one_the_model_reports(extra):
    step = build_step(**extra)

    assert step.model_version == MODEL_VERSION


def test_config_version_overrides_the_model_one():
    step = build_step(MODEL_VERSION="2.1.0")

    assert step.model_version == "2.1.0"


def test_db_write_uses_the_step_version():
    step = build_step(MODEL_VERSION="2.1.0")

    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:
        step.post_execute([prediction()])

    assert store.call_args.kwargs["classifier_version"] == "2.1.0"


def test_scribe_records_use_the_step_version():
    step = build_step(MODEL_VERSION="2.1.0")

    records = step._format_scribe_records([prediction()])

    assert {r["classifier_version"] for r in records} == {210}
