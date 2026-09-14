"""How the step handles ssObject (solar system) alerts.

The step resolves LSST identity once per alert (oid, sid) and hands every
alert to the model. It knows nothing about asteroids: the asteroid rule lives
in the rubin model (see alerce_classifiers tests). These tests run without a
database, a Kafka broker, or a downloaded model: the model is a stub that
returns fixed probabilities for whatever rows it receives.
"""
import io
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from rubin_stamp_classifier_step.db.db import _format_data
from rubin_stamp_classifier_step.step import StampClassifierStep
from tests.unit.stub_model import CLASSES, MODEL_VERSION, StubModel

TAXONOMY = {name: idx + 10 for idx, name in enumerate(CLASSES)}
CLS_ID = 3
OUTPUT_SCHEMA_FIELDS = {
    "diaObjectId", "ssObjectId", "diaSourceId", "probabilities", "midpointMjdTai", "ra", "dec",
}


@pytest.fixture
def step():
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": {},
        "MODEL_CONFIG": {
            "CLASS": "tests.unit.stub_model.StubModel",
            "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip"},
            "CLS_ID": CLS_ID,
        },
    }
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):
        yield StampClassifierStep(config=config)


def fits_bytes() -> bytes:
    buf = io.BytesIO()
    fits.PrimaryHDU(np.ones((3, 3), dtype=np.float32)).writeto(buf)
    return buf.getvalue()


def alert(dia_object_id, ss_object_id, dia_source_id=500):
    return {
        "diaSource": {
            "diaObjectId": dia_object_id,
            "ssObjectId": ss_object_id,
            "diaSourceId": dia_source_id,
            "midpointMjdTai": 60000.5,
            "ra": 10.0,
            "dec": -20.0,
            "psfFlux": 1.0,
            "psfFluxErr": 0.1,
            "scienceFlux": 2.0,
            "scienceFluxErr": 0.2,
            "snr": 5.0,
        },
        "cutoutScience": fits_bytes(),
        "cutoutDifference": fits_bytes(),
        "cutoutTemplate": fits_bytes(),
    }


def processed(oid, sid, dia_source_id=500):
    """A message as pre_execute emits it."""
    return {
        "diaObjectId": oid if sid == 1 else None,
        "ssObjectId": oid if sid == 2 else None,
        "oid": oid,
        "sid": sid,
        "diaSourceId": dia_source_id,
        "midpointMjdTai": 60000.5,
        "ra": 10.0,
        "dec": -20.0,
        "airmass": 1.0,
        "magLim": 25.0,
        "psfFlux": 1.0,
        "psfFluxErr": 0.1,
        "scienceFlux": 2.0,
        "scienceFluxErr": 0.2,
        "seeing": 0.7,
        "snr": 5.0,
        "visit_image": np.ones((3, 3)),
        "difference_image": np.ones((3, 3)),
        "reference_image": np.ones((3, 3)),
        # The raw alert rides along for the SN forwarder; irrelevant here.
        "alert": None,
    }


# --- pre_execute: which alerts survive and what identity they carry ---


@pytest.mark.parametrize("dia_object_id", [None, 0])
def test_pre_execute_resolves_ss_only_alert_as_sid_2(step, dia_object_id):
    out = step.pre_execute([alert(dia_object_id, 777)])

    assert len(out) == 1
    assert out[0]["oid"] == 777
    assert out[0]["sid"] == 2
    assert out[0]["ssObjectId"] == 777
    assert isinstance(out[0]["visit_image"], np.ndarray)


@pytest.mark.parametrize("ss_object_id", [None, 0])
def test_pre_execute_resolves_dia_only_alert_as_sid_1(step, ss_object_id):
    out = step.pre_execute([alert(123, ss_object_id)])

    assert len(out) == 1
    assert out[0]["oid"] == 123
    assert out[0]["sid"] == 1
    assert out[0]["diaObjectId"] == 123


def test_pre_execute_drops_alert_with_both_ids_set(step):
    assert step.pre_execute([alert(123, 777)]) == []


@pytest.mark.parametrize("ids", [(None, None), (0, 0), (None, 0), (0, None)])
def test_pre_execute_drops_alert_with_neither_id(step, ids):
    assert step.pre_execute([alert(*ids)]) == []


# --- execute: every alert goes to the model, identity comes back out ---


def test_execute_hands_every_alert_to_the_model_with_its_sid(step):
    step.execute([processed(777, 2, dia_source_id=1), processed(123, 1, dia_source_id=2)])

    assert step.model.calls == [[(777, 2), (123, 1)]]


def test_execute_output_for_ss_object_carries_model_row_and_identity(step):
    out = step.execute([processed(777, 2)])

    assert out == [
        {
            "oid": 777,
            "sid": 2,
            "diaObjectId": 0,
            "ssObjectId": 777,
            "diaSourceId": 500,
            "probabilities": StubModel.ROW,
            "midpointMjdTai": 60000.5,
            "ra": 10.0,
            "dec": -20.0,
            "alert": None,
        }
    ]


def test_execute_output_for_dia_object_zeroes_ss_object_id(step):
    out = step.execute([processed(123, 1)])

    assert out[0]["oid"] == 123
    assert out[0]["sid"] == 1
    assert out[0]["diaObjectId"] == 123
    assert out[0]["ssObjectId"] == 0


def test_execute_keeps_input_order(step):
    out = step.execute([processed(777, 2, dia_source_id=1), processed(123, 1, dia_source_id=2)])

    assert [m["diaSourceId"] for m in out] == [1, 2]


def test_execute_emits_one_output_per_message_but_predicts_each_oid_once(step):
    out = step.execute([processed(123, 1, dia_source_id=1), processed(123, 1, dia_source_id=2)])

    assert step.model.calls == [[(123, 1)]]
    assert [m["diaSourceId"] for m in out] == [1, 2]


# --- pre_produce: only the output schema fields reach the topic ---


def test_pre_produce_strips_internal_identity_fields(step):
    out = step.pre_produce(step.execute([processed(777, 2)]))

    assert set(out[0]) == OUTPUT_SCHEMA_FIELDS


# --- db formatter: oid and sid come from the message ---


def prediction(oid, sid):
    # The formatters must read oid and sid only, so the raw id fields are
    # deliberately uninformative here.
    return {
        "oid": oid,
        "sid": sid,
        "diaObjectId": 0,
        "ssObjectId": 0,
        "diaSourceId": 500,
        "probabilities": {"AGN": 0.0, "SN": 0.0, "VS": 0.0, "asteroid": 1.0, "bogus": 0.0},
        "midpointMjdTai": 60000.5,
        "ra": 10.0,
        "dec": -20.0,
    }


def test_db_rows_take_oid_and_sid_from_the_message():
    rows = _format_data(CLS_ID, MODEL_VERSION, TAXONOMY, [prediction(777, 2), prediction(123, 1)])

    assert len(rows) == 2 * len(CLASSES)
    assert {(r["oid"], r["sid"]) for r in rows} == {(777, 2), (123, 1)}
    assert {r["classifier_id"] for r in rows} == {CLS_ID}
    assert {r["classifier_version"] for r in rows} == {100}
    assert {r["lastmjd"] for r in rows} == {60000.5}


def test_db_rows_rank_asteroid_first_then_zero_classes_in_declaration_order():
    rows = _format_data(CLS_ID, MODEL_VERSION, TAXONOMY, [prediction(777, 2)])

    by_rank = {r["ranking"]: (r["class_id"], r["probability"]) for r in rows}
    assert by_rank == {
        1: (TAXONOMY["asteroid"], 1.0),
        2: (TAXONOMY["AGN"], 0.0),
        3: (TAXONOMY["SN"], 0.0),
        4: (TAXONOMY["VS"], 0.0),
        5: (TAXONOMY["bogus"], 0.0),
    }


# --- scribe formatter: same identity rules as the db formatter ---


def test_scribe_records_take_oid_and_sid_from_the_message(step):
    records = step._format_scribe_records([prediction(777, 2), prediction(123, 1)])

    assert len(records) == 2 * len(CLASSES)
    assert {(r["oid"], r["sid"]) for r in records} == {(777, 2), (123, 1)}
    assert {r["classifier_id"] for r in records} == {CLS_ID}
    assert {r["classifier_version"] for r in records} == {100}
    by_rank = {r["ranking"]: (r["class_id"], r["probability"]) for r in records if r["oid"] == 777}
    assert by_rank[1] == (TAXONOMY["asteroid"], 1.0)
