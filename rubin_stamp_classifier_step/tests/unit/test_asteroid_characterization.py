"""Characterization tests for how the step handles ssObject (asteroid) alerts.

These pin the behaviour that exists today, before the asteroid rule moves
from the step into the model. They run without a database, a Kafka broker,
or a downloaded model: the model is a stub that returns fixed probabilities
for whatever diaObjectIds it receives.
"""
import io
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from alerce_classifiers.base.dto import OutputDTO
from rubin_stamp_classifier_step.db.db import _format_data
from rubin_stamp_classifier_step.step import StampClassifierStep

CLASSES = ["AGN", "SN", "VS", "asteroid", "bogus"]
TAXONOMY = {name: idx + 10 for idx, name in enumerate(CLASSES)}
MODEL_VERSION = "1.0.0"
CLS_ID = 3


class StubModel:
    """Returns the same probability row for every diaObjectId it is asked about."""

    ROW = {"AGN": 0.1, "SN": 0.6, "VS": 0.1, "asteroid": 0.1, "bogus": 0.1}

    def __init__(self, *args, **kwargs):
        self.dict_mapping_classes = dict(enumerate(CLASSES))
        self.model_version = MODEL_VERSION
        self.predict_calls = []

    def predict(self, input_dto) -> OutputDTO:
        index = input_dto.stamps.index
        self.predict_calls.append(list(index))
        probs = pd.DataFrame([self.ROW] * len(index), index=index, columns=CLASSES)
        return OutputDTO(probabilities=probs, hierarchical=None)


@pytest.fixture
def step():
    config = {
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},
        "PRODUCER_CONFIG": {"CLASS": "apf.core.step.DefaultProducer"},
        "DB_CONFIG": {},
        "MODEL_CONFIG": {"MODEL_PATH": "https://example.org/1.0.0/model.zip", "CLS_ID": CLS_ID},
    }
    with mock.patch("rubin_stamp_classifier_step.step.StampClassifierModel", StubModel), \
         mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
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


def processed(dia_object_id, ss_object_id, dia_source_id=500):
    """A message as pre_execute emits it, without the stamps."""
    return {
        "diaObjectId": dia_object_id,
        "ssObjectId": ss_object_id,
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
    }


# --- pre_execute: which alerts survive and what identity they carry ---


@pytest.mark.parametrize("dia_object_id", [None, 0])
def test_pre_execute_keeps_ss_only_alert_with_empty_dia_object_id(step, dia_object_id):
    out = step.pre_execute([alert(dia_object_id, 777)])

    assert len(out) == 1
    assert out[0]["diaObjectId"] == dia_object_id
    assert out[0]["ssObjectId"] == 777
    assert isinstance(out[0]["visit_image"], np.ndarray)


def test_pre_execute_drops_alert_with_both_ids_set(step):
    assert step.pre_execute([alert(123, 777)]) == []


@pytest.mark.parametrize("ids", [(None, None), (0, 0), (None, 0), (0, None)])
def test_pre_execute_drops_alert_with_neither_id(step, ids):
    assert step.pre_execute([alert(*ids)]) == []


# --- execute: the hardcoded asteroid row ---


def test_execute_gives_asteroid_full_probability_without_the_model(step):
    out = step.execute([processed(0, 777)])

    assert out == [
        {
            "diaObjectId": 0,
            "ssObjectId": 777,
            "diaSourceId": 500,
            "probabilities": {"AGN": 0.0, "SN": 0.0, "VS": 0.0, "asteroid": 1.0, "bogus": 0.0},
            "midpointMjdTai": 60000.5,
            "ra": 10.0,
            "dec": -20.0,
        }
    ]
    assert step.model.predict_calls == []


def test_execute_treats_none_dia_object_id_as_asteroid(step):
    out = step.execute([processed(None, 777)])

    assert out[0]["diaObjectId"] == 0
    assert out[0]["ssObjectId"] == 777
    assert out[0]["probabilities"]["asteroid"] == 1.0


def test_execute_sends_only_dia_objects_to_the_model(step):
    step.execute([processed(0, 777, dia_source_id=1), processed(123, 0, dia_source_id=2)])

    assert step.model.predict_calls == [[123]]


def test_execute_orders_model_rows_before_asteroid_rows(step):
    out = step.execute([processed(0, 777, dia_source_id=1), processed(123, 0, dia_source_id=2)])

    assert [m["diaSourceId"] for m in out] == [2, 1]
    assert out[0] == {
        "diaObjectId": 123,
        "ssObjectId": 0,
        "diaSourceId": 2,
        "probabilities": StubModel.ROW,
        "midpointMjdTai": 60000.5,
        "ra": 10.0,
        "dec": -20.0,
    }


def test_execute_zeroes_ss_object_id_on_model_rows(step):
    out = step.execute([processed(123, None)])

    assert out[0]["ssObjectId"] == 0


# --- db formatter: oid and sid for asteroid predictions ---


def asteroid_prediction():
    return {
        "diaObjectId": 0,
        "ssObjectId": 777,
        "diaSourceId": 500,
        "probabilities": {"AGN": 0.0, "SN": 0.0, "VS": 0.0, "asteroid": 1.0, "bogus": 0.0},
        "midpointMjdTai": 60000.5,
        "ra": 10.0,
        "dec": -20.0,
    }


def test_db_rows_for_asteroid_use_ss_object_id_and_sid_2():
    rows = _format_data(CLS_ID, MODEL_VERSION, TAXONOMY, [asteroid_prediction()])

    assert len(rows) == len(CLASSES)
    assert {r["oid"] for r in rows} == {777}
    assert {r["sid"] for r in rows} == {2}
    assert {r["classifier_id"] for r in rows} == {CLS_ID}
    assert {r["classifier_version"] for r in rows} == {100}
    assert {r["lastmjd"] for r in rows} == {60000.5}


def test_db_rows_for_asteroid_rank_asteroid_first_then_zero_classes_in_declaration_order():
    rows = _format_data(CLS_ID, MODEL_VERSION, TAXONOMY, [asteroid_prediction()])

    by_rank = {r["ranking"]: (r["class_id"], r["probability"]) for r in rows}
    assert by_rank == {
        1: (TAXONOMY["asteroid"], 1.0),
        2: (TAXONOMY["AGN"], 0.0),
        3: (TAXONOMY["SN"], 0.0),
        4: (TAXONOMY["VS"], 0.0),
        5: (TAXONOMY["bogus"], 0.0),
    }


def test_db_rows_for_dia_object_use_dia_object_id_and_sid_1():
    prediction = {**asteroid_prediction(), "diaObjectId": 123, "ssObjectId": 0}
    rows = _format_data(CLS_ID, MODEL_VERSION, TAXONOMY, [prediction])

    assert {r["oid"] for r in rows} == {123}
    assert {r["sid"] for r in rows} == {1}


# --- scribe formatter: same identity rules as the db formatter ---


def test_scribe_records_for_asteroid_use_ss_object_id_and_sid_2(step):
    records = step._format_scribe_records([asteroid_prediction()])

    assert len(records) == len(CLASSES)
    assert {r["oid"] for r in records} == {777}
    assert {r["sid"] for r in records} == {2}
    assert {r["classifier_id"] for r in records} == {CLS_ID}
    assert {r["classifier_version"] for r in records} == {100}
    by_rank = {r["ranking"]: (r["class_id"], r["probability"]) for r in records}
    assert by_rank[1] == (TAXONOMY["asteroid"], 1.0)


def test_scribe_records_for_dia_object_use_dia_object_id_and_sid_1(step):
    prediction = {**asteroid_prediction(), "diaObjectId": 123, "ssObjectId": 0}
    records = step._format_scribe_records([prediction])

    assert {r["oid"] for r in records} == {123}
    assert {r["sid"] for r in records} == {1}
