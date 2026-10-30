"""The step drops alerts the model returned no probability for."""  # module docstring
import logging  # warning capture
import os  # schema path helper
from unittest import mock  # patching

import pytest  # fixtures

from alerce_classifiers.base.dto import OutputDTO  # model output container
from rubin_stamp_classifier_step.step import StampClassifierStep  # step under test
from tests.unit.stub_model import StubModel  # base stub
from tests.unit.test_ss_object_handling import OUTPUT_SCHEMA_PATH, TAXONOMY, processed  # reused helpers

SKIPPED_OID = 2  # the oid the stub model leaves out


class SkippingStubModel(StubModel):  # stub that omits one oid
    """A stub model that returns no probability row for SKIPPED_OID."""

    def predict(self, input_dto) -> OutputDTO:  # same signature as the stub
        """Predict like StubModel, then drop the SKIPPED_OID row."""
        out = super().predict(input_dto)  # full frame from the base stub
        kept = out.probabilities.drop(index=SKIPPED_OID, errors="ignore")  # remove the row
        return OutputDTO(probabilities=kept, hierarchical=None)  # filtered output


@pytest.fixture  # pytest fixture, rebuilt per test
def step():  # step wired to the skipping stub
    """A StampClassifierStep whose model skips SKIPPED_OID, with no DB or Kafka."""
    config = {  # step configuration
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},  # no real consumer
        "PRODUCER_CONFIG": {  # recording producer
            "CLASS": "tests.unit.recording_producer.RecordingProducer",  # test producer
            "SCHEMA_PATH": OUTPUT_SCHEMA_PATH,  # output schema
        },  # end of the config section
        "DB_CONFIG": {},  # unused, connection is patched
        "MODEL_CONFIG": {  # model selection
            "CLASS": "tests.unit.test_skipped_alerts.SkippingStubModel",  # skipping stub
            "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip"},  # fake artifact
            "CLS_ID": 3,  # classifier id
        },  # end of the config section
    }  # end of the config
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):  # patch DB
        yield StampClassifierStep(config=config)  # build the step


def test_execute_emits_no_message_for_an_oid_the_model_skipped(step, caplog):  # skipped oid emits nothing
    """The skipped oid yields no output, order is kept and a warning names it."""
    with caplog.at_level(logging.WARNING):  # capture warnings
        out = step.execute([processed(1, 1), processed(SKIPPED_OID, 1), processed(3, 1)])  # run
    assert [m["oid"] for m in out] == [1, 3]  # order kept, skipped oid gone
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]  # warning records only
    assert len(warnings) == 1  # exactly one warning per batch
    assert f"[{SKIPPED_OID}]" in warnings[0].message  # the skipped oid list is in it


def test_post_execute_only_sees_the_kept_messages(step):  # post_execute sees kept messages
    """store_probability receives only the messages execute kept."""
    out = step.execute([processed(1, 1), processed(SKIPPED_OID, 1)])  # run
    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:  # spy on DB write
        store.side_effect = lambda *a, **kw: {m['oid'] for m in kw['predictions']}  # every oid counts as written
        step.post_execute(out)  # persist
    assert [p["oid"] for p in store.call_args.kwargs["predictions"]] == [1]  # only the kept oid


@pytest.fixture  # pytest fixture, rebuilt per test
def forwarding_step():  # step with the SN forwarder enabled
    """A step like `step` but with a recording SN forwarder producer."""
    config = {  # step configuration
        "CONSUMER_CONFIG": {"CLASS": "apf.core.step.DefaultConsumer"},  # no real consumer
        "PRODUCER_CONFIG": {  # recording producer
            "CLASS": "tests.unit.recording_producer.RecordingProducer",  # test producer
            "SCHEMA_PATH": OUTPUT_SCHEMA_PATH,  # output schema
        },  # end of the config section
        "SN_FORWARD_PRODUCER_CONFIG": {  # enables the forwarder
            "CLASS": "tests.unit.recording_producer.RecordingProducer",  # records forwarded alerts
            "TOPIC": "sn_candidates",  # forward topic name
        },  # end of the config section
        "DB_CONFIG": {},  # unused, connection is patched
        "MODEL_CONFIG": {  # model selection
            "CLASS": "tests.unit.test_skipped_alerts.SkippingStubModel",  # skipping stub
            "PARAMS": {"model_path": "https://example.org/1.0.0/model.zip"},  # fake artifact
            "CLS_ID": 3,  # classifier id
        },  # end of the config section
    }  # end of the config
    with mock.patch("rubin_stamp_classifier_step.step.PSQLConnection"), \
         mock.patch("rubin_stamp_classifier_step.step.get_taxonomy_by_classifier_id", return_value=TAXONOMY):  # patch DB
        yield StampClassifierStep(config=config)  # build the step


def test_post_execute_forwards_only_the_kept_alerts(forwarding_step):  # only kept alerts forwarded
    """The SN forwarder emits the kept oid's alert and nothing for the skipped oid."""
    kept, skipped = processed(1, 1), processed(SKIPPED_OID, 1)  # one kept, one skipped message
    kept["alert"], skipped["alert"] = {"tag": "kept"}, {"tag": "skipped"}  # distinguishable alerts
    out = forwarding_step.execute([kept, skipped])  # run
    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:  # spy on DB write
        store.side_effect = lambda *a, **kw: {m['oid'] for m in kw['predictions']}  # every oid counts as written
        forwarding_step.post_execute(out)  # persist and forward
    assert [p["oid"] for p in store.call_args.kwargs["predictions"]] == [1]  # DB sees only the kept oid
    produced = forwarding_step.sn_forward_producer.produced  # what the forwarder emitted
    assert [m for m, _ in produced] == [{"tag": "kept"}]  # exactly the kept alert, none for the skipped oid
