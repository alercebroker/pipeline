"""The step, end to end, against a local broker and a local seeded database.

Everything except the model is faked: the messages are generated from the
feature_step schema (`fake_features`) and the taxonomy is the local seed
(`taxonomy_seed`). The model is real, which is why these tests need `MODEL_PATH`
and skip without it.

What this covers that the unit suite cannot: the apf consumer/producer wiring,
the §8 startup queries against a real Postgres, and the scribe envelope as it
lands on a topic. The probabilities themselves are meaningless -- the features
are random -- so nothing here asserts on their values, only on their shape,
their count, and the identifiers they were written against.
"""
import json
import os

import pytest

from lc_classification_multisurvey_step.probabilities import (
    classifier_version_to_smallint,
)
from lc_classification_multisurvey_step.step import LateClassifierMultisurvey

from . import taxonomy_seed

pytestmark = pytest.mark.skipif(
    not os.getenv("MODEL_PATH"),
    reason=(
        "MODEL_PATH is unset; these tests run the real BHRF 2.1.0 model, which is "
        "not in this repo. Point it at the directory holding "
        "hierarchical_random_forest_model.pkl."
    ),
)

MESSAGE_COUNT = 5


def _consume_commands(scribe_consumer) -> list:
    """Every scribe command on the topic, decoded.

    `consume()` yields a *batch* — a list — whenever `consume.messages` is above
    1, and a bare message otherwise, so both shapes are handled rather than
    assuming the one this consumer happens to be configured for.
    """
    commands = []
    for batch in scribe_consumer(num_messages=1000).consume():
        messages = batch if isinstance(batch, list) else [batch]
        commands.extend(json.loads(message["payload"]) for message in messages)
    return commands


def test_step_writes_the_full_batch_to_the_scribe(
    kafka_service, psql_service, produce_features, scribe_consumer, step_config
):
    """One command per class per oid, carrying the seeded identifiers.

    Deliberately one test rather than several: constructing the step loads the
    1.6 GB pickle, so every extra test that starts one costs another load. The
    §8 startup assertions are unit-tested against a fake session in
    `tests/unittest/test_taxonomy.py` and do not need a database to prove.
    """
    messages = produce_features(MESSAGE_COUNT)

    LateClassifierMultisurvey(config=step_config()).start()

    commands = _consume_commands(scribe_consumer)

    # Every class of all five heads, for every oid: nothing was dropped. A head
    # missing from the seed would take exactly its own classes out of this count.
    assert len(commands) == MESSAGE_COUNT * taxonomy_seed.CLASSES_PER_OID

    expected_oids = {int(message["oid"]) for message in messages}
    assert {command["payload"]["oid"] for command in commands} == expected_oids

    assert {command["step"] for command in commands} == {"update-probability"}
    assert {command["survey"] for command in commands} == {"ztf"}

    payloads = [command["payload"] for command in commands]

    seeded_ids = set(taxonomy_seed.classifier_ids().values())
    assert {payload["classifier_id"] for payload in payloads} == seeded_ids

    assert {payload["classifier_version"] for payload in payloads} == {
        classifier_version_to_smallint(taxonomy_seed.CLASSIFIER_VERSION)
    }
    assert {payload["sid"] for payload in payloads} == {0}

    # `lastmjd` is the max detection mjd of the winning message, and
    # `fake_features` lays those out per oid, so it is exact rather than a range.
    expected_lastmjd = {
        int(message["oid"]): max(d["mjd"] for d in message["detections"])
        for message in messages
    }
    assert all(
        payload["lastmjd"] == expected_lastmjd[payload["oid"]] for payload in payloads
    )

    # Ranking is dense over the classes of one head, so it starts at 1 and never
    # exceeds the largest head.
    rankings = [payload["ranking"] for payload in payloads]
    assert min(rankings) == 1
    assert max(rankings) <= max(
        len(classes) for classes in taxonomy_seed.HEAD_CLASSES.values()
    )

    assert all(0.0 <= payload["probability"] <= 1.0 for payload in payloads)


def test_messages_without_features_produce_nothing(
    kafka_service, psql_service, produce_features, scribe_consumer, step_config
):
    """`filter_messages` drops them, so the batch never reaches the model."""
    produce_features(MESSAGE_COUNT, with_features=False)

    LateClassifierMultisurvey(config=step_config()).start()

    assert _consume_commands(scribe_consumer) == []


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_step_refuses_to_start_when_the_taxonomy_is_unseeded(
    kafka_service, psql_service, step_config
):
    """§8 assertion 2: an unknown classifier name is a deploy error, not a warning.

    The constructor raises after `GenericStep.__init__` has already built the
    consumer, so that consumer is never torn down and its finalizer complains
    during interpreter shutdown. That is the step's own construction order, not
    something this test can hold differently -- there is no object to close.
    """
    config = step_config()
    config["MODEL_CONFIG"]["CLASSIFIER_NAME"] = "lc_classifier_not_seeded"

    with pytest.raises(ValueError, match="classifier table has no row for"):
        LateClassifierMultisurvey(config=config)
