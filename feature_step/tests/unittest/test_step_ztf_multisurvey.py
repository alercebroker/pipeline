"""ZTF multisurvey path: `magstats_ms_ztf` message -> features.

Covers the flat, three-array input contract (no `extra_fields`) that
`schemas/magstats_ms_step/ztf/output.avsc` defines.
"""
import logging
import random
import unittest
from unittest import mock

import pandas as pd

from features.step import FeatureStep
from features.utils.parsers import (
    detections_to_astro_object,
    prepare_ao_features_for_db,
)
from lc_classifier.features.core.base import query_ao_table
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

from ..message_factory_ztf_ms import (
    allwise_match,
    candidate,
    forced_photometry,
    generate_message,
)

CONSUMER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "PARAMS": {"bootstrap.servers": "server", "group.id": "group_id"},
    "TOPICS": ["topic"],
}
PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"}
SCRIBE_PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test-scribe"}


def build_step(**extra_config):
    config = {
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        "SURVEY": "ztf",
        **extra_config,
    }
    step = FeatureStep(config=config, db_sql=mock.MagicMock())
    # `produce_to_scribe` reaches through to `scribe_producer.producer.produce`,
    # so a plain MagicMock (not an autospec of GenericProducer) is what fits.
    step.scribe_producer = mock.MagicMock()
    # The mocked db_sql makes these two MagicMocks; pin them so scribe commands
    # stay JSON-serializable.
    step.extractor_version = 1
    step.feature_name_lut = {}
    return step


def astro_object_from(message, xmatches=None, references_db=None):
    """pre_execute + execute packing, without Kafka."""
    step = build_step()
    prepared = step.pre_execute([message])[0]
    epochs = [
        {**e, "aid": e["oid"], "index_column": f'{e["measurement_id"]}_{e["oid"]}'}
        for e in prepared["detections"]
    ]
    return detections_to_astro_object(epochs, [], xmatches, references_db)


class PreExecuteTestCase(unittest.TestCase):
    def test_three_arrays_are_merged_into_detections(self):
        message = generate_message(
            n_detections=6, n_previous_detections=4, n_forced=5
        )
        result = build_step().pre_execute([message])[0]

        self.assertEqual(15, len(result["detections"]))
        measurement_ids = {d["measurement_id"] for d in result["detections"]}
        for source in ("detections", "previous_detections", "forced_photometries"):
            for epoch in message[source]:
                self.assertIn(epoch["measurement_id"], measurement_ids)

    def test_missing_forced_photometries_is_not_none(self):
        message = generate_message()
        message["forced_photometries"] = None
        result = build_step().pre_execute([message])[0]

        self.assertEqual(10, len(result["detections"]))

        del message["forced_photometries"]
        result = build_step().pre_execute([message])[0]
        self.assertEqual(10, len(result["detections"]))

    def test_forced_rows_outside_allowed_procstatus_are_dropped(self):
        rng = random.Random(1)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [candidate(oid, 1, "g", 60000.0, rng, rb=0.9)]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 10, "g", 60001.0, rng, procstatus="0"),
            forced_photometry(oid, 11, "g", 60002.0, rng, procstatus="57"),
            forced_photometry(oid, 12, "g", 60003.0, rng, procstatus="2"),
            forced_photometry(oid, 13, "g", 60004.0, rng, procstatus=0),
            forced_photometry(oid, 14, "g", 60005.0, rng, procstatus=2),
        ]

        result = build_step().pre_execute([message])[0]

        kept = [d["measurement_id"] for d in result["detections"]]
        self.assertEqual([1, 10, 11, 13], sorted(kept))

    def test_low_rb_detections_are_dropped(self):
        rng = random.Random(2)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, 1, "g", 60000.0, rng, rb=0.9),
            candidate(oid, 2, "g", 60001.0, rng, rb=0.1),
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = []

        result = build_step().pre_execute([message])[0]

        self.assertEqual([1], [d["measurement_id"] for d in result["detections"]])

    def test_min_detections_counts_only_non_forced_rows(self):
        message = generate_message(
            n_detections=1, n_previous_detections=0, n_forced=8
        )
        step = build_step(MIN_DETECTIONS_FEATURES=2)

        self.assertEqual(0, len(step.pre_execute([message])))

        message = generate_message(
            n_detections=2, n_previous_detections=0, n_forced=0
        )
        self.assertEqual(1, len(step.pre_execute([message])))


if __name__ == "__main__":
    unittest.main()
