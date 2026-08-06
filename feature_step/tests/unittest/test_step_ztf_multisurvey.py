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


class ParserTestCase(unittest.TestCase):
    def test_forced_epochs_keep_their_corrected_magnitude(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        forced_mag = ao.forced_photometry[ao.forced_photometry["unit"] == "magnitude"]
        self.assertEqual(3, len(forced_mag))
        self.assertTrue(forced_mag["brightness"].notna().all())
        self.assertTrue(forced_mag["e_brightness"].notna().all())

        expected = sorted(f["mag_corr"] for f in message["forced_photometries"])
        self.assertEqual(expected, sorted(forced_mag["brightness"].tolist()))

    def test_detections_keep_their_corrected_magnitude(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertEqual(6, len(det_mag))
        self.assertTrue(det_mag["brightness"].notna().all())

        expected = sorted(
            d["magpsf_corr"]
            for d in message["detections"] + message["previous_detections"]
        )
        self.assertEqual(expected, sorted(det_mag["brightness"].tolist()))

    def test_previous_detections_reach_the_astro_object(self):
        message = generate_message(n_detections=3, n_previous_detections=4, n_forced=0)
        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        got = set(det_mag["candid"].tolist())
        for epoch in message["previous_detections"]:
            self.assertIn(epoch["measurement_id"], got)

    def test_forced_rows_keep_distnr_rfid_and_procstatus(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        forced_mag = ao.forced_photometry[ao.forced_photometry["unit"] == "magnitude"]
        self.assertTrue(forced_mag["distnr"].notna().all())
        self.assertTrue(forced_mag["rfid"].notna().all())
        self.assertEqual({"0"}, set(forced_mag["procstatus"].unique()))

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertTrue(det_mag["rb"].notna().all())
        self.assertTrue(det_mag["distnr"].notna().all())

    def test_i_band_epochs_are_labelled_not_nan(self):
        # ZTF i-band is rare but real; the band map must keep its `i` entry so
        # those rows are labelled rather than becoming NaN.
        rng = random.Random(3)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, 1, "g", 60000.0, rng),
            candidate(oid, 2, "r", 60001.0, rng),
            candidate(oid, 3, "i", 60002.0, rng),
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = []

        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertEqual({"g", "r", "i"}, set(det_mag["fid"]))

    def test_forced_argument_must_be_empty(self):
        message = generate_message()
        prepared = build_step().pre_execute([message])[0]
        epochs = [
            {**e, "aid": e["oid"], "index_column": "x"} for e in prepared["detections"]
        ]

        with self.assertRaises(NotImplementedError):
            detections_to_astro_object(epochs, [epochs[0]], None, None)

    def test_uncorrected_epochs_get_nan_brightness_not_a_neighbours_value(self):
        # Both spellings are nullable. A row that populates neither must end up
        # NaN -- never silently filled from the other column or another row.
        rng = random.Random(9)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, i, "g", 60000.0 + i, rng) for i in range(1, 5)
        ]
        message["detections"][0]["magpsf_corr"] = None
        message["detections"][0]["sigmapsf_corr_ext"] = None
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 100 + i, "g", 60010.0 + i, rng) for i in range(2)
        ]
        message["forced_photometries"][0]["mag_corr"] = None
        message["forced_photometries"][0]["e_mag_corr_ext"] = None

        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        forced_mag = ao.forced_photometry[ao.forced_photometry["unit"] == "magnitude"]

        self.assertEqual([1], det_mag[det_mag["brightness"].isna()]["candid"].tolist())
        self.assertEqual(
            [100], forced_mag[forced_mag["brightness"].isna()]["candid"].tolist()
        )
        self.assertEqual(
            [2, 3, 4], sorted(det_mag[det_mag["brightness"].notna()]["candid"])
        )
        self.assertEqual(
            [101], sorted(forced_mag[forced_mag["brightness"].notna()]["candid"])
        )

    def test_int_procstatus_survives_the_bogus_flag_frame(self):
        # `procstatus` is re-checked by the preprocessor after passing through a
        # DataFrame. A column mixing ints with None becomes float64, and 0 would
        # stringify to "0.0" — dropping every forced epoch.
        rng = random.Random(4)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, i, "g", 60000.0 + i, rng) for i in range(1, 6)
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 100 + i, "g", 60010.0 + i, rng, procstatus=0)
            for i in range(3)
        ]

        ao = astro_object_from(message)
        self.assertEqual({"0"}, set(ao.forced_photometry["procstatus"]))

        ZTFLightcurvePreprocessor(drop_bogus=True).preprocess_single_object(ao)
        self.assertEqual(6, len(ao.forced_photometry))

    def test_int_procstatus_outside_the_allowed_set_is_still_dropped(self):
        rng = random.Random(5)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, i, "g", 60000.0 + i, rng) for i in range(1, 6)
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 100 + i, "g", 60010.0 + i, rng, procstatus=2)
            for i in range(3)
        ]

        ao = astro_object_from(message)
        self.assertEqual(0, len(ao.forced_photometry))


if __name__ == "__main__":
    unittest.main()
