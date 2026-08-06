import unittest

from lc_classifier.features.core.base import discard_bogus_detections


def forced(measurement_id, procstatus):
    return {"measurement_id": measurement_id, "forced": True, "procstatus": procstatus}


def detection(measurement_id, rb):
    return {"measurement_id": measurement_id, "forced": False, "rb": rb}


class TestDiscardBogusDetections(unittest.TestCase):
    def test_str_procstatus(self):
        epochs = [forced(1, "0"), forced(2, "57"), forced(3, "2")]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1, 2], kept)

    def test_int_procstatus_is_coerced(self):
        epochs = [forced(1, 0), forced(2, 57), forced(3, 2)]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1, 2], kept)

    def test_missing_procstatus_is_kept(self):
        kept = discard_bogus_detections([{"measurement_id": 1, "forced": True}])
        self.assertEqual(1, len(kept))

    def test_low_rb_detection_is_dropped(self):
        epochs = [detection(1, 0.9), detection(2, 0.1)]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1], kept)

    def test_rb_is_only_applied_to_non_forced_rows(self):
        epochs = [{"measurement_id": 1, "forced": True, "rb": 0.1, "procstatus": "0"}]
        self.assertEqual(1, len(discard_bogus_detections(epochs)))

    def test_extra_fields_shape_still_supported(self):
        epochs = [
            {"measurement_id": 1, "forced": False, "extra_fields": {"rb": 0.9}},
            {"measurement_id": 2, "forced": False, "extra_fields": {"rb": 0.1}},
        ]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1], kept)
