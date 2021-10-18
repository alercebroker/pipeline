import unittest
from db_plugins.db.mongo import models


class MongoModelsTest(unittest.TestCase):
    def test_object_creates(self):
        o = models.Object(
            aid="aid",
            sid="sid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra="meanra",
            meandec="meandec",
        )
        self.assertIsInstance(o, models.Object)
        self.assertIsInstance(o, dict)
        self.assertEqual(
            o["loc"], {"type": "Point", "coordinates": ["meanra", "meandec"]}
        )
        self.assertEqual(
            o._meta.tablename,
            models.Object.__tablename__,
        )

    def test_object_fails_creation(self):
        with self.assertRaises(AttributeError) as e:
            models.Object()
        self.assertEqual(str(e.exception), "Object model needs aid attribute")

    def test_object_with_extra_fields(self):
        o = models.Object(
            aid="aid",
            sid="sid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra="meanra",
            meandec="meandec",
            extra="extra",
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
        o = models.Object(
            aid="aid",
            sid="sid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            meanra="meanra",
            meandec="meandec",
            extra_fields={"extra": "extra"},
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})

    def test_detection_creates(self):
        d = models.Detection(
            aid="aid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            rb="rb",
            mag="mag",
            sigmag="sigmag",
        )
        self.assertIsInstance(d, models.Detection)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["aid"], "aid")

    def test_detection_fails_creation(self):
        with self.assertRaises(AttributeError) as e:
            models.Detection()
        self.assertEqual(str(e.exception), "Detection model needs aid attribute")

    def test_detection_with_extra_fields(self):
        o = models.Detection(
            aid="aid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            rb="rb",
            mag="mag",
            sigmag="sigmag",
            extra="extra",
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
        o = models.Detection(
            aid="aid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            rb="rb",
            mag="mag",
            sigmag="sigmag",
            extra_fields={"extra": "extra"},
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})

    def test_non_detection_creates(self):
        o = models.NonDetection(
            aid="aid",
            sid="sid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
        )
        self.assertIsInstance(o, models.NonDetection)
        self.assertIsInstance(o, dict)
        self.assertEqual(o["aid"], "aid")

    def test_non_detection_fails_creation(self):
        with self.assertRaises(AttributeError) as e:
            models.NonDetection()
        self.assertEqual(str(e.exception), "NonDetection model needs aid attribute")

    def test_non_detection_with_extra_fields(self):
        o = models.NonDetection(
            aid="aid",
            sid="sid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
            extra="extra",
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
        o = models.NonDetection(
            aid="aid",
            sid="sid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
            extra_fields={"extra": "extra"},
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
