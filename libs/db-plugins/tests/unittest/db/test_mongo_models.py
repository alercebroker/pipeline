import unittest
from db_plugins.db.mongo import models


class MongoModelsTest(unittest.TestCase):
    def test_object_creates(self):
        o = models.Object(
            aid="aid",
            oid="oid",
            tid="tid",
            sid="sid",
            lastmjd="lastmjd",
            firstmjd="firstmjd",
            deltajd=1,
            corrected=True,
            stellar=True,
            sigmara=0.1,
            sigmadec=0.2,
            meanra=100.0,
            meandec=50.0,
            ndet="ndet",
        )
        self.assertIsInstance(o, models.Object)
        self.assertIsInstance(o, dict)
        self.assertEqual(o["loc"], {"type": "Point", "coordinates": [-80.0, 50]})
        self.assertEqual(
            o._meta.tablename,
            models.Object.__tablename__,
        )

    def test_object_fails_creation(self):
        with self.assertRaises(AttributeError) as e:
            models.Object()
        self.assertEqual(str(e.exception), "Object model needs _id attribute")

    def test_detection_creates(self):
        d = models.Detection(
            tid="tid",
            aid="aid",
            oid="oid",
            pid="pid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            rb="rb",
            mag="mag",
            e_mag="e_mag",
            mag_corr="mag_corr",
            e_mag_corr="e_mag_corr",
            e_mag_corr_ext="e_mag_corr_ext",
            e_ra="e_ra",
            e_dec="e_dec",
            isdiffpos="isdiffpos",
            magpsf_corr="magpsf_corr",
            sigmapsf_corr="sigmapsf_corr",
            sigmapsf_corr_ext="sigmapsf_corr_ext",
            corrected="corrected",
            dubious="dubious",
            parent_candid="parent_candid",
            has_stamp="has_stamp",
            rbversion="rbversion",
        )
        self.assertIsInstance(d, models.Detection)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["aid"], "aid")

    def test_detection_fails_creation(self):
        with self.assertRaisesRegex(
            AttributeError, "Detection model needs .+? attribute"
        ):
            models.Detection()

    def test_detection_with_extra_fields(self):
        o = models.Detection(
            tid="tid",
            aid="aid",
            oid="oid",
            pid="pid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            mag="mag",
            e_mag="e_mag",
            mag_corr="mag_corr",
            e_mag_corr="e_mag_corr",
            e_mag_corr_ext="e_mag_corr_ext",
            parent_candid="parent_candidate",
            dubious="dubious",
            e_ra="e_ra",
            e_dec="e_dec",
            isdiffpos="isdiffpos",
            corrected="corrected",
            has_stamp="has_stamp",
            extra="extra",
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
        o = models.Detection(
            tid="tid",
            aid="aid",
            oid="oid",
            pid="pid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            ra="ra",
            dec="dec",
            rb="rb",
            mag="mag",
            e_mag="e_mag",
            mag_corr="mag_corr",
            e_mag_corr="e_mag_corr",
            e_mag_corr_ext="e_mag_corr_ext",
            dubious="dubious",
            e_ra="e_ra",
            e_dec="e_dec",
            isdiffpos="isdiffpos",
            corrected="corrected",
            parent_candid="parent_candid",
            has_stamp="has_stamp",
            rbversion="rbversion",
            extra_fields={"extra": "extra"},
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})

    def test_forced_photometry_creates(self):
        fp = models.ForcedPhotometry(
            tid="tid",
            aid="aid",
            oid="oid",
            pid="pid",
            sid="sid",
            candid="candid",
            mjd="mjd",
            fid="fid",
            rb="rb",
            ra="ra",
            dec="dec",
            e_ra="e_ra",
            e_dec="e_dec",
            mag="mag",
            e_mag="e_mag",
            mag_corr="mag_corr",
            e_mag_corr="e_mag_corr",
            e_mag_corr_ext="e_mag_corr_ext",
            dubious="dubious",
            isdiffpos="isdiffpos",
            corrected="corrected",
            parent_candid="parent_candid",
            has_stamp="has_stamp",
            rbversion="rbversion",
            extra_fields={},
        )
        self.assertIsInstance(fp, models.ForcedPhotometry)
        self.assertIsInstance(fp, dict)
        self.assertEqual(fp["aid"], "aid")

    def test_non_detection_creates(self):
        o = models.NonDetection(
            candid="candid",
            aid="aid",
            oid="oid",
            sid="sid",
            tid="sid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
        )
        self.assertIsInstance(o, models.NonDetection)
        self.assertIsInstance(o, dict)
        self.assertEqual(o["aid"], "aid")

    def test_non_detection_fails_creation(self):
        with self.assertRaisesRegex(
            AttributeError, "NonDetection model needs .+ attribute"
        ):
            models.NonDetection()
        # self.assertEqual(str(e.exception), "NonDetection model needs aid attribute")

    def test_non_detection_with_extra_fields(self):
        o = models.NonDetection(
            candid="candid",
            aid="aid",
            oid="oid",
            sid="sid",
            tid="tid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
            extra="extra",
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
        o = models.NonDetection(
            candid="candid",
            aid="aid",
            sid="sid",
            oid="oid",
            tid="tid",
            mjd="mjd",
            diffmaglim="diffmaglim",
            fid="fid",
            extra_fields={"extra": "extra"},
        )
        self.assertEqual(o["extra_fields"], {"extra": "extra"})
