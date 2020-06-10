from .core import *
from db_plugins.db.sql import DatabaseConnection
from db_plugins.db.sql.models import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest
import json
import time


db = DatabaseConnection()
db.init_app("sqlite:///:memory:", Base)
db.create_db()


class DatabaseConnectionTest(unittest.TestCase):
    def setUp(self):
        self.connection = db.engine.connect()
        self.trans = self.connection.begin()
        db.create_session(bind=self.connection)
        astro_object = AstroObject(
            oid="ZTF1",
            nobs=1,
            lastmjd=1.0,
            meanra=1.0,
            meandec=1.0,
            sigmadec=1.0,
            deltajd=1.0,
            firstmjd=1.0,
        )
        db.session.add(astro_object)
        class_object = Class(name="Super Nova", acronym="SN")
        db.session.add(class_object)
        classifier = Classifier(name="test")
        db.session.add(classifier)
        classification = Classification(
            astro_object="ZTF1",
            classifier_name="test",
            class_name="Super Nova")
        db.session.add(classification)
        db.session.commit()

    def tearDown(self):
        db.session.close()
        self.trans.rollback()
        self.connection.close()

    def test_get_or_create(self):
        instance, created = db.get_or_create(AstroObject, {"oid": "ZTF1"})
        self.assertIsInstance(instance, AstroObject)
        self.assertFalse(created)

    def test_check_exists(self):
        self.assertTrue(db.check_exists(AstroObject, {"oid": "ZTF1"}))

    def test_update(self):
        instance, _ = db.get_or_create(AstroObject, {"oid": "ZTF1"})
        updated = db.update(instance, {"oid": "ZTF2"})
        self.assertEqual(updated.oid, "ZTF2")
        instance, created = db.get_or_create(AstroObject, {"oid": "ZTF1"})
        self.assertTrue(created)

    def test_create_session(self):
        session = db.create_session()
        from sqlalchemy.orm.session import Session as S

        self.assertIsInstance(session, S)

    def test_create_scoped_session(self):
        session = db.create_scoped_session()
        from sqlalchemy.orm import scoped_session as S

        self.assertIsInstance(session, S)

    def test_bulk_insert(self):
        astro_objects = [{"oid": "ZTF2"}, {"oid": "ZTF3"}]
        db.bulk_insert(astro_objects, AstroObject)
        objects = db.session.query(AstroObject).all()
        self.assertEqual(len(objects), 3)

    def test_query_all(self):
        results = db.query([AstroObject])
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_query_filter(self):
        #filters = [{'field': "oid", 'op': '==', 'value': "ZTF1"}]
        results = db.query([AstroObject], None,
                        None, None, None, None, AstroObject.oid == "ZTF1")
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_join(self):
        results = db.query([Classification, AstroObject, Class], None, None, None,
                                 Classification.astro_object == AstroObject.oid,
                                 Classification.class_name == Class.name)
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].AstroObject.oid, "ZTF1")
        self.assertEqual(results["results"][0].Classification.class_name, "Super Nova")
        self.assertEqual(results["results"][0].Class.name, "Super Nova")

    def test_query_pagination(self):
        for i in range(19):
            db.get_or_create(AstroObject, {"oid": "ZTF" + str(i + 2)})
        db.session.commit()
        results = db.query([AstroObject], 0, 10)
        self.assertEqual(len(results["results"]), 10)
        self.assertEqual(results["total"], 20)

    def test_order_desc(self):
        for i in range(19):
            db.get_or_create(AstroObject, {"oid": "ZTF" + str(i + 2), "nobs": i})
        db.session.commit()
        results = db.query([AstroObject], None, None, None, AstroObject.nobs, "DESC")
        for i in range(10):
            self.assertGreater(results["results"][i].nobs, results["results"][19-i].nobs)

    def test_order_asc(self):
        for i in range(19):
            db.get_or_create(AstroObject, {"oid": "ZTF" + str(i + 2), "nobs": 100-i})
        db.session.commit()
        results = db.query([AstroObject], None, None, None, AstroObject.nobs, "ASC")
        for i in range(10):
            self.assertLess(results["results"][i].nobs, results["results"][19-i].nobs)


class ClassTest(unittest.TestCase, GenericClassTest):
    def setUp(self):
        self.connection = db.engine.connect()
        self.trans = self.connection.begin()
        self.session = db.create_session(bind=self.connection)
        self.model = Class(name="Super Nova", acronym="SN")
        astro_object = AstroObject(
            oid="ZTF1",
            nobs=1,
            lastmjd=1.0,
            meanra=1.0,
            meandec=1.0,
            sigmadec=1.0,
            deltajd=1.0,
            firstmjd=1.0,
        )
        classifier = Classifier(name="test")
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="SN"
        )
        self.model.classifications.append(classification)
        db.session.add(astro_object)
        db.session.commit()

    def tearDown(self):
        db.session.close()
        self.trans.rollback()
        self.connection.close()


class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
    def setUp(self):
        self.connection = db.engine.connect()
        self.trans = self.connection.begin()
        self.session = db.create_session(bind=self.connection)
        self.model = Taxonomy(name="test")
        class_ = Class(name="SN")
        classifier = Classifier(name="asdasd")
        self.model.classifiers.append(classifier)
        self.model.classes.append(class_)
        db.session.commit()

    def tearDown(self):
        db.session.close()
        self.trans.rollback()
        self.connection.close()


class ClassifierTest(GenericClassifierTest, unittest.TestCase):
    def setUp(self):
        self.connection = db.engine.connect()
        self.trans = self.connection.begin()
        self.session = db.create_session(bind=self.connection)
        self.model = Classifier(name="Late Classifier")
        astro_object = AstroObject(
            oid="ZTF1",
            nobs=1,
            lastmjd=1.0,
            meanra=1.0,
            meandec=1.0,
            sigmadec=1.0,
            deltajd=1.0,
            firstmjd=1.0,
        )
        classifier = Classifier(name="test")
        class_ = Class(name="SN")
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="SN"
        )
        self.model.classifications.append(classification)

    def tearDown(self):
        db.session.close()
        self.trans.rollback()
        self.connection.close()


class XMatchTest(GenericXMatchTest, unittest.TestCase):
    pass


class MagnitudeStatisticsTest(GenericMagnitudeStatisticsTest, unittest.TestCase):
    pass


class ClassificationTest(GenericClassificationTest, unittest.TestCase):
    pass


class AstroObjectTest(GenericAstroObjectTest, unittest.TestCase):
    def setUp(self):
        self.connection = db.engine.connect()
        self.trans = self.connection.begin()
        self.session = db.create_session(bind=self.connection)

        class_ = Class(name="Super Nova", acronym="SN")
        taxonomy = Taxonomy(name="Test")
        class_.taxonomies.append(taxonomy)
        classifier = Classifier(name="C1")
        taxonomy.classifiers.append(classifier)
        self.model = AstroObject(
            oid="ZTF1",
            nobs=1,
            lastmjd=1.0,
            meanra=1.0,
            meandec=1.0,
            sigmara=1.0,
            sigmadec=1.0,
            deltajd=1.0,
            firstmjd=1.0,
        )
        self.model.xmatches.append(Xmatch(catalog_id="C1", catalog_object_id="O1"))
        self.model.magnitude_statistics = MagnitudeStatistics(
            fid=1,
            magnitude_type="psf",
            mean=1.0,
            median=1.0,
            max_mag=1.0,
            min_mag=1.0,
            sigma=1.0,
            last=1.0,
            first=1.0,
        )
        self.model.classifications.append(
            Classification(
                class_name="Super Nova", probability=1.0, classifier_name="C1"
            )
        )

        features_object = FeaturesObject(data=json.loads('{"test":"test"}'))
        features_object.features = Features(version="V1")
        self.model.features.append(features_object)
        self.model.detections.append(
            Detection(
                candid="t",
                mjd=1,
                fid=1,
                ra=1,
                dec=1,
                rb=1,
                magap=1,
                magpsf=1,
                sigmapsf=1,
                sigmagap=1,
                alert=json.loads('{"test":"test"}'),
            )
        )
        self.model.non_detections.append(NonDetection(mjd=1, fid=1, diffmaglim=1))
        # self.session.add(self.model)

    def tearDown(self):
        db.session.close()
        self.trans.rollback()
        self.connection.close()


class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
    pass


class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
    pass


class DetectionTest(GenericDetectionTest, unittest.TestCase):
    pass
