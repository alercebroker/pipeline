from .core import *
from db_plugins.db.sql import *
from db_plugins.db.sql.models import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest
import json


engine = create_engine('sqlite:///:memory:')
Session = sessionmaker()
Base.metadata.create_all(engine)


class SQLMethodsTest(unittest.TestCase):
    def setUp(self):
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)
        astro_object = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0,
                                   meanra=1.0, meandec=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        self.session.add(astro_object)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.trans.rollback()
        self.connection.close()

    def test_get_or_create(self):
        instance, created = get_or_create(
            self.session, AstroObject, {"oid": "ZTF1"})
        self.assertIsInstance(instance, AstroObject)
        self.assertFalse(created)

    def test_check_exists(self):
        self.assertTrue(check_exists(
            self.session, AstroObject, {"oid": "ZTF1"}))

    def test_update(self):
        instance, _ = get_or_create(self.session, AstroObject, {"oid": "ZTF1"})
        updated = update(instance, {"oid": "ZTF2"})
        self.assertEqual(updated.oid, "ZTF2")
        instance, created = get_or_create(
            self.session, AstroObject, {"oid": "ZTF1"})
        self.assertTrue(created)

    def test_get_session(self):
        fake_credentials = {
            "PSQL": {
                "USER": "fake",
                "HOST": "fake",
                "PASSWORD": "fake",
                "PORT": "1",
                "DB_NAME": "fake"
            }
        }
        session = get_session(fake_credentials)
        from sqlalchemy.orm.session import Session as S
        self.assertIsInstance(session, S)

    def test_add_to_database_instance(self):
        astro_object = AstroObject(oid="ZTF2")
        add_to_database(self.session, astro_object)
        objects = self.session.query(AstroObject).all()
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[1], astro_object)

    def test_add_to_database_list(self):
        astro_objects = [AstroObject(oid="ZTF2"), AstroObject(oid="ZTF3")]
        add_to_database(self.session, astro_objects)
        objects = self.session.query(AstroObject).all()
        self.assertEqual(len(objects), 3)

    def test_bulk_insert(self):
        astro_objects = [{"oid": "ZTF2"}, {"oid": "ZTF3"}]
        bulk_insert(astro_objects, AstroObject, self.session)
        objects = self.session.query(AstroObject).all()
        self.assertEqual(len(objects), 3)

    def test_query_all(self):
        results = query(self.session, AstroObject)
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_query_filter(self):
        results = query(self.session, AstroObject, None,
                        None, None, AstroObject.oid == "ZTF1")
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_query_pagination(self):
        for i in range(19):
            get_or_create(self.session, AstroObject, {"oid": "ZTF" + str(i+2)})
        self.session.commit()
        results = query(self.session,AstroObject, 1, 10)
        self.assertEqual(len(results["results"]), 10)
        self.assertEqual(results["total"], 20)
        


class ClassTest(unittest.TestCase, GenericClassTest):

    def setUp(self):
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)
        self.model = Class(name="Super Nova", acronym="SN")
        astro_object = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0,
                                   meanra=1.0, meandec=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        classifier = Classifier(name="test")
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="SN")
        self.model.classifications.append(classification)
        self.session.add(astro_object)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.trans.rollback()
        self.connection.close()


class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
    def setUp(self):
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)
        self.model = Taxonomy(name="test")
        class_ = Class(name="SN")
        classifier = Classifier(name="asdasd")
        self.model.classifiers.append(classifier)
        self.model.classes.append(class_)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.trans.rollback()
        self.connection.close()


class ClassifierTest(GenericClassifierTest, unittest.TestCase):
    def setUp(self):
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)
        self.model = Classifier(name="Late Classifier")
        astro_object = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0,
                                   meanra=1.0, meandec=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        classifier = Classifier(name="test")
        class_ = Class(name="SN")
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="SN")
        self.model.classifications.append(classification)

    def tearDown(self):
        self.session.close()
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
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)

        class_ = Class(name="Super Nova", acronym="SN")
        taxonomy = Taxonomy(name="Test")
        class_.taxonomies.append(taxonomy)
        classifier = Classifier(name="C1")
        taxonomy.classifiers.append(classifier)
        self.model = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0, meanra=1.0,
                                 meandec=1.0, sigmara=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        self.model.xmatches.append(
            Xmatch(catalog_id="C1", catalog_object_id="O1"))
        self.model.magnitude_statistics = MagnitudeStatistics(
            fid=1, magnitude_type="psf", mean=1.0, median=1.0, max_mag=1.0, min_mag=1.0, sigma=1.0, last=1.0, first=1.0)
        self.model.classifications.append(Classification(
            class_name="Super Nova", probability=1.0, classifier_name="C1"))

        features_object = FeaturesObject(data=json.loads('{"test":"test"}'))
        features_object.features = Features(version="V1")
        self.model.features.append(features_object)
        self.model.detections.append(Detection(candid="t", mjd=1, fid=1, ra=1, dec=1, rb=1,
                                               magap=1, magpsf=1, sigmapsf=1, sigmagap=1, alert=json.loads('{"test":"test"}')))
        self.model.non_detections.append(
            NonDetection(mjd=1, fid=1, diffmaglim=1))
        # self.session.add(self.model)

    def tearDown(self):
        self.session.close()
        self.trans.rollback()
        self.connection.close()


class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
    pass


class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
    pass


class DetectionTest(GenericDetectionTest, unittest.TestCase):
    pass
