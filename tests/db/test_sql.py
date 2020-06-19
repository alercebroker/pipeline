from test_core import *
from db_plugins.db import SQLDatabase
from db_plugins.db.sql.models import *
from db_plugins.db.sql import Pagination
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest
import json
import time
import datetime


class DatabaseConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config = {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"}
        session_options = {
            "autocommit": False,
            "autoflush": True,
        }
        self.db = SQLDatabase()
        self.db.connect(config=config, session_options=session_options)

    @classmethod
    def tearDownClass(self):
        self.db.drop_db()
        self.db.session.close()

    def setUp(self):
        self.db.create_db()
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
        self.db.session.add(astro_object)
        class_object = Class(name="Super Nova", acronym="SN")
        self.db.session.add(class_object)
        classifier = Classifier(name="test")
        self.db.session.add(classifier)
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="Super Nova"
        )
        self.db.session.add(classification)
        self.db.session.commit()

    def tearDown(self):
        self.db.session.close()
        self.db.drop_db()

    def test_get_or_create(self):
        instance, created = self.db.session.query().get_or_create(
            AstroObject, {"oid": "ZTF1"}
        )
        self.assertIsInstance(instance, AstroObject)
        self.assertFalse(created)

    def test_check_exists(self):
        self.assertTrue(
            self.db.session.query().check_exists(AstroObject, {"oid": "ZTF1"})
        )

    def test_update(self):
        instance, _ = self.db.session.query().get_or_create(
            AstroObject, {"oid": "ZTF1"}
        )
        updated = self.db.session.query().update(instance, {"oid": "ZTF2"})
        self.assertEqual(updated.oid, "ZTF2")
        instance, created = self.db.session.query().get_or_create(
            AstroObject, {"oid": "ZTF1"}
        )
        self.assertTrue(created)

    def test_bulk_insert(self):
        astro_objects = [{"oid": "ZTF2"}, {"oid": "ZTF3"}]
        self.db.session.query().bulk_insert(astro_objects, AstroObject)
        objects = self.db.session.query(AstroObject).all()
        self.assertEqual(len(objects), 3)

    def test_query_all(self):
        results = self.db.session.query().query([AstroObject])
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_query_filter(self):
        # filters = [{'field': "oid", 'op': '==', 'value': "ZTF1"}]
        results = self.db.session.query().query(
            [AstroObject], None, None, None, None, None, AstroObject.oid == "ZTF1"
        )
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].oid, "ZTF1")

    def test_join(self):
        results = self.db.session.query().query(
            [Classification, AstroObject, Class],
            None,
            None,
            None,
            Classification.astro_object == AstroObject.oid,
            Classification.class_name == Class.name,
        )
        self.assertEqual(len(results["results"]), 1)
        self.assertEqual(results["results"][0].AstroObject.oid, "ZTF1")
        self.assertEqual(results["results"][0].Classification.class_name, "Super Nova")
        self.assertEqual(results["results"][0].Class.name, "Super Nova")

    def test_query_pagination(self):
        for i in range(19):
            self.db.session.query().get_or_create(
                AstroObject, {"oid": "ZTF" + str(i + 2)}
            )
        self.db.session.commit()
        results = self.db.session.query().query([AstroObject], 0, 10)
        self.assertEqual(len(results["results"]), 10)
        self.assertEqual(results["total"], 20)

    def test_order_desc(self):
        for i in range(19):
            self.db.session.query().get_or_create(
                AstroObject, {"oid": "ZTF" + str(i + 2), "nobs": i}
            )
        self.db.session.commit()
        results = self.db.session.query().query(
            [AstroObject], None, None, None, AstroObject.nobs, "DESC"
        )
        for i in range(10):
            self.assertGreater(
                results["results"][i].nobs, results["results"][19 - i].nobs
            )

    def test_order_asc(self):
        for i in range(19):
            self.db.session.query().get_or_create(
                AstroObject, {"oid": "ZTF" + str(i + 2), "nobs": 100 - i}
            )
        self.db.session.commit()
        results = self.db.session.query().query(
            [AstroObject], None, None, None, AstroObject.nobs, "ASC"
        )
        for i in range(10):
            self.assertLess(results["results"][i].nobs, results["results"][19 - i].nobs)

    def test_paginate(self):
        pagination = self.db.session.query(AstroObject).paginate()
        self.assertIsInstance(pagination, Pagination)
        self.assertEqual(pagination.total, 1)
        self.assertEqual(pagination.items[0].oid, "ZTF1")


class ScopedDatabaseConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config = {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"}
        session_options = {
            "autocommit": False,
            "autoflush": False,
        }
        self.db = SQLDatabase()
        self.db.connect(config=config, session_options=session_options, use_scoped=True)

    @classmethod
    def tearDownClass(self):
        self.db.drop_db()
        self.db.session.remove()

    def setUp(self):
        self.db.create_db()
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
        self.db.session.add(astro_object)
        self.db.session.commit()

    def tearDown(self):
        self.db.drop_db()
        self.db.session.remove()

    def test_query_property(self):
        self.assertEqual(len(AstroObject.query.all()), 1)
        self.assertEqual(AstroObject.query.first().oid, "ZTF1")

    def test_method_access_from_session(self):
        instance, created = self.db.session.query().get_or_create(
            model=AstroObject, filter_by={"oid": "ZTF1"}
        )
        self.assertIsInstance(instance, AstroObject)
        self.assertFalse(created)

    def test_method_access_from_query_property(self):
        instance, created = AstroObject.query.get_or_create(filter_by={"oid": "ZTF1"})
        self.assertIsInstance(instance, AstroObject)
        self.assertFalse(created)


class ClassTest(unittest.TestCase, GenericClassTest):
    pass


class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
    pass


class ClassifierTest(GenericClassifierTest, unittest.TestCase):
    pass


class XMatchTest(GenericXMatchTest, unittest.TestCase):
    pass


class MagnitudeStatisticsTest(GenericMagnitudeStatisticsTest, unittest.TestCase):
    pass


class ClassificationTest(GenericClassificationTest, unittest.TestCase):
    pass


class AstroObjectTest(GenericAstroObjectTest, unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config = {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"}
        session_options = {
            "autocommit": False,
            "autoflush": False,
        }
        self.db = SQLDatabase()
        self.db.connect(config=config, session_options=session_options)

    @classmethod
    def tearDownClass(self):
        self.db.drop_db()
        self.db.session.close()

    def setUp(self):
        self.db.create_db()
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
        self.model.xmatches.append(Xmatch(catalog_id="C1", catalog_oid="O1"))
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
        self.model.non_detections.append(
            NonDetection(mjd=1, fid=1, diffmaglim=1, datetime=datetime.datetime.now())
        )
        self.db.session.add(self.model)
        self.db.session.commit()

    def tearDown(self):
        self.db.session.close()
        self.db.drop_db()


class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
    pass


class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
    pass


class DetectionTest(GenericDetectionTest, unittest.TestCase):
    pass
