from .core import *
from apf.db.sql import Base, models, Session
from apf.db.sql.models import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest
import json


engine = create_engine('sqlite:///:memory:')
Session = sessionmaker()
Base.metadata.create_all(engine)

class ClassTest(GenericClassTest, unittest.TestCase):
    model = Class(name="Super Nova", acronym="SN")
    def setUp(self):
        self.connection = engine.connect()
        self.trans = self.connection.begin()
        self.session = Session(bind=self.connection)
        astro_object = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0,
                                   meanra=1.0, meandec=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        classifier = Classifier(name="test")
        classification = Classification(
            astro_object="ZTF1", classifier_name="test", class_name="SN")
        print("before",self.model.classifications)
        self.model.classifications.append(classification)
        print("after",self.model.classifications)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.trans.rollback()
        self.connection.close()
        


# class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
#     model = Taxonomy(name="Test")
#     def setUp(self):
#         self.connection = engine.connect()
#         self.trans = self.connection.begin()
#         self.session = Session(bind=self.connection)

#         class_ = Class(name="SN")
#         classifier = Classifier(name="asdasd")
#         print("before",self.model.classifiers)
#         self.model.classifiers.append(classifier)
#         print("after",self.model.classifiers)
#         self.model.classes.append(class_)
#         self.session.commit()

#     def tearDown(self):
#         self.session.close()
#         self.trans.rollback()
#         self.connection.close()
        


# class ClassifierTest(GenericClassifierTest, unittest.TestCase):
#     def setUp(self):
#         self.connection = engine.connect()
#         self.trans = self.connection.begin()
#         self.session = Session(bind=self.connection)
#         self.model = Classifier(name="Late Classifier")
#         astro_object = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0,
#                                    meanra=1.0, meandec=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
#         classifier = Classifier(name="test")
#         class_ = Class(name="SN")
#         classification = Classification(
#             astro_object="ZTF1", classifier_name="test", class_name="SN")
#         self.model.classifications.append(classification)
#         # self.session.add_all([self.model, astro_object, classifier, class_, classification])
#     def tearDown(self):
#         self.session.close()
#         self.trans.rollback()
#         self.connection.close()

# class XMatchTest(GenericXMatchTest, unittest.TestCase):
#     pass


# class MagnitudeStatisticsTest(GenericMagnitudeStatisticsTest, unittest.TestCase):
#     pass


# class ClassificationTest(GenericClassificationTest, unittest.TestCase):
#     pass


# class AstroObjectTest(GenericAstroObjectTest, unittest.TestCase):

#     def setUp(self):
#         self.connection = engine.connect()
#         self.trans = self.connection.begin()
#         self.session = Session(bind=self.connection)
        
#         class_ = Class(name="Super Nova", acronym="SN")
#         taxonomy = Taxonomy(name="Test")
#         class_.taxonomies.append(taxonomy)
#         classifier = Classifier(name="C1")
#         taxonomy.classifiers.append(classifier)
#         self.model = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0, meanra=1.0,
#                                  meandec=1.0, sigmara=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
#         self.model.xmatches.append(
#             Xmatch(catalog_id="C1", catalog_object_id="O1"))
#         self.model.magnitude_statistics = MagnitudeStatistics(
#             fid=1, magnitude_type="psf", mean=1.0, median=1.0, max_mag=1.0, min_mag=1.0, sigma=1.0, last=1.0, first=1.0)
#         self.model.classifications.append(Classification(
#             class_name="Super Nova", probability=1.0, classifier_name="C1"))

#         features_object = FeaturesObject(data=json.loads('{"test":"test"}'))
#         features_object.features = Features(version="V1")
#         self.model.features.append(features_object)
#         self.model.detections.append(Detection(candid="t", mjd=1, fid=1, ra=1, dec=1, rb=1,
#                                                magap=1, magpsf=1, sigmapsf=1, sigmagap=1, alert=json.loads('{"test":"test"}')))
#         self.model.non_detections.append(
#             NonDetection(mjd=1, fid=1, diffmaglim=1))
#         # self.session.add(self.model)

#     def tearDown(self):
#         self.session.close()
#         self.trans.rollback()
#         self.connection.close()


# class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
#     pass


# class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
#     pass


# class DetectionTest(GenericDetectionTest, unittest.TestCase):
#     pass
