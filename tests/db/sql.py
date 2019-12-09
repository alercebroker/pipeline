from .core import *
from apf.db.sql import Base, models, Session
from apf.db.sql.models import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest
import json


class ClassTest(GenericClassTest, unittest.TestCase):
    model = Class(name="Super Nova", acronym="SN")


class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
    model = Taxonomy(name="Test")


class ClassifierTest(GenericClassifierTest, unittest.TestCase):
    pass


class MagRefTest(GenericMagRefTest, unittest.TestCase):
    pass


class XMatchTest(GenericXMatchTest, unittest.TestCase):
    pass


class MagnitudeStatisticsTest(GenericMagnitudeStatisticsTest, unittest.TestCase):
    pass


class ClassificationTest(GenericClassificationTest, unittest.TestCase):
    pass


class AstroObjectTest(GenericAstroObjectTest, unittest.TestCase):

    def setUp(self):
        engine = create_engine('sqlite:///:memory:')
        Session = sessionmaker(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        class_ = Class(name="Super Nova", acronym="SN")
        taxonomy = Taxonomy(name="Test")
        class_.taxonomies.append(taxonomy)
        classifier = Classifier(name="C1")
        taxonomy.classifiers.append(classifier)
        self.model = AstroObject(oid="ZTF1", nobs=1, lastmjd=1.0, meanra=1.0,
                                 meandec=1.0, sigmara=1.0, sigmadec=1.0, deltajd=1.0, firstmjd=1.0)
        self.model.magref = MagRef(
            fid=1, rcid=1, field=1, magref=1.0, sigmagref=1.0, corrected=1.0)
        self.model.xmatches.append(
            Xmatch(catalog_id="C1", catalog_object_id="O1"))
        self.model.magnitude_statistics = MagnitudeStatistics(
            fid=1, magnitude_type="psf", mean=1.0, median=1.0, max_mag=1.0, min_mag=1.0, sigma=1.0, last=1.0, first=1.0)
        self.model.classifications.append(Classification(
            class_name="Super Nova", probability=1.0, classifier_name="C1"))
        self.model.features.append(
            Features(data=json.loads('{"test": "test"}')))
        self.model.detections.append(Detection(candid="t", mjd=1, fid=1, ra=1, dec=1, rb=1,
                                               magap=1, magpsf=1, sigmapsf=1, sigmagap=1, alert=json.loads('{"test":"test"}')))
        self.model.non_detections.append(NonDetection(mjd=1, fid=1, diffmaglim=1))



class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
    pass


class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
    pass


class DetectionTest(GenericDetectionTest, unittest.TestCase):
    pass
