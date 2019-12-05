from .core import *
from apf.db.sql import Base, models, Session
from apf.db.sql.models import *
from sqlalchemy import create_engine
import unittest


class ClassTest(GenericClassTest, unittest.TestCase):
    model = Class
    params = {"name": "Super Nova", "acronym": "SN"}


class TaxonomyTest(GenericTaxonomyTest, unittest.TestCase):
    model = Taxonomy
    params = {"name": "Taxonomy1"}


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
    model = AstroObject
    params = {}

    def setUp(self):
        engine = create_engine('sqlite:///:memory:')
        Session = sessionmaker(bind=engine)
        session = Session()
        Base.metadata.create_all(engine)
        self.session.add(MagRef())


class FeaturesTest(GenericFeaturesTest, unittest.TestCase):
    pass


class NonDetectionTest(GenericNonDetectionTest, unittest.TestCase):
    pass


class DetectionTest(GenericDetectionTest, unittest.TestCase):
    pass
