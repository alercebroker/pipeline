from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from db_plugins.db.sql.models import *


class ClassSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Class


class TaxonomySchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Taxonomy


class ClassifierSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Classifier


class AstroObjectSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = AstroObject


class ClassificationSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Classification


class MagnitudeStatisticsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = MagnitudeStatistics


class FeaturesSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = FeaturesObject


class DetectionSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Detection


class NonDetectionSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = NonDetection
