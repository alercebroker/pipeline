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


class ObjectSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Object


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


class DataqualitySchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Dataquality

class Gaia_ztfSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Gaia_ztf


class Ss_ztfSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Ss_ztf


class Ps1_ztfSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Ps1_ztf


class ReferenceSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Reference


class PipelineSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Pipeline


class StepSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Step
