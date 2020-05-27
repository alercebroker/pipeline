from flask import Flask
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field
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
