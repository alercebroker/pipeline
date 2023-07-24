from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from .models import *


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
