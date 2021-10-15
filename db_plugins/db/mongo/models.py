from db_plugins.db import models as generic_models
from pymongo import (
    TEXT,
    GEOSPHERE,
    IndexModel,
    DESCENDING,
    ASCENDING,
)
from db_plugins.db.mongo.orm import base_creator, Field, SpecialField

Base = base_creator()


class Object(Base, generic_models.Object):
    """Mongo implementation of the Object class.

    Contains definitions of indexes and custom attributes like loc.
    """

    def loc_definition(**kwargs):
        return {
            "type": "Point",
            "coordinates": [kwargs["meanra"], kwargs["meandec"]],
        }

    alerce_id = Field("alerce_id")
    survey_id = Field("survey_id")
    lastmjd = Field("lastmjd")
    firstmjd = Field("firstmjd")
    loc = SpecialField("loc", loc_definition)
    meanra = Field("meanra")
    meandec = Field("meandec")

    __table_args__ = [
        IndexModel([("alerce_id", TEXT)]),
        IndexModel([("survey_id", TEXT)]),
        IndexModel([("lastmjd", DESCENDING)]),
        IndexModel([("firstmjd", DESCENDING)]),
        IndexModel([("loc", GEOSPHERE)]),
        IndexModel([("meanra", ASCENDING)]),
        IndexModel([("meandec", ASCENDING)]),
    ]


class Detection(Base):
    some_attr = None
    __table_args__ = [IndexModel([("some_attr", TEXT)])]
