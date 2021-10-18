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


def create_extra_fields(Model, **kwargs):
    if "extra_fields" in kwargs:
        return kwargs["extra_fields"]
    else:
        for field in Model._meta.fields:
            try:
                kwargs.pop(field)
            except (KeyError):
                pass
        return kwargs


class Object(generic_models.Object, Base):
    """Mongo implementation of the Object class.

    Contains definitions of indexes and custom attributes like loc.
    """

    def loc_definition(**kwargs):
        return {
            "type": "Point",
            "coordinates": [kwargs["meanra"], kwargs["meandec"]],
        }

    aid = Field()
    sid = Field()
    lastmjd = Field()
    firstmjd = Field()
    loc = SpecialField(loc_definition)
    meanra = Field()
    meandec = Field()
    extra_fields = SpecialField(create_extra_fields)

    __table_args__ = [
        IndexModel([("aid", TEXT)]),
        IndexModel([("sid", TEXT)]),
        IndexModel([("lastmjd", DESCENDING)]),
        IndexModel([("firstmjd", DESCENDING)]),
        IndexModel([("loc", GEOSPHERE)]),
        IndexModel([("meanra", ASCENDING)]),
        IndexModel([("meandec", ASCENDING)]),
    ]
    __tablename__ = "object"


class Detection(Base, generic_models.Detection):

    aid = Field()
    sid = Field()
    candid = Field()
    mjd = Field()
    fid = Field()
    ra = Field()
    dec = Field()
    rb = Field()
    mag = Field()
    sigmag = Field()
    extra_fields = SpecialField(create_extra_fields)
    __table_args__ = [IndexModel([("aid", TEXT)])]
    __tablename__ = "detection"


class NonDetection(Base, generic_models.NonDetection):

    aid = Field()
    sid = Field()
    mjd = Field()
    diffmaglim = Field()
    fid = Field()
    extra_fields = SpecialField(create_extra_fields)

    __table_args__ = [
        IndexModel([("aid", TEXT)]),
        IndexModel([("sid", TEXT)]),
        IndexModel([("mjd", DESCENDING)]),
        IndexModel([("fid", ASCENDING)]),
    ]
    __tablename__ = "non_detection"
