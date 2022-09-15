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
            except KeyError:
                pass
        return kwargs


class Object(generic_models.Object, Base):
    """Mongo implementation of the Object class.

    Contains definitions of indexes and custom attributes like loc.
    """

    _id = SpecialField(lambda **kwargs: kwargs["aid"] or kwargs["_id"])  # ALeRCE object ID (unique ID in database)
    oid = Field()  # Should include OID of all surveys (same survey can have many OIDs)
    tid = Field()  # Should include all telescopes contributing with detections
    lastmjd = Field()
    firstmjd = Field()
    ndet = Field()
    loc = SpecialField(lambda **kwargs: {"type": "Point", "coordinates": [kwargs["meanra"] - 180, kwargs["meandec"]]})
    meanra = Field()
    meandec = Field()
    magstats = SpecialField(lambda **kwargs: kwargs.get("magstats", []))
    features = SpecialField(lambda **kwargs: kwargs.get("features", []))
    probabilities = SpecialField(lambda **kwargs: kwargs.get("probabilities", []))
    xmatch = SpecialField(lambda **kwargs: kwargs.get("xmatch", []))

    __table_args__ = [
        IndexModel([("oid", ASCENDING)]),
        IndexModel([("lastmjd", DESCENDING)]),
        IndexModel([("firstmjd", DESCENDING)]),
        IndexModel([("loc", GEOSPHERE)]),
    ]
    __tablename__ = "object"


class Detection(Base, generic_models.Detection):

    _id = SpecialField(lambda **kwargs: kwargs["candid"] or kwargs["_id"])
    tid = Field()  # Telescope ID
    aid = Field()
    oid = Field()
    mjd = Field()
    fid = Field()
    ra = Field()
    e_ra = Field()
    dec = Field()
    e_dec = Field()
    mag = Field()
    e_mag = Field()
    rfid = Field()
    isdiffpos = Field()
    corrected = Field()
    has_stamp = Field()
    step_id_corr = Field()
    extra_fields = SpecialField(create_extra_fields)
    __table_args__ = [IndexModel([("aid", ASCENDING), ("tid", ASCENDING)])]
    __tablename__ = "detection"


class NonDetection(Base, generic_models.NonDetection):

    aid = Field()
    tid = Field()
    oid = Field()
    mjd = Field()
    fid = Field()
    diffmaglim = Field()
    extra_fields = SpecialField(create_extra_fields)

    __table_args__ = [
        IndexModel([("aid", ASCENDING), ("tid", ASCENDING)]),
    ]
    __tablename__ = "non_detection"


class Taxonomy(Base):
    classifier_name = Field()
    classifier_version = Field()
    classes = Field()

    __table_args__ = [
        IndexModel([("classifier_name", ASCENDING), ("classifier_version", ASCENDING)]),
    ]
    __tablename__ = "taxonomy"
