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


def create_magstats(**kwargs):
    return kwargs.get("magstats", [])


def create_features(**kwargs):
    return kwargs.get("features", [])


def create_probabilities(**kwargs):
    return kwargs.get("probabilities", [])


def create_xmatch(**kwargs):
    return kwargs.get("xmatch", [])


class Object(generic_models.Object, Base):
    """Mongo implementation of the Object class.

    Contains definitions of indexes and custom attributes like loc.
    """

    def loc_definition(**kwargs):
        return {
            "type": "Point",
            "coordinates": [kwargs["meanra"] - 180.0, kwargs["meandec"]],
        }

    aid = Field()  # ALeRCE candidate id (unique id of object in the ALeRCE database)
    oid = (
        Field()
    )  # Object id should include objects id of all surveys (same survey can provide different object ids)
    lastmjd = Field()
    firstmjd = Field()
    ndet = Field()
    loc = SpecialField(loc_definition)
    meanra = Field()
    meandec = Field()
    magstats = SpecialField(create_magstats)
    features = SpecialField(create_features)
    probabilities = SpecialField(create_probabilities)
    xmatch = SpecialField(create_xmatch)
    extra_fields = SpecialField(create_extra_fields)

    __table_args__ = [
        IndexModel([("aid", ASCENDING), ("oid", ASCENDING)]),
        IndexModel([("lastmjd", DESCENDING)]),
        IndexModel([("firstmjd", DESCENDING)]),
        IndexModel([("loc", GEOSPHERE)]),
        IndexModel([("meanra", ASCENDING)]),
        IndexModel([("meandec", ASCENDING)]),
    ]
    __tablename__ = "object"


class Detection(Base, generic_models.Detection):

    tid = (
        Field()
    )  # Telescope id (this gives the spatial coordinates of the observatory, e.g. ZTF, ATLAS-HKO, ATLAS-MLO)
    aid = Field()
    oid = Field()
    candid = Field()
    mjd = Field()
    fid = Field()
    ra = Field()
    dec = Field()
    rb = Field()
    mag = Field()
    e_mag = Field()
    rfid = Field()
    e_ra = Field()
    e_dec = Field()
    isdiffpos = Field()
    corrected = Field()
    parent_candid = Field()
    has_stamp = Field()
    step_id_corr = Field()
    rbversion = Field()
    extra_fields = SpecialField(create_extra_fields)
    __table_args__ = [IndexModel([("aid", ASCENDING), ("tid", ASCENDING)])]
    __tablename__ = "detection"


class NonDetection(Base, generic_models.NonDetection):

    aid = Field()
    tid = Field()
    oid = Field()
    mjd = Field()
    diffmaglim = Field()
    fid = Field()
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
