from pymongo import ASCENDING, DESCENDING, GEOSPHERE, IndexModel
from .orm import Field, SpecialField, ModelMetaClass


class BaseModel(dict, metaclass=ModelMetaClass):
    def __init__(self, **kwargs):
        model = {}
        if "_id" in kwargs and "_id" not in self._meta.fields:
            model["_id"] = kwargs["_id"]
        for field, fclass in self._meta.fields.items():
            try:
                try:
                    model[field] = fclass.callback(**kwargs)
                except AttributeError:
                    model[field] = kwargs[field]
            except KeyError:
                raise AttributeError(
                    f"{self.__class__.__name__} model needs {field} attribute"
                )
        super().__init__(**model)


class BaseModelWithExtraFields(BaseModel):
    @classmethod
    def create_extra_fields(cls, **kwargs):
        if "extra_fields" in kwargs:
            return kwargs["extra_fields"]
        else:
            return {k: v for k, v in kwargs.items() if k not in cls._meta.fields}


class Object(BaseModel):
    """Mongo implementation of the Object class.

    Contains definitions of indexes and custom attributes like loc.
    """

    _id = SpecialField(
        lambda **kwargs: kwargs.get("oid") or kwargs["_id"]
    )  # Survey object ID (unique ID in database)
    aid = Field()  # Alerce object ID (groups objects from different surveys)
    tid = Field()  # List with all telescopes the object has been observed with
    sid = Field()  # List with all surveys which their telescopes observed this obj
    corrected = Field()
    stellar = Field()
    firstmjd = Field()
    lastmjd = Field()
    deltajd = Field()
    ndet = Field()
    meanra = Field()
    sigmara = Field()
    meandec = Field()
    sigmadec = Field()
    loc = SpecialField(
        lambda **kwargs: {
            "type": "Point",
            "coordinates": [kwargs["meanra"] - 180, kwargs["meandec"]],
        }
    )
    magstats = SpecialField(lambda **kwargs: kwargs.get("magstats", []))
    features = SpecialField(lambda **kwargs: kwargs.get("features", {}))
    probabilities = SpecialField(lambda **kwargs: kwargs.get("probabilities", []))
    xmatch = SpecialField(lambda **kwargs: kwargs.get("xmatch", []))

    __table_args__ = [
        IndexModel([("aid", ASCENDING), ("sid", ASCENDING)], name="aid_sid"),
        IndexModel([("lastmjd", DESCENDING)], name="lastmjd"),
        IndexModel([("firstmjd", DESCENDING)], name="firstmjd"),
        IndexModel([("loc", GEOSPHERE)], name="radec"),
        IndexModel(
            [
                ("probabilities.classifier_name", ASCENDING),
                ("probabilities.version", DESCENDING),
                ("probabilities.class_rank_1", DESCENDING),
                ("probabilities.probability_rank_1", DESCENDING),
            ],
            name="probabilities",
        ),
        IndexModel([("features.survey", ASCENDING)], name="features"),
    ]
    __tablename__ = "object"


class Detection(BaseModelWithExtraFields):
    @classmethod
    def create_extra_fields(cls, **kwargs):
        kwargs = super().create_extra_fields(**kwargs)
        kwargs.pop("candid", None)  # Prevents candid being duplicated in extra_fields
        return kwargs

    tid = Field()  # Telescope ID
    sid = Field()  # Survey ID
    aid = Field()  # object alerce identifier
    pid = Field()
    oid = Field()  # object survey identifier
    candid = Field()  # alert identifier
    mjd = Field()
    fid = Field()
    ra = Field()
    e_ra = Field()
    dec = Field()
    e_dec = Field()
    mag = Field()  # magpsf in ZTF alerts
    e_mag = Field()  # sigmapsf in ZTF alerts
    mag_corr = Field()  # magpsf_corr in ZTF alerts
    e_mag_corr = Field()  # sigmapsf_corr in ZTF alerts
    e_mag_corr_ext = Field()  # sigmapsf_corr_ext in ZTF alerts
    isdiffpos = Field()
    corrected = Field()
    dubious = Field()
    parent_candid = Field()
    has_stamp = Field()

    __table_args__ = [
        IndexModel([("aid", ASCENDING), ("sid", ASCENDING)]),
        IndexModel([("oid", ASCENDING), ("candid", ASCENDING)], unique=True),
    ]
    __tablename__ = "detection"


class ForcedPhotometry(BaseModelWithExtraFields):
    @classmethod
    def create_extra_fields(cls, **kwargs):
        kwargs = super().create_extra_fields(**kwargs)
        kwargs.pop("candid", None)  # Prevents candid being duplicated in extra_fields
        return kwargs

    tid = Field()  # Telescope ID
    sid = Field()  # Survey ID
    aid = Field()
    pid = Field()
    oid = Field()
    mjd = Field()
    fid = Field()
    ra = Field()
    e_ra = Field()
    dec = Field()
    e_dec = Field()
    mag = Field()  # magpsf in ZTF alerts
    e_mag = Field()  # sigmapsf in ZTF alerts
    mag_corr = Field()  # magpsf_corr in ZTF alerts
    e_mag_corr = Field()  # sigmapsf_corr in ZTF alerts
    e_mag_corr_ext = Field()  # sigmapsf_corr_ext in ZTF alerts
    isdiffpos = Field()
    corrected = Field()
    dubious = Field()
    parent_candid = Field()
    has_stamp = Field()

    __table_args__ = [
        IndexModel([("aid", ASCENDING), ("sid", ASCENDING)]),
        IndexModel([("oid", ASCENDING), ("pid", ASCENDING)], unique=True),
    ]
    __tablename__ = "forced_photometry"


class NonDetection(BaseModelWithExtraFields):
    @classmethod
    def create_extra_fields(cls, **kwargs):
        kwargs = super().create_extra_fields(**kwargs)
        kwargs.pop("candid", None)  # Prevents candid being duplicated in extra_fields
        return kwargs

    aid = Field()
    tid = Field()
    sid = Field()
    oid = Field()
    mjd = Field()
    fid = Field()
    diffmaglim = Field()

    __table_args__ = [
        IndexModel(
            [("oid", ASCENDING), ("fid", ASCENDING), ("mjd", ASCENDING)],
            name="unique",
            unique=True,
        ),
        IndexModel([("aid", ASCENDING), ("sid", ASCENDING)], name="aid_sid"),
    ]
    __tablename__ = "non_detection"
