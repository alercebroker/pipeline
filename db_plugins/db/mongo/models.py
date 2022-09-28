from pymongo import ASCENDING, DESCENDING, GEOSPHERE, IndexModel
from db_plugins.db.mongo.orm import Field, SpecialField, BaseMetaClass


class BaseModel(dict, metaclass=BaseMetaClass):
    def __init__(self, **kwargs):
        model = {}
        if "_id" in kwargs and "_id" not in self._meta.fields:
            model["_id"] = kwargs["_id"]
        for field, fclass in self._meta.fields.items():
            try:
                if isinstance(fclass, SpecialField):
                    model[field] = fclass.callback(**kwargs)
                else:
                    model[field] = kwargs[field]
            except KeyError:
                raise AttributeError(f"{self.__class__.__name__} model needs {field} attribute")
        super().__init__(**model)

    @classmethod
    def set_database(cls, database):
        cls.metadata.database = database


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

    _id = SpecialField(lambda **kwargs: kwargs.get("aid") or kwargs["_id"])  # ALeRCE object ID (unique ID in database)
    oid = Field()  # Should include OID of all surveys (same survey can have many OIDs)
    tid = Field()  # Should include all telescopes contributing with detections
    lastmjd = Field()
    firstmjd = Field()
    ndet = Field()
    meanra = Field()
    meandec = Field()
    loc = SpecialField(lambda **kwargs: {"type": "Point", "coordinates": [kwargs["meanra"] - 180, kwargs["meandec"]]})
    magstats = SpecialField(lambda **kwargs: kwargs.get("magstats", []))
    features = SpecialField(lambda **kwargs: kwargs.get("features", []))
    probabilities = SpecialField(lambda **kwargs: kwargs.get("probabilities", []))
    xmatch = SpecialField(lambda **kwargs: kwargs.get("xmatch", []))

    __table_args__ = [
        IndexModel([("oid", ASCENDING)]),
        IndexModel([("lastmjd", DESCENDING)]),
        IndexModel([("firstmjd", DESCENDING)]),
        IndexModel([("loc", GEOSPHERE)]),
        IndexModel(
            [("probabilities.classifier_name", ASCENDING),
             ("probabilities.classifier_version", DESCENDING),
             ("probabilities.probability", DESCENDING)],
            partialFilterExpresion={"probabilities.ranking": 1}
        )
    ]
    __tablename__ = "object"


class Detection(BaseModelWithExtraFields):
    @classmethod
    def create_extra_fields(cls, **kwargs):
        kwargs = super().create_extra_fields(**kwargs)
        kwargs.pop("candid", None)  # Prevents repeated candid entry in extra_fields
        return kwargs

    _id = SpecialField(lambda **kwargs: kwargs.get("candid") or kwargs["_id"])
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

    __table_args__ = [IndexModel([("aid", ASCENDING)]), IndexModel([("tid", ASCENDING)])]
    __tablename__ = "detection"


class NonDetection(BaseModelWithExtraFields):

    aid = Field()
    tid = Field()
    oid = Field()
    mjd = Field()
    fid = Field()
    diffmaglim = Field()

    __table_args__ = [
        IndexModel([("aid", ASCENDING)]), IndexModel([("tid", ASCENDING)]),
    ]
    __tablename__ = "non_detection"


class Taxonomy(BaseModel):
    classifier_name = Field()
    classifier_version = Field()
    classes = Field()

    __table_args__ = [
        IndexModel([("classifier_name", ASCENDING),
                    ("classifier_version", DESCENDING)]),
    ]
    __tablename__ = "taxonomy"
