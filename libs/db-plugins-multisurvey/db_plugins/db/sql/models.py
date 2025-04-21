from sqlalchemy import (
    VARCHAR,
    BigInteger,
    Boolean,
    Column,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, REAL
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Commons:
    def __getitem__(self, field):
        return self.__dict__[field]


class Object(Base):
    __tablename__ = "object"

    oid = Column(BigInteger, nullable=False)
    tid = Column(SmallInteger, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    meanra = Column(DOUBLE_PRECISION, nullable=False)
    meandec = Column(DOUBLE_PRECISION, nullable=False)
    sigmara = Column(DOUBLE_PRECISION)
    sigmadec = Column(DOUBLE_PRECISION)
    firstmjd = Column(DOUBLE_PRECISION, nullable=False)
    lastmjd = Column(DOUBLE_PRECISION, nullable=False)
    deltamjd = Column(DOUBLE_PRECISION, nullable=False)
    n_det = Column(Integer, nullable=False)
    n_forced = Column(Integer, nullable=False)
    n_non_det = Column(Integer, nullable=False)
    corrected = Column(Boolean, nullable=False)
    stellar = Column(Boolean, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("oid", name="pk_object_oid"),
        Index("ix_object_n_det", "n_det", postgresql_using="btree"),
        Index("ix_object_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("ix_object_lastmjd", "lastmjd", postgresql_using="btree"),
        Index("ix_object_meanra", "meanra", postgresql_using="btree"),
        Index("ix_object_meandec", "meandec", postgresql_using="btree"),
    )


class ZtfObject(Base):
    __tablename__ = "ztf_object"

    oid = Column(BigInteger, nullable=False)
    g_r_max = Column(REAL)
    g_r_max_corr = Column(REAL)
    g_r_mean = Column(REAL)
    g_r_mean_corr = Column(REAL)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_ztfobject_oid"),)


class Detection(Base):
    __tablename__ = "detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger, nullable=False)  # int2,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_detection_oid_measurementid"
        ),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_detection_oid", "oid", postgresql_using="hash"),
    )


class ZtfDetection(Base):
    __tablename__ = "ztf_detection"

    oid = Column(BigInteger, nullable=False) #int8,
    measurement_id = Column(BigInteger, nullable=False) #int8,
    pid = Column(BigInteger) #int8,
    diffmaglim = Column(REAL) #float4,
    isdiffpos = Column(Integer) #bool,
    nid = Column(Integer) #int4,
    magpsf = Column(REAL) #float4,
    sigmapsf = Column(REAL) # float4,
    magap = Column(REAL) # float4,
    sigmagap = Column(REAL) # float4,
    distnr= Column(REAL) # float4,
    rb = Column(REAL)# float4,
    rbversion = Column(VARCHAR)# varchar,
    drb = Column(REAL)# float4,
    drbversion = Column(VARCHAR)# varchar,
    magapbig = Column(REAL)# float4,
    sigmagapbig = Column(REAL)# float4,
    rfid = Column(BigInteger)# int8,
    magpsf_corr = Column(Integer)# float4,
    sigmapsf_corr = Column(Integer)# float4,
    sigmapsf_corr_ext = Column(Integer)# float4,
    corrected = Column(Boolean)# bool,
    dubious = Column(Boolean)# bool,
    parent_candid = Column(BigInteger)# int8,
    has_stamp = Column(Boolean)# bool,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfdetection_oid_measurementid"
        ),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )


class ForcedPhotometry(Base):
    __tablename__ = "forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger, nullable=False)  # int2,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_forcedphotometry_oid_measurementid"
        ),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfForcedPhotometry(Base):
    __tablename__ = "ztf_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    pid = Column(BigInteger) # int8
    mag = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    e_mag = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    mag_corr = Column(DOUBLE_PRECISION)  # float8,
    e_mag_corr = Column(DOUBLE_PRECISION)  # float8,
    e_mag_corr_ext = Column(DOUBLE_PRECISION)  # float8,
    isdiffpos = Column(Integer, nullable=False)  # int4 NOT NULL,
    corrected = Column(Boolean, nullable=False)  # bool NOT NULL,
    dubious = Column(Boolean, nullable=False)  # bool NOT NULL,
    parent_candid = Column(BigInteger)  # varchar,
    has_stamp = Column(Boolean, nullable=False)  # bool NOT NULL,
    field = Column(Integer, nullable=False)  # int4,
    rcid = Column(Integer, nullable=False)  # int4,
    rfid = Column(BigInteger, nullable=False)  # int8,
    sciinpseeing = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    scibckgnd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    scisigpix = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    magzpsci = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    magzpsciunc = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    magzpscirms = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    clrcoeff = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    clrcounc = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    exptime = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    adpctdif1 = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    adpctdif2 = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    programid = Column(Integer, nullable=False)  # int4,
    procstatus = Column(VARCHAR, nullable=False)  # varchar,
    distnr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ranr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    decnr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    magnr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    sigmagnr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    chinr = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    sharpnr = Column(DOUBLE_PRECISION, nullable=False)  # float8

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfforcedphotometry_oid_measurementid"
        ),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class NonDetection(Base):
    __tablename__ = "non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    band = Column(SmallInteger, nullable=False)  # int2,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float4,

    __table_args__ = (
        PrimaryKeyConstraint("oid", "mjd", name="pk_oid_mjd"),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_non_detection_oid", "oid", postgresql_using="hash"),
    )
