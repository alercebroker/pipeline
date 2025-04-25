from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    SmallInteger,
    VARCHAR,
    String,
    ForeignKey,
    Float,
    Boolean,
    ARRAY,
    Index,
    UniqueConstraint,
    DateTime,
    JSON,
    PrimaryKeyConstraint,
    ForeignKeyConstraint,
)

from sqlalchemy.dialects.postgresql import REAL, DOUBLE_PRECISION

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Commons:
    def __getitem__(self, field):
        return self.__dict__[field]

class Object(Base):
    __tablename__ = "object"

    oid = Column(BigInteger)
    tid = Column(SmallInteger)
    sid = Column(SmallInteger)
    meanra = Column(DOUBLE_PRECISION)
    meandec = Column(DOUBLE_PRECISION)
    sigmara = Column(DOUBLE_PRECISION)
    sigmadec = Column(DOUBLE_PRECISION)
    firstmjd = Column(DOUBLE_PRECISION)
    lastmjd = Column(DOUBLE_PRECISION)
    deltamjd = Column(DOUBLE_PRECISION)
    n_det = Column(Integer)
    n_forced = Column(Integer)
    n_non_det = Column(Integer)
    corrected = Column(Boolean)
    stellar = Column(Boolean)

    __table_args__ = (
        PrimaryKeyConstraint('oid', name='pk_object_oid'),
        Index("ix_object_n_det", "n_det", postgresql_using="btree"),
        Index("ix_object_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("ix_object_lastmjd", "lastmjd", postgresql_using="btree"),
        Index("ix_object_meanra", "meanra", postgresql_using="btree"),
        Index("ix_object_meandec", "meandec", postgresql_using="btree")
    )

class ZtfObject(Base):

    __tablename__ = "ztf_object"

    oid = Column(BigInteger)
    g_r_max = Column(REAL)
    g_r_max_corr = Column(REAL)
    g_r_mean = Column(REAL)
    g_r_mean_corr = Column(REAL)

    __table_args__ = (
        PrimaryKeyConstraint('oid', name='pk_ztfobject_oid'),
    )
    
class Detection(Base):

    __tablename__ = "detection"

    oid = Column(BigInteger)#int8,
    measurement_id = Column(BigInteger) #int8,
    mjd = Column(DOUBLE_PRECISION)#float8,
    ra = Column(DOUBLE_PRECISION)#float8,
    dec = Column(DOUBLE_PRECISION)#float8,
    band = Column(SmallInteger)#int2,


    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_detection_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_detection_oid", "oid", postgresql_using="hash"),
    )

class ZtfDetection(Base):

    __tablename__ = "ztf_detection"

    oid = Column(BigInteger) #int8,
    measurement_id = Column(BigInteger) #int8,
    pid = Column(BigInteger) #int8,
    diffmaglim = Column(REAL) #float4,
    isdiffpos = Column(Integer, nullable=False) #int4 not null,
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
    sigmagapbig = Column(Integer)# float4,
    rfid = Column(BigInteger)# int8,
    rband = Column(BigInteger)# int8,
    magpsf_corr = Column(Integer)# float4,
    sigmapsf_corr = Column(Integer)# float4,
    sigmapsf_corr_ext = Column(Integer)# float4,
    corrected = Column(Boolean)# bool,
    dubious = Column(Boolean)# bool,
    parent_candid = Column(BigInteger)# int8,
    has_stamp = Column(Boolean)# bool,
    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_ztfdetection_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )

class ForcedPhotometry(Base):

    __tablename__ = "forced_photometry"

    oid = Column(BigInteger)# int8,
    measurement_id = Column(BigInteger)# int8,
    mjd = Column(DOUBLE_PRECISION)# float8,
    ra = Column(DOUBLE_PRECISION)# float8,
    dec = Column(DOUBLE_PRECISION)# float8,
    band = Column(SmallInteger)# int2,

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_forcedphotometry_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )

class ZtfForcedPhotometry(Base):

    __tablename__ = "ztf_forced_photometry"

    oid = Column(BigInteger)  # int8,
    pid = Column(BigInteger) #int8,
    measurement_id = Column(BigInteger)  # int8,
    mag = Column(DOUBLE_PRECISION)  # float8
    e_mag = Column(DOUBLE_PRECISION)  # float8,
    mag_corr = Column(DOUBLE_PRECISION)  # float8,
    e_mag_corr = Column(DOUBLE_PRECISION)  # float8,
    e_mag_corr_ext = Column(DOUBLE_PRECISION)  # float8,
    isdiffpos = Column(Integer, nullable=False)  # int4 NOT NULL,
    corrected = Column(Boolean, nullable=False)  # bool NOT NULL,
    dubious = Column(Boolean, nullable=False)  # bool NOT NULL,
    parent_candid = Column(BigInteger)  # varchar,
    has_stamp = Column(Boolean, nullable=False)  # bool NOT NULL,
    field = Column(Integer)  # int4,
    rcid = Column(Integer)  # int4,
    rfid = Column(BigInteger)  # int8,
    rband = Column(BigInteger)# int8,
    sciinpseeing = Column(DOUBLE_PRECISION)  # float8,
    scibckgnd = Column(DOUBLE_PRECISION)  # float8,
    scisigpix = Column(DOUBLE_PRECISION)  # float8,
    magzpsci = Column(DOUBLE_PRECISION)  # float8,
    magzpsciunc = Column(DOUBLE_PRECISION)  # float8,
    magzpscirms = Column(DOUBLE_PRECISION)  # float8,
    clrcoeff = Column(DOUBLE_PRECISION)  # float8,
    clrcounc = Column(DOUBLE_PRECISION)  # float8,
    exptime = Column(DOUBLE_PRECISION)  # float8,
    adpctdif1 = Column(DOUBLE_PRECISION)  # float8,
    adpctdif2 = Column(DOUBLE_PRECISION)  # float8,
    diffmaglim = Column(DOUBLE_PRECISION)  # float8,
    programid = Column(Integer)  # int4,
    procstatus = Column(VARCHAR)  # varchar,
    distnr = Column(DOUBLE_PRECISION)  # float8,
    ranr = Column(DOUBLE_PRECISION)  # float8,
    decnr = Column(DOUBLE_PRECISION)  # float8,
    magnr = Column(DOUBLE_PRECISION)  # float8,
    sigmagnr = Column(DOUBLE_PRECISION)  # float8,
    chinr = Column(DOUBLE_PRECISION)  # float8,
    sharpnr = Column(DOUBLE_PRECISION)  # float8

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_ztfforcedphotometry_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )

class NonDetection(Base):

    __tablename__ = 'non_detection'

    oid = Column(BigInteger)# int8,
    band = Column(SmallInteger)# int2,
    mjd = Column(DOUBLE_PRECISION)# float8,
    diffmaglim = Column(REAL)# float4,

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'mjd', name='pk_oid_mjd'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_non_detection_oid", "oid", postgresql_using="hash"),
    )