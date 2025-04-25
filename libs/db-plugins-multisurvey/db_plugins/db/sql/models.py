from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    SmallInteger,
    VARCHAR,
    Boolean,
    Index,
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

    oid = Column(BigInteger)
    measurement_id = Column(BigInteger) 
    mjd = Column(DOUBLE_PRECISION)
    ra = Column(DOUBLE_PRECISION)
    dec = Column(DOUBLE_PRECISION)
    band = Column(SmallInteger)


    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_detection_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_detection_oid", "oid", postgresql_using="hash"),
    )

class ZtfDetection(Base):

    __tablename__ = "ztf_detection"

    oid = Column(BigInteger) 
    measurement_id = Column(BigInteger) 
    pid = Column(BigInteger) 
    diffmaglim = Column(REAL) 
    isdiffpos = Column(Integer, nullable=False) 
    nid = Column(Integer) 
    magpsf = Column(REAL) 
    sigmapsf = Column(REAL) 
    magap = Column(REAL) 
    sigmagap = Column(REAL) 
    distnr= Column(REAL) 
    rb = Column(REAL)
    rbversion = Column(VARCHAR)
    drb = Column(REAL)
    drbversion = Column(VARCHAR)
    magapbig = Column(REAL)
    sigmagapbig = Column(Integer)
    rfid = Column(BigInteger)
    rband = Column(BigInteger)
    magpsf_corr = Column(Integer)
    sigmapsf_corr = Column(Integer)
    sigmapsf_corr_ext = Column(Integer)
    corrected = Column(Boolean)
    dubious = Column(Boolean)
    parent_candid = Column(BigInteger)
    has_stamp = Column(Boolean)
    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_ztfdetection_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )

class ForcedPhotometry(Base):

    __tablename__ = "forced_photometry"

    oid = Column(BigInteger)
    measurement_id = Column(BigInteger)
    mjd = Column(DOUBLE_PRECISION)
    ra = Column(DOUBLE_PRECISION)
    dec = Column(DOUBLE_PRECISION)
    band = Column(SmallInteger)

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_forcedphotometry_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )

class ZtfForcedPhotometry(Base):

    __tablename__ = "ztf_forced_photometry"

    oid = Column(BigInteger)  
    pid = Column(BigInteger) 
    measurement_id = Column(BigInteger)  
    mag = Column(DOUBLE_PRECISION)  
    e_mag = Column(DOUBLE_PRECISION)  
    mag_corr = Column(DOUBLE_PRECISION)  
    e_mag_corr = Column(DOUBLE_PRECISION)  
    e_mag_corr_ext = Column(DOUBLE_PRECISION)  
    isdiffpos = Column(Integer, nullable=False)  
    corrected = Column(Boolean, nullable=False)  
    dubious = Column(Boolean, nullable=False)  
    parent_candid = Column(BigInteger)  
    has_stamp = Column(Boolean, nullable=False)  
    field = Column(Integer)  
    rcid = Column(Integer)  
    rfid = Column(BigInteger)  
    rband = Column(BigInteger)
    sciinpseeing = Column(DOUBLE_PRECISION)  
    scibckgnd = Column(DOUBLE_PRECISION)  
    scisigpix = Column(DOUBLE_PRECISION)  
    magzpsci = Column(DOUBLE_PRECISION)  
    magzpsciunc = Column(DOUBLE_PRECISION)  
    magzpscirms = Column(DOUBLE_PRECISION)  
    clrcoeff = Column(DOUBLE_PRECISION)  
    clrcounc = Column(DOUBLE_PRECISION)  
    exptime = Column(DOUBLE_PRECISION)  
    adpctdif1 = Column(DOUBLE_PRECISION)  
    adpctdif2 = Column(DOUBLE_PRECISION)  
    diffmaglim = Column(DOUBLE_PRECISION)  
    programid = Column(Integer)  
    procstatus = Column(VARCHAR)  
    distnr = Column(DOUBLE_PRECISION)  
    ranr = Column(DOUBLE_PRECISION)  
    decnr = Column(DOUBLE_PRECISION)  
    magnr = Column(DOUBLE_PRECISION)  
    sigmagnr = Column(DOUBLE_PRECISION)  
    chinr = Column(DOUBLE_PRECISION)  
    sharpnr = Column(DOUBLE_PRECISION)  

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'measurement_id', name='pk_ztfforcedphotometry_oid_measurementid'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )

class NonDetection(Base):

    __tablename__ = 'non_detection'

    oid = Column(BigInteger)
    band = Column(SmallInteger)
    mjd = Column(DOUBLE_PRECISION)
    diffmaglim = Column(REAL)

    __table_args__ = (
        PrimaryKeyConstraint('oid', 'mjd', name='pk_oid_mjd'),
        ForeignKeyConstraint([oid], [Object.oid]),
        Index("ix_non_detection_oid", "oid", postgresql_using="hash"),
    )