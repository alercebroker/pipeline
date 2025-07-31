from sqlalchemy import (
    TIMESTAMP,
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
    sigmara = Column(DOUBLE_PRECISION, nullable=True)
    sigmadec = Column(DOUBLE_PRECISION, nullable=True)
    firstmjd = Column(DOUBLE_PRECISION, nullable=False)
    lastmjd = Column(DOUBLE_PRECISION, nullable=False)
    deltamjd = Column(DOUBLE_PRECISION, nullable=False, default=0.0)
    n_det = Column(Integer, nullable=False, default=1)
    n_forced = Column(Integer, nullable=False, default=1)
    n_non_det = Column(Integer, nullable=False, default=1)
    corrected = Column(Boolean, nullable=False, default=False)
    stellar = Column(Boolean, nullable=True, default=None)

    __table_args__ = (
        PrimaryKeyConstraint("oid", "sid", name="pk_object_oid_sid"),
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


class LsstDiaObject(Base):
    __tablename__ = "lsst_dia_object"

    oid = Column(BigInteger, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstdiaobject_oid"),)


class LsstSsObject(Base):
    __tablename__ = "lsst_ss_object"

    oid = Column(BigInteger, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstssobject_oid"),)


class Detection(Base):
    __tablename__ = "detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger, nullable=False)  # int2,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_detection_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_detection_oid", "oid", postgresql_using="hash"),
    )


class ZtfDetection(Base):
    __tablename__ = "ztf_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    pid = Column(BigInteger)  # int8,
    diffmaglim = Column(REAL)  # float4,
    isdiffpos = Column(Integer)  # bool,
    nid = Column(Integer)  # int4,
    magpsf = Column(REAL)  # float4,
    sigmapsf = Column(REAL)  # float4,
    magap = Column(REAL)  # float4,
    sigmagap = Column(REAL)  # float4,
    distnr = Column(REAL)  # float4,
    rb = Column(REAL)  # float4,
    rbversion = Column(VARCHAR)  # varchar,
    drb = Column(REAL)  # float4,
    drbversion = Column(VARCHAR)  # varchar,
    magapbig = Column(REAL)  # float4,
    sigmagapbig = Column(REAL)  # float4,
    rfid = Column(BigInteger)  # int8,
    magpsf_corr = Column(Integer)  # float4,
    sigmapsf_corr = Column(Integer)  # float4,
    sigmapsf_corr_ext = Column(Integer)  # float4,
    corrected = Column(Boolean)  # bool,
    dubious = Column(Boolean)  # bool,
    parent_candid = Column(BigInteger)  # int8,
    has_stamp = Column(Boolean)  # bool,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfdetection_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )


class LsstDetection(Base):
    __tablename__ = "lsst_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    parentDiaSourceId = Column(BigInteger)

    psfFlux = Column(REAL)
    psfFluxErr = Column(REAL)

    psfFlux_flag = Column(Boolean)
    psfFlux_flag_edge = Column(Boolean)
    psfFlux_flag_noGoodPixels = Column(Boolean)

    raErr = Column(REAL)
    decErr = Column(REAL)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstdetection_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_lsstdetection_oid", "oid", postgresql_using="hash"),
    )


class ForcedPhotometry(Base):
    __tablename__ = "forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger, nullable=False)  # int2,

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_forcedphotometry_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfForcedPhotometry(Base):
    __tablename__ = "ztf_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    pid = Column(BigInteger)  # int8
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
    sciinpseeing = Column(REAL, nullable=False)  # float8,
    scibckgnd = Column(REAL, nullable=False)  # float8,
    scisigpix = Column(REAL, nullable=False)  # float8,
    magzpsci = Column(REAL, nullable=False)  # float8,
    magzpsciunc = Column(REAL, nullable=False)  # float8,
    magzpscirms = Column(REAL, nullable=False)  # float8,
    clrcoeff = Column(REAL, nullable=False)  # float8,
    clrcounc = Column(REAL, nullable=False)  # float8,
    exptime = Column(REAL, nullable=False)  # float8,
    adpctdif1 = Column(REAL, nullable=False)  # float8,
    adpctdif2 = Column(REAL, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float8,
    programid = Column(Integer, nullable=False)  # int4,
    procstatus = Column(VARCHAR, nullable=False)  # varchar,
    distnr = Column(REAL, nullable=False)  # float8,
    ranr = Column(DOUBLE_PRECISION, nullable=False)  # float8,<---
    decnr = Column(DOUBLE_PRECISION, nullable=False)  # float8,<---
    magnr = Column(REAL, nullable=False)  # float8,
    sigmagnr = Column(REAL, nullable=False)  # float8,
    chinr = Column(REAL, nullable=False)  # float8,
    sharpnr = Column(REAL, nullable=False)  # float8

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfforcedphotometry_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class LsstForcedPhotometry(Base):
    __tablename__ = "lsst_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,

    visit = Column(BigInteger, nullable=False)
    detector = Column(Integer, nullable=False)

    psfFlux = Column(REAL, nullable=False)
    psfFluxErr = Column(REAL, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstforcedphotometry_oid_measurementid"
        ),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_lsst_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class NonDetection(Base):
    __tablename__ = "non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    band = Column(SmallInteger, nullable=False)  # int2,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float4,

    __table_args__ = (
        PrimaryKeyConstraint("oid", "mjd", name="pk_oid_mjd"),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_non_detection_oid", "oid", postgresql_using="hash"),
    )


class LsstNonDetection(Base):
    __tablename__ = "lsst_non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    ccdVisitId = Column(BigInteger, nullable=False)
    band = Column(SmallInteger, nullable=False)
    mjd = Column(DOUBLE_PRECISION, nullable=False)
    diaNoise = Column(REAL, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("oid", "mjd", name="pk_lsstnondetection_oid_mjd"),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
        Index("ix_lsst_non_detection_oid", "oid", postgresql_using="hash"),
    )


class ZtfSS(Base):
    __tablename__ = "ztf_ss"

    oid = Column(BigInteger, nullable=False)
    measurement_id = Column(BigInteger, nullable=False)
    ssdistnr = Column(REAL)
    ssmagnr = Column(REAL)
    ssnamenr = Column(VARCHAR)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfss_oid_measurement_id"
        ),
        Index("ix_zrt_ss_oid", "oid", postgresql_using="btree"),
    )


class ZtfPS1(Base):
    __tablename__ = "ztf_ps1"

    oid = Column(BigInteger, nullable=False)
    measurement_id = Column(BigInteger, nullable=False)
    objectidps1 = Column(BigInteger)
    sgmag1 = Column(REAL)
    srmag1 = Column(REAL)
    simag1 = Column(REAL)
    szmag1 = Column(REAL)
    sgscore1 = Column(REAL)
    distpsnr1 = Column(REAL)
    objectidps2 = Column(BigInteger)
    sgmag2 = Column(REAL)
    srmag2 = Column(REAL)
    simag2 = Column(REAL)
    szmag2 = Column(REAL)
    sgscore2 = Column(REAL)
    distpsnr2 = Column(REAL)
    objectidps3 = Column(BigInteger)
    sgmag3 = Column(REAL)
    srmag3 = Column(REAL)
    simag3 = Column(REAL)
    szmag3 = Column(REAL)
    sgscore3 = Column(REAL)
    distpsnr3 = Column(REAL)
    nmtchps = Column(SmallInteger)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfps1_oid_measurement_id"
        ),
        Index("ix_ztf_ps1_oid", "oid", postgresql_using="btree"),
    )


class ZtfGaia(Base):
    __tablename__ = "ztf_gaia"

    oid = Column(BigInteger, nullable=False)
    measurement_id = Column(BigInteger, nullable=False)
    neargaia = Column(REAL)
    neargaiabright = Column(REAL)
    maggaia = Column(REAL)
    maggaiabright = Column(REAL)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_ztfgaia_oid"),)


class ZtfDataquality(Base):
    __tablename__ = "ztf_dataquality"

    oid = Column(BigInteger, nullable=False)
    measurement_id = Column(BigInteger, nullable=False)
    xpos = Column(REAL)
    ypos = Column(REAL)
    chipsf = Column(REAL)
    sky = Column(REAL)
    fwhm = Column(REAL)
    classtar = Column(REAL)
    mindtoedge = Column(REAL)
    seeratio = Column(REAL)
    aimage = Column(REAL)
    bimage = Column(REAL)
    aimagerat = Column(REAL)
    bimagerat = Column(REAL)
    nneg = Column(Integer)
    nbad = Column(Integer)
    sumrat = Column(REAL)
    scorr = Column(REAL)
    dsnrms = Column(REAL)
    ssnrms = Column(REAL)
    magzpsci = Column(REAL)
    magzpsciunc = Column(REAL)
    magzpscirms = Column(REAL)
    nmatches = Column(Integer)
    clrcoeff = Column(REAL)
    clrcounc = Column(REAL)
    zpclrcov = Column(REAL)
    zpmed = Column(REAL)
    clrmed = Column(REAL)
    clrrms = Column(REAL)
    exptime = Column(REAL)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_ztfdataquality_oid_measurement_id"
        ),
        Index("ix_ztf_dataquality_oid", "oid", postgresql_using="btree"),
        Index(
            "ix_ztf_dataquality_measurement_id",
            "measurement_id",
            postgresql_using="btree",
        ),
    )


class ZtfReference(Base):
    __tablename__ = "ztf_reference"

    oid = Column(BigInteger, nullable=False)
    rfid = Column(BigInteger, nullable=False)
    measurement_id = Column(BigInteger, nullable=False)
    band = Column(Integer)
    rcid = Column(Integer)
    field = Column(Integer)
    magnr = Column(REAL)
    sigmagnr = Column(REAL)
    chinr = Column(REAL)
    sharpnr = Column(REAL)
    ranr = Column(DOUBLE_PRECISION)
    decnr = Column(DOUBLE_PRECISION)
    mjdstartref = Column(DOUBLE_PRECISION)
    mjdendref = Column(DOUBLE_PRECISION)
    nframesref = Column(Integer)

    __table_args__ = (
        PrimaryKeyConstraint("oid", "rfid", name="pk_ztfreference_oid_rfid"),
        Index("ix_ztf_reference_oid", "oid", postgresql_using="btree"),
    )


class MagStat(Base):
    __tablename__ = "magstat"

    oid = Column(BigInteger, nullable=False)  # int8
    sid = Column(SmallInteger, nullable=False)  # int2,
    band = Column(SmallInteger, nullable=False)  # int2
    stellar = Column(Boolean)  # bool
    corrected = Column(Boolean)  # bool
    ndubious = Column(BigInteger)  # int8
    dmdt_first = Column(BigInteger)  # int8
    dm_first = Column(BigInteger)  # int8
    sigmadm_first = Column(BigInteger)  # int8
    dt_first = Column(BigInteger)  # int8
    magmean = Column(DOUBLE_PRECISION)  # float8
    magmedian = Column(DOUBLE_PRECISION)  # float8
    magmax = Column(DOUBLE_PRECISION)  # float8
    magmin = Column(DOUBLE_PRECISION)  # float8
    magsigma = Column(DOUBLE_PRECISION)  # float8
    maglast = Column(BigInteger)  # int8
    magfirst = Column(BigInteger)  # int8
    magmean_corr = Column(DOUBLE_PRECISION)  # float8
    magmedian_corr = Column(DOUBLE_PRECISION)  # float8
    magmax_corr = Column(DOUBLE_PRECISION)  # float8
    magmin_corr = Column(DOUBLE_PRECISION)  # float8
    magsigma_corr = Column(DOUBLE_PRECISION)  # float8
    maglast_corr = Column(DOUBLE_PRECISION)  # float8
    magfirst_corr = Column(DOUBLE_PRECISION)  # float8
    step_id_corr = Column(VARCHAR)  # varchar
    saturation_rate = Column(DOUBLE_PRECISION)  # float8
    last_update = Column(TIMESTAMP)  # timestamp

    __table_args__ = (
        PrimaryKeyConstraint("oid", "band", name="pk_magstat_oid_band"),
        ForeignKeyConstraint([oid, sid], [Object.oid, Object.sid]),
    )


class LsstIdMapper(Base):
    __tablename__ = "lsst_idmapper"

    lsst_id_serial = Column(BigInteger, autoincrement=True)
    lsst_diaObjectId = Column(BigInteger)

    __table_args__ = (
        PrimaryKeyConstraint("lsst_id_serial", name="pk_lsst_idmapper_serial"),
    )

class Probability(Base):
    __tablename__ = "probability"

    oid = Column(BigInteger)
    sid = Column(SmallInteger, nullable=False)
    classifier_id = Column(SmallInteger)
    classifier_version = Column(SmallInteger)
    class_id = Column(SmallInteger, nullable=False)
    probability = Column(REAL, nullable=False)
    ranking = Column(SmallInteger)
    lastmjd = Column(DOUBLE_PRECISION, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "classifier_id", "classifier_version", "class_id",
            name="pk_probability_oid_classifierid_classid"
        ),
        Index("ix_probability_oid", "oid", postgresql_using="hash"),
    )