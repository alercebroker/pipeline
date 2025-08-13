from sqlalchemy import (
    TIMESTAMP,
    VARCHAR,
    BigInteger,
    Boolean,
    Column,
    Index,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    DateTime,
    func,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, REAL
from sqlalchemy.orm import DeclarativeBase
"""

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

    # all fields 7.4
    ra_dec_Cov = Column(REAL)
    pmRa = Column(REAL)
    pmRaErr = Column(REAL)
    pmDec = Column(REAL)
    pmDecErr = Column(REAL)
    parallax = Column(REAL)
    parallaxErr = Column(REAL)
    pmRa_pmDec_Cov = Column(REAL)
    pmRa_parallax_Cov = Column(REAL)
    pmDec_parallax_Cov = Column(REAL)
    pmParallaxLnL = Column(REAL)
    pmParallaxChi2 = Column(REAL)
    pmParallaxNdata = Column(Integer)
    u_psfFluxMean = Column(REAL)
    u_psfFluxMeanErr = Column(REAL)
    u_psfFluxSigma = Column(REAL)
    u_psfFluxChi2 = Column(REAL)
    u_psfFluxNdata = Column(Integer)
    u_fpFluxMean = Column(REAL)
    u_fpFluxMeanErr = Column(REAL)
    u_fpFluxSigma = Column(REAL)
    g_psfFluxMean = Column(REAL)
    g_psfFluxMeanErr = Column(REAL)
    g_psfFluxSigma = Column(REAL)
    g_psfFluxChi2 = Column(REAL)
    g_psfFluxNdata = Column(Integer)
    g_fpFluxMean = Column(REAL)
    g_fpFluxMeanErr = Column(REAL)
    g_fpFluxSigma = Column(REAL)
    r_psfFluxMean = Column(REAL)
    r_psfFluxMeanErr = Column(REAL)
    r_psfFluxSigma = Column(REAL)
    r_psfFluxChi2 = Column(REAL)
    r_psfFluxNdata = Column(Integer)
    r_fpFluxMean = Column(REAL)
    r_fpFluxMeanErr = Column(REAL)
    r_fpFluxSigma = Column(REAL)
    i_psfFluxMean = Column(REAL)
    i_psfFluxMeanErr = Column(REAL)
    i_psfFluxSigma = Column(REAL)
    i_psfFluxChi2 = Column(REAL)
    i_psfFluxNdata = Column(Integer)
    i_fpFluxMean = Column(REAL)
    i_fpFluxMeanErr = Column(REAL)
    i_fpFluxSigma = Column(REAL)
    z_psfFluxMean = Column(REAL)
    z_psfFluxMeanErr = Column(REAL)
    z_psfFluxSigma = Column(REAL)
    z_psfFluxChi2 = Column(REAL)
    z_psfFluxNdata = Column(Integer)
    z_fpFluxMean = Column(REAL)
    z_fpFluxMeanErr = Column(REAL)
    z_fpFluxSigma = Column(REAL)
    y_psfFluxMean = Column(REAL)
    y_psfFluxMeanErr = Column(REAL)
    y_psfFluxSigma = Column(REAL)
    y_psfFluxChi2 = Column(REAL)
    y_psfFluxNdata = Column(Integer)
    y_fpFluxMean = Column(REAL)
    y_fpFluxMeanErr = Column(REAL)
    y_fpFluxSigma = Column(REAL)
    nearbyObj1 = Column(BigInteger)
    nearbyObj1Dist = Column(REAL)
    nearbyObj1LnP = Column(REAL)
    nearbyObj2 = Column(BigInteger)
    nearbyObj2Dist = Column(REAL)
    nearbyObj2LnP = Column(REAL)
    nearbyObj3 = Column(BigInteger)
    nearbyObj3Dist = Column(REAL)
    nearbyObj3LnP = Column(REAL)
    u_psfFluxErrMean = Column(REAL)
    g_psfFluxErrMean = Column(REAL)
    r_psfFluxErrMean = Column(REAL)
    i_psfFluxErrMean = Column(REAL)
    z_psfFluxErrMean = Column(REAL)
    y_psfFluxErrMean = Column(REAL)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstdiaobject_oid"),)


class LsstSsObject(Base):
    __tablename__ = "lsst_ss_object"

    oid = Column(BigInteger, nullable=False)

    # all fields 7.4
    discoverySubmissionDate = Column(DOUBLE_PRECISION)
    firstObservationDate = Column(DOUBLE_PRECISION)
    arc = Column(REAL)
    numObs = Column(Integer)
    MOID = Column(REAL)
    MOIDTrueAnomaly = Column(REAL)
    MOIDEclipticLongitude = Column(REAL)
    MOIDDeltaV = Column(REAL)
    u_H = Column(REAL)
    u_G12 = Column(REAL)
    u_HErr = Column(REAL)
    u_G12Err = Column(REAL)
    u_H_u_G12_Cov = Column(REAL)
    u_Chi2 = Column(REAL)
    u_Ndata = Column(Integer)
    g_H = Column(REAL)
    g_G12 = Column(REAL)
    g_HErr = Column(REAL)
    g_G12Err = Column(REAL)
    g_H_g_G12_Cov = Column(REAL)
    g_Chi2 = Column(REAL)
    g_Ndata = Column(Integer)
    r_H = Column(REAL)
    r_G12 = Column(REAL)
    r_HErr = Column(REAL)
    r_G12Err = Column(REAL)
    r_H_r_G12_Cov = Column(REAL)
    r_Chi2 = Column(REAL)
    r_Ndata = Column(Integer)
    i_H = Column(REAL)
    i_G12 = Column(REAL)
    i_HErr = Column(REAL)
    i_G12Err = Column(REAL)
    i_H_i_G12_Cov = Column(REAL)
    i_Chi2 = Column(REAL)
    i_Ndata = Column(Integer)
    z_H = Column(REAL)
    z_G12 = Column(REAL)
    z_HErr = Column(REAL)
    z_G12Err = Column(REAL)
    z_H_z_G12_Cov = Column(REAL)
    z_Chi2 = Column(REAL)
    z_Ndata = Column(Integer)
    y_H = Column(REAL)
    y_G12 = Column(REAL)
    y_HErr = Column(REAL)
    y_G12Err = Column(REAL)
    y_H_y_G12_Cov = Column(REAL)
    y_Chi2 = Column(REAL)
    y_Ndata = Column(Integer)
    medianExtendedness = Column(REAL)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstssobject_oid"),)


class Detection(Base):
    __tablename__ = "detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger)  # int2, nulleable because  original schema

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", "sid", name="pk_detection_oid_measurementid_sid"
        ),
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
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )


class LsstDetection(Base):
    __tablename__ = "lsst_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    parentDiaSourceId = Column(BigInteger)

    # psfFlux = Column(REAL)
    # psfFluxErr = Column(REAL)

    # psfFlux_flag = Column(Boolean)
    # psfFlux_flag_edge = Column(Boolean)
    # psfFlux_flag_noGoodPixels = Column(Boolean)

    # raErr = Column(REAL)
    # decErr = Column(REAL)

    # all the fields 7.4
    visit = Column(BigInteger)
    detector = Column(Integer)
    raErr = Column(REAL)
    decErr = Column(REAL)
    ra_dec_Cov = Column(REAL)
    x = Column(REAL)
    xErr = Column(REAL)
    y = Column(REAL)
    yErr = Column(REAL)
    x_y_Cov = Column(REAL)
    centroid_flag = Column(Boolean)
    apFlux = Column(REAL)
    apFluxErr = Column(REAL)
    apFlux_flag = Column(Boolean)
    apFlux_flag_apertureTruncated = Column(Boolean)
    is_negative = Column(Boolean)
    snr = Column(REAL)
    psfFlux = Column(REAL)
    psfFluxErr = Column(REAL)
    psfRa = Column(DOUBLE_PRECISION)
    psfRaErr = Column(REAL)
    psfDec = Column(DOUBLE_PRECISION)
    psfDecErr = Column(REAL)
    psfFlux_psfRa_Cov = Column(REAL)
    psfFlux_psfDec_Cov = Column(REAL)
    psfRa_psfDec_Cov = Column(REAL)
    psfLnL = Column(REAL)
    psfChi2 = Column(REAL)
    psfNdata = Column(Integer)
    psfFlux_flag = Column(Boolean)
    psfFlux_flag_edge = Column(Boolean)
    psfFlux_flag_noGoodPixels = Column(Boolean)
    trailFlux = Column(REAL)
    trailFluxErr = Column(REAL)
    trailRa = Column(DOUBLE_PRECISION)
    trailRaErr = Column(REAL)
    trailDec = Column(DOUBLE_PRECISION)
    trailDecErr = Column(REAL)
    trailLength = Column(REAL)
    trailLengthErr = Column(REAL)
    trailAngle = Column(REAL)
    trailAngleErr = Column(REAL)
    trailFlux_trailRa_Cov = Column(REAL)
    trailFlux_trailDec_Cov = Column(REAL)
    trailFlux_trailLength_Cov = Column(REAL)
    trailFlux_trailAngle_Cov = Column(REAL)
    trailRa_trailDec_Cov = Column(REAL)
    trailRa_trailLength_Cov = Column(REAL)
    trailRa_trailAngle_Cov = Column(REAL)
    trailDec_trailLength_Cov = Column(REAL)
    trailDec_trailAngle_Cov = Column(REAL)
    trailLength_trailAngle_Cov = Column(REAL)
    trailLnL = Column(REAL)
    trailChi2 = Column(REAL)
    trailNdata = Column(Integer)
    trail_flag_edge = Column(Boolean)
    dipoleMeanFlux = Column(REAL)
    dipoleMeanFluxErr = Column(REAL)
    dipoleFluxDiff = Column(REAL)
    dipoleFluxDiffErr = Column(REAL)
    dipoleRa = Column(DOUBLE_PRECISION)
    dipoleRaErr = Column(REAL)
    dipoleDec = Column(DOUBLE_PRECISION)
    dipoleDecErr = Column(REAL)
    dipoleLength = Column(REAL)
    dipoleLengthErr = Column(REAL)
    dipoleAngle = Column(REAL)
    dipoleAngleErr = Column(REAL)
    dipoleMeanFlux_dipoleFluxDiff_Cov = Column(REAL)
    dipoleMeanFlux_dipoleRa_Cov = Column(REAL)
    dipoleMeanFlux_dipoleDec_Cov = Column(REAL)
    dipoleMeanFlux_dipoleLength_Cov = Column(REAL)
    dipoleMeanFlux_dipoleAngle_Cov = Column(REAL)
    dipoleFluxDiff_dipoleRa_Cov = Column(REAL)
    dipoleFluxDiff_dipoleDec_Cov = Column(REAL)
    dipoleFluxDiff_dipoleLength_Cov = Column(REAL)
    dipoleFluxDiff_dipoleAngle_Cov = Column(REAL)
    dipoleRa_dipoleDec_Cov = Column(REAL)
    dipoleRa_dipoleLength_Cov = Column(REAL)
    dipoleRa_dipoleAngle_Cov = Column(REAL)
    dipoleDec_dipoleLength_Cov = Column(REAL)
    dipoleDec_dipoleAngle_Cov = Column(REAL)
    dipoleLength_dipoleAngle_Cov = Column(REAL)
    dipoleLnL = Column(REAL)
    dipoleChi2 = Column(REAL)
    dipoleNdata = Column(Integer)
    forced_PsfFlux_flag = Column(Boolean)
    forced_PsfFlux_flag_edge = Column(Boolean)
    forced_PsfFlux_flag_noGoodPixels = Column(Boolean)
    snapDiffFlux = Column(REAL)
    snapDiffFluxErr = Column(REAL)
    fpBkgd = Column(REAL)
    fpBkgdErr = Column(REAL)
    ixx = Column(REAL)
    ixxErr = Column(REAL)
    iyy = Column(REAL)
    iyyErr = Column(REAL)
    ixy = Column(REAL)
    ixyErr = Column(REAL)
    ixx_iyy_Cov = Column(REAL)
    ixx_ixy_Cov = Column(REAL)
    iyy_ixy_Cov = Column(REAL)
    ixxPSF = Column(REAL)
    iyyPSF = Column(REAL)
    ixyPSF = Column(REAL)
    shape_flag = Column(Boolean)
    shape_flag_no_pixels = Column(Boolean)
    shape_flag_not_contained = Column(Boolean)
    shape_flag_parent_source = Column(Boolean)
    extendedness = Column(REAL)
    reliability = Column(REAL)
    dipoleFitAttempted = Column(Boolean)
    pixelFlags = Column(Boolean)
    pixelFlags_bad = Column(Boolean)
    pixelFlags_cr = Column(Boolean)
    pixelFlags_crCenter = Column(Boolean)
    pixelFlags_edge = Column(Boolean)
    pixelFlags_nodata = Column(Boolean)
    pixelFlags_nodataCenter = Column(Boolean)
    pixelFlags_interpolated = Column(Boolean)
    pixelFlags_interpolatedCenter = Column(Boolean)
    pixelFlags_offimage = Column(Boolean)
    pixelFlags_saturated = Column(Boolean)
    pixelFlags_saturatedCenter = Column(Boolean)
    pixelFlags_suspect = Column(Boolean)
    pixelFlags_suspectCenter = Column(Boolean)
    pixelFlags_streak = Column(Boolean)
    pixelFlags_streakCenter = Column(Boolean)
    pixelFlags_injected = Column(Boolean)
    pixelFlags_injectedCenter = Column(Boolean)
    pixelFlags_injected_template = Column(Boolean)
    pixelFlags_injected_templateCenter = Column(Boolean)


    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstdetection_oid_measurementid"
        ),
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
    band = Column(SmallInteger)  # int2, nulleable original schema

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", "sid", name="pk_forcedphotometry_oid_measurementid_sid"
        ),
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
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class LsstForcedPhotometry(Base):
    __tablename__ = "lsst_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,

    visit = Column(BigInteger, nullable=False)
    detector = Column(Integer, nullable=False)

    psfFlux = Column(REAL)
    psfFluxErr = Column(REAL)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstforcedphotometry_oid_measurementid"
        ),
        Index("ix_lsst_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfNonDetection(Base):
    __tablename__ = "ztf_non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    band = Column(SmallInteger, nullable=False)  # int2,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float4,

    __table_args__ = (
        PrimaryKeyConstraint("oid", "mjd", name="pk_oid_mjd"),
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
        PrimaryKeyConstraint("oid", "sid", "band", name="pk_magstat_oid_sid_band"),
    )

class classifier(Base):
    __tablename__ = "classifier"
    classifier_id = Column(Integer, primary_key=True)
    classifier_name = Column(VARCHAR)
    classifier_version = Column(SmallInteger)

class Taxonomy(Base):
    __tablename__ = "taxonomy"
    class_id = Column(Integer, primary_key=True)
    class_name = Column(VARCHAR)
    order = Column(Integer)
    classifier_id = Column(SmallInteger)

"""


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


# Ommiting lsst ss object for now

class LsstDiaObject(Base):
    __tablename__ = "lsst_dia_object"

    oid = Column(BigInteger, nullable=False)

    # all fields 8.0 draft
    # diaObjectId = Column(BigInteger, nullable=False) => This is oid
    validityStart = Column(DateTime, nullable=False)
    ra = Column(DOUBLE_PRECISION, nullable=False)
    raErr = Column(REAL, nullable=True)
    dec = Column(DOUBLE_PRECISION, nullable=False)
    decErr = Column(REAL, nullable=True)
    ra_dec_Cov = Column(REAL, nullable=True)
    u_psfFluxMean = Column(REAL, nullable=True)
    u_psfFluxMeanErr = Column(REAL, nullable=True)
    u_psfFluxSigma = Column(REAL, nullable=True)
    u_psfFluxChi2 = Column(REAL, nullable=True)
    u_psfFluxNdata = Column(Integer, nullable=True)
    u_fpFluxMean = Column(REAL, nullable=True)
    u_fpFluxMeanErr = Column(REAL, nullable=True)
    g_psfFluxMean = Column(REAL, nullable=True)
    g_psfFluxMeanErr = Column(REAL, nullable=True)
    g_psfFluxSigma = Column(REAL, nullable=True)
    g_psfFluxChi2 = Column(REAL, nullable=True)
    g_psfFluxNdata = Column(Integer, nullable=True)
    g_fpFluxMean = Column(REAL, nullable=True)
    g_fpFluxMeanErr = Column(REAL, nullable=True)
    r_psfFluxMean = Column(REAL, nullable=True)
    r_psfFluxMeanErr = Column(REAL, nullable=True)
    r_psfFluxSigma = Column(REAL, nullable=True)
    r_psfFluxChi2 = Column(REAL, nullable=True)
    r_psfFluxNdata = Column(Integer, nullable=True)
    r_fpFluxMean = Column(REAL, nullable=True)
    r_fpFluxMeanErr = Column(REAL, nullable=True)
    i_psfFluxMean = Column(REAL, nullable=True)
    i_psfFluxMeanErr = Column(REAL, nullable=True)
    i_psfFluxSigma = Column(REAL, nullable=True)
    i_psfFluxChi2 = Column(REAL, nullable=True)
    i_psfFluxNdata = Column(Integer, nullable=True)
    i_fpFluxMean = Column(REAL, nullable=True)
    i_fpFluxMeanErr = Column(REAL, nullable=True)
    z_psfFluxMean = Column(REAL, nullable=True)
    z_psfFluxMeanErr = Column(REAL, nullable=True)
    z_psfFluxSigma = Column(REAL, nullable=True)
    z_psfFluxChi2 = Column(REAL, nullable=True)
    z_psfFluxNdata = Column(Integer, nullable=True)
    z_fpFluxMean = Column(REAL, nullable=True)
    z_fpFluxMeanErr = Column(REAL, nullable=True)
    y_psfFluxMean = Column(REAL, nullable=True)
    y_psfFluxMeanErr = Column(REAL, nullable=True)
    y_psfFluxSigma = Column(REAL, nullable=True)
    y_psfFluxChi2 = Column(REAL, nullable=True)
    y_psfFluxNdata = Column(Integer, nullable=True)
    y_fpFluxMean = Column(REAL, nullable=True)
    y_fpFluxMeanErr = Column(REAL, nullable=True)
    u_scienceFluxMean = Column(REAL, nullable=True)
    u_scienceFluxMeanErr = Column(REAL, nullable=True)
    u_scienceFluxSigma = Column(REAL, nullable=True)
    g_scienceFluxMean = Column(REAL, nullable=True)
    g_scienceFluxMeanErr = Column(REAL, nullable=True)
    g_scienceFluxSigma = Column(REAL, nullable=True)
    r_scienceFluxMean = Column(REAL, nullable=True)
    r_scienceFluxMeanErr = Column(REAL, nullable=True)
    r_scienceFluxSigma = Column(REAL, nullable=True)  
    i_scienceFluxMean = Column(REAL, nullable=True)
    i_scienceFluxMeanErr = Column(REAL, nullable=True)
    i_scienceFluxSigma = Column(REAL, nullable=True)
    z_scienceFluxMean = Column(REAL, nullable=True)
    z_scienceFluxMeanErr = Column(REAL, nullable=True)
    z_scienceFluxSigma = Column(REAL, nullable=True)
    y_scienceFluxMean = Column(REAL, nullable=True)
    y_scienceFluxMeanErr = Column(REAL, nullable=True)
    y_scienceFluxSigma = Column(REAL, nullable=True)
    u_psfFluxSkew = Column(REAL, nullable=True)
    u_psfFluxMin = Column(REAL, nullable=True)
    u_psfFluxMax = Column(REAL, nullable=True)
    u_psfFluxMaxSlope = Column(REAL, nullable=True)
    u_psfFluxErrMean = Column(REAL, nullable=True)
    g_psfFluxSkew = Column(REAL, nullable=True)
    g_psfFluxMin = Column(REAL, nullable=True)
    g_psfFluxMax = Column(REAL, nullable=True)
    g_psfFluxMaxSlope = Column(REAL, nullable=True)
    g_psfFluxErrMean = Column(REAL, nullable=True)
    r_psfFluxSkew = Column(REAL, nullable=True)
    r_psfFluxMin = Column(REAL, nullable=True)
    r_psfFluxMax = Column(REAL, nullable=True)
    r_psfFluxMaxSlope = Column(REAL, nullable=True)
    r_psfFluxErrMean = Column(REAL, nullable=True)
    i_psfFluxSkew = Column(REAL, nullable=True)
    i_psfFluxMin = Column(REAL, nullable=True)
    i_psfFluxMax = Column(REAL, nullable=True)
    i_psfFluxMaxSlope = Column(REAL, nullable=True)
    i_psfFluxErrMean = Column(REAL, nullable=True)
    z_psfFluxSkew = Column(REAL, nullable=True)
    z_psfFluxMin = Column(REAL, nullable=True)
    z_psfFluxMax = Column(REAL, nullable=True)
    z_psfFluxMaxSlope = Column(REAL, nullable=True)
    z_psfFluxErrMean = Column(REAL, nullable=True)
    y_psfFluxSkew = Column(REAL, nullable=True)
    y_psfFluxMin = Column(REAL, nullable=True)
    y_psfFluxMax = Column(REAL, nullable=True)
    y_psfFluxMaxSlope = Column(REAL, nullable=True)
    y_psfFluxErrMean = Column(REAL, nullable=True)
    firstDiaSourceMjdTai = Column(DOUBLE_PRECISION, nullable=True)
    lastDiaSourceMjdTai = Column(DOUBLE_PRECISION, nullable=True)
    nDiaSources = Column(Integer, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstdiaobject_oid"),)


class Detection(Base):
    __tablename__ = "detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger)  # int2, nulleable because  original schema

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", "sid", name="pk_detection_oid_measurementid_sid"
        ),
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
        Index("ix_ztfdetection_oid", "oid", postgresql_using="hash"),
    )

# Ommiting lsst ss source for now, this is just dia Source


class LsstDetection(Base):
    __tablename__ = "lsst_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    parentDiaSourceId = Column(BigInteger)


    # diaSourceId = Column(BigInteger, nullable=False) this is oid
    visit = Column(BigInteger, nullable=False)
    detector = Column(Integer, nullable=False)
    #diaObjectId = Column(BigInteger) # From these two we select measurement_id 
    #ssObjectId = Column(BigInteger)
    #ra = Column(DOUBLE_PRECISION, nullable=False) goes into detection
    raErr = Column(REAL)
    #dec = Column(DOUBLE_PRECISION, nullable=False) goes into detection
    decErr = Column(REAL)
    ra_dec_Cov = Column(REAL)
    x = Column(REAL, nullable=False)
    xErr = Column(REAL)
    y = Column(REAL, nullable=False)
    yErr = Column(REAL)
    centroid_flag = Column(Boolean)
    apFlux = Column(REAL)
    apFluxErr = Column(REAL)
    apFlux_flag = Column(Boolean)
    apFlux_flag_apertureTruncated = Column(Boolean)
    isNegative = Column(Boolean)
    snr = Column(REAL)
    psfFlux = Column(REAL)
    psfFluxErr = Column(REAL)
    psfLnL = Column(REAL)
    psfChi2 = Column(REAL)
    psfNdata = Column(Integer)
    psfFlux_flag = Column(Boolean)
    psfFlux_flag_edge = Column(Boolean)
    psfFlux_flag_noGoodPixels = Column(Boolean)
    trailFlux = Column(REAL)
    trailFluxErr = Column(REAL)
    trailRa = Column(DOUBLE_PRECISION)
    trailRaErr = Column(REAL)
    trailDec = Column(DOUBLE_PRECISION)
    trailDecErr = Column(REAL)
    trailLength = Column(REAL)
    trailLengthErr = Column(REAL)
    trailAngle = Column(REAL)
    trailAngleErr = Column(REAL)
    trailChi2 = Column(REAL)
    trailNdata = Column(Integer)
    trail_flag_edge = Column(Boolean)
    dipoleMeanFlux = Column(REAL)
    dipoleMeanFluxErr = Column(REAL)
    dipoleFluxDiff = Column(REAL)
    dipoleFluxDiffErr = Column(REAL)
    dipoleLength = Column(REAL)
    dipoleAngle = Column(REAL)
    dipoleChi2 = Column(REAL)
    dipoleNdata = Column(Integer)
    scienceFlux = Column(REAL)
    scienceFluxErr = Column(REAL)
    forced_PsfFlux_flag = Column(Boolean)
    forced_PsfFlux_flag_edge = Column(Boolean)
    forced_PsfFlux_flag_noGoodPixels = Column(Boolean)
    templateFlux = Column(REAL)
    templateFluxErr = Column(REAL)
    ixx = Column(REAL)
    iyy = Column(REAL)
    ixy = Column(REAL)
    ixxPSF = Column(REAL)
    iyyPSF = Column(REAL)
    ixyPSF = Column(REAL)
    shape_flag = Column(Boolean)
    shape_flag_no_pixels = Column(Boolean)
    shape_flag_not_contained = Column(Boolean)
    shape_flag_parent_source = Column(Boolean)
    extendedness = Column(REAL)
    reliability = Column(REAL)
    # band = Column(SmallInteger) Goes into detection
    isDipole = Column(Boolean)
    dipoleFitAttempted = Column(Boolean)
    time_processed = Column(DateTime, nullable=False)
    bboxSize = Column(BigInteger)
    pixelFlags = Column(Boolean)
    pixelFlags_bad = Column(Boolean)
    pixelFlags_cr = Column(Boolean)
    pixelFlags_crCenter = Column(Boolean)
    pixelFlags_edge = Column(Boolean)
    pixelFlags_nodata = Column(Boolean)
    pixelFlags_nodataCenter = Column(Boolean)
    pixelFlags_interpolated = Column(Boolean)
    pixelFlags_interpolatedCenter = Column(Boolean)
    pixelFlags_offimage = Column(Boolean)
    pixelFlags_saturated = Column(Boolean)
    pixelFlags_saturatedCenter = Column(Boolean)
    pixelFlags_suspect = Column(Boolean)
    pixelFlags_suspectCenter = Column(Boolean)
    pixelFlags_streak = Column(Boolean)
    pixelFlags_streakCenter = Column(Boolean)
    pixelFlags_injected = Column(Boolean)
    pixelFlags_injectedCenter = Column(Boolean)
    pixelFlags_injected_template = Column(Boolean)
    pixelFlags_injected_templateCenter = Column(Boolean)
    glint_trail = Column(Boolean)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstdetection_oid_measurementid"
        ),
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
    band = Column(SmallInteger)  # int2, nulleable original schema

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", "sid", name="pk_forcedphotometry_oid_measurementid_sid"
        ),
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
        Index("ix_ztf_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class LsstForcedPhotometry(Base):
    __tablename__ = "lsst_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,

    visit = Column(BigInteger, nullable=False)
    detector = Column(Integer, nullable=False)
    psfFlux = Column(REAL)
    psfFluxErr = Column(REAL)
    scienceFlux = Column(REAL)
    scienceFluxErr = Column(REAL)
    time_processed = Column(DateTime, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstforcedphotometry_oid_measurementid"
        ),
        Index("ix_lsst_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfNonDetection(Base):
    __tablename__ = "ztf_non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    band = Column(SmallInteger, nullable=False)  # int2,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float4,

    __table_args__ = (
        PrimaryKeyConstraint("oid", "mjd", name="pk_oid_mjd"),
        Index("ix_non_detection_oid", "oid", postgresql_using="hash"),
    )

# Ommiting lsst non detection for now



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
        PrimaryKeyConstraint("oid", "sid", "band", name="pk_magstat_oid_sid_band"),
    )

class classifier(Base):
    __tablename__ = "classifier"
    classifier_id = Column(Integer, primary_key=True)
    classifier_name = Column(VARCHAR)
    classifier_version = Column(SmallInteger)

class Taxonomy(Base):
    __tablename__ = "taxonomy"
    class_id = Column(Integer, primary_key=True)
    class_name = Column(VARCHAR)
    order = Column(Integer)
    classifier_id = Column(SmallInteger)
