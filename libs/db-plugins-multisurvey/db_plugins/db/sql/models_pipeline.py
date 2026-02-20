from sqlalchemy import (
    VARCHAR,
    BigInteger,
    Boolean,
    Column,
    Connection,
    Date,
    Index,
    Integer,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, REAL
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    __n_partitions__: int | None = None

    @classmethod
    def __partition_on__(cls, _partition_idx: int) -> str:
        return ""

    @classmethod
    def __create_partitions__(cls, conn: Connection, schema: str = None):
        schema_prefix = f"{schema}." if schema else ""
        for i in range(cls.__n_partitions__):
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {schema_prefix}{cls.__tablename__}_part_{i} 
                PARTITION OF {schema_prefix}{cls.__tablename__} 
                {cls.__partition_on__(i)}
            """)
            )
        conn.commit()


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

    created_date = Column(Date, server_default=func.now())
    updated_date = Column(Date, onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("oid", "sid", name="pk_object_oid_sid"),
        Index("ix_object_n_det", "n_det", postgresql_using="btree"),
        Index("ix_object_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("ix_object_lastmjd", "lastmjd", postgresql_using="btree"),
        Index("ix_object_meanra", "meanra", postgresql_using="btree"),
        Index("ix_object_meandec", "meandec", postgresql_using="btree"),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 8

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"


class ZtfObject(Base):
    __tablename__ = "ztf_object"

    oid = Column(BigInteger, nullable=False)
    g_r_max = Column(REAL)
    g_r_max_corr = Column(REAL)
    g_r_mean = Column(REAL)
    g_r_mean_corr = Column(REAL)
    corrected = Column(Boolean, nullable=False, default=False)
    stellar = Column(Boolean, nullable=True, default=None)
    reference_change = Column(Boolean)  # bool
    diffpos = Column(Boolean)  # bool
    ndethist = Column(Integer)  # int4,
    ncovhist = Column(Integer)  # int4,
    mjdstarthist = Column(DOUBLE_PRECISION)
    mjdendhist = Column(DOUBLE_PRECISION)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_ztfobject_oid"),)


# Ommiting lsst ss object for now


class LsstDiaObject(Base):
    __tablename__ = "lsst_dia_object"

    oid = Column(BigInteger, nullable=False)

    # all fields 8.0 draft
    # diaObjectId = Column(BigInteger, nullable=False) => This is oid
    validityStartMjdTai = Column(DOUBLE_PRECISION, nullable=False)  # v9 field
    ra = Column(DOUBLE_PRECISION, nullable=False)
    raErr = Column(REAL, nullable=True)
    dec = Column(DOUBLE_PRECISION, nullable=False)
    decErr = Column(REAL, nullable=True)
    ra_dec_Cov = Column(REAL, nullable=True)
    u_psfFluxMean = Column(REAL, nullable=True)
    u_psfFluxMeanErr = Column(REAL, nullable=True)
    u_psfFluxSigma = Column(REAL, nullable=True)
    u_psfFluxNdata = Column(Integer, nullable=True)
    u_fpFluxMean = Column(REAL, nullable=True)
    u_fpFluxMeanErr = Column(REAL, nullable=True)
    g_psfFluxMean = Column(REAL, nullable=True)
    g_psfFluxMeanErr = Column(REAL, nullable=True)
    g_psfFluxSigma = Column(REAL, nullable=True)
    g_psfFluxNdata = Column(Integer, nullable=True)
    g_fpFluxMean = Column(REAL, nullable=True)
    g_fpFluxMeanErr = Column(REAL, nullable=True)
    r_psfFluxMean = Column(REAL, nullable=True)
    r_psfFluxMeanErr = Column(REAL, nullable=True)
    r_psfFluxSigma = Column(REAL, nullable=True)
    r_psfFluxNdata = Column(Integer, nullable=True)
    r_fpFluxMean = Column(REAL, nullable=True)
    r_fpFluxMeanErr = Column(REAL, nullable=True)
    i_psfFluxMean = Column(REAL, nullable=True)
    i_psfFluxMeanErr = Column(REAL, nullable=True)
    i_psfFluxSigma = Column(REAL, nullable=True)
    i_psfFluxNdata = Column(Integer, nullable=True)
    i_fpFluxMean = Column(REAL, nullable=True)
    i_fpFluxMeanErr = Column(REAL, nullable=True)
    z_psfFluxMean = Column(REAL, nullable=True)
    z_psfFluxMeanErr = Column(REAL, nullable=True)
    z_psfFluxSigma = Column(REAL, nullable=True)
    z_psfFluxNdata = Column(Integer, nullable=True)
    z_fpFluxMean = Column(REAL, nullable=True)
    z_fpFluxMeanErr = Column(REAL, nullable=True)
    y_psfFluxMean = Column(REAL, nullable=True)
    y_psfFluxMeanErr = Column(REAL, nullable=True)
    y_psfFluxSigma = Column(REAL, nullable=True)
    y_psfFluxNdata = Column(Integer, nullable=True)
    y_fpFluxMean = Column(REAL, nullable=True)
    y_fpFluxMeanErr = Column(REAL, nullable=True)
    u_scienceFluxMean = Column(REAL, nullable=True)
    u_scienceFluxMeanErr = Column(REAL, nullable=True)
    g_scienceFluxMean = Column(REAL, nullable=True)
    g_scienceFluxMeanErr = Column(REAL, nullable=True)
    r_scienceFluxMean = Column(REAL, nullable=True)
    r_scienceFluxMeanErr = Column(REAL, nullable=True)
    i_scienceFluxMean = Column(REAL, nullable=True)
    i_scienceFluxMeanErr = Column(REAL, nullable=True)
    z_scienceFluxMean = Column(REAL, nullable=True)
    z_scienceFluxMeanErr = Column(REAL, nullable=True)
    y_scienceFluxMean = Column(REAL, nullable=True)
    y_scienceFluxMeanErr = Column(REAL, nullable=True)
    u_psfFluxMin = Column(REAL, nullable=True)
    u_psfFluxMax = Column(REAL, nullable=True)
    u_psfFluxMaxSlope = Column(REAL, nullable=True)
    u_psfFluxErrMean = Column(REAL, nullable=True)
    g_psfFluxMin = Column(REAL, nullable=True)
    g_psfFluxMax = Column(REAL, nullable=True)
    g_psfFluxMaxSlope = Column(REAL, nullable=True)
    g_psfFluxErrMean = Column(REAL, nullable=True)
    r_psfFluxMin = Column(REAL, nullable=True)
    r_psfFluxMax = Column(REAL, nullable=True)
    r_psfFluxMaxSlope = Column(REAL, nullable=True)
    r_psfFluxErrMean = Column(REAL, nullable=True)
    i_psfFluxMin = Column(REAL, nullable=True)
    i_psfFluxMax = Column(REAL, nullable=True)
    i_psfFluxMaxSlope = Column(REAL, nullable=True)
    i_psfFluxErrMean = Column(REAL, nullable=True)
    z_psfFluxMin = Column(REAL, nullable=True)
    z_psfFluxMax = Column(REAL, nullable=True)
    z_psfFluxMaxSlope = Column(REAL, nullable=True)
    z_psfFluxErrMean = Column(REAL, nullable=True)
    y_psfFluxMin = Column(REAL, nullable=True)
    y_psfFluxMax = Column(REAL, nullable=True)
    y_psfFluxMaxSlope = Column(REAL, nullable=True)
    y_psfFluxErrMean = Column(REAL, nullable=True)
    firstDiaSourceMjdTai = Column(DOUBLE_PRECISION, nullable=True)
    lastDiaSourceMjdTai = Column(DOUBLE_PRECISION, nullable=True)
    nDiaSources = Column(Integer, nullable=False)
    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("oid", name="pk_lsstdiaobject_oid"),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 8

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"


class LsstSsObject(Base):
    __tablename__ = "lsst_ss_object"

    oid = Column(BigInteger, primary_key=True, nullable=False)

    designation = Column(String, nullable=True)  #!!!!!!!
    nObs = Column(Integer, nullable=False)
    arc = Column(REAL, nullable=False)

    firstObservationMjdTai = Column(DOUBLE_PRECISION, nullable=True)

    MOIDEarth = Column(REAL, nullable=True)
    MOIDEarthDeltaV = Column(REAL, nullable=True)
    MOIDEarthEclipticLongitude = Column(REAL, nullable=True)
    MOIDEarthTrueAnomaly = Column(REAL, nullable=True)
    MOIDEarthTrueAnomalyObject = Column(REAL, nullable=True)

    tisserand_J = Column(REAL, nullable=True)

    extendednessMax = Column(REAL, nullable=True)
    extendednessMedian = Column(REAL, nullable=True)
    extendednessMin = Column(REAL, nullable=True)

    u_nObs = Column(Integer, nullable=True)
    u_H = Column(REAL, nullable=True)
    u_HErr = Column(REAL, nullable=True)
    u_G12 = Column(REAL, nullable=True)
    u_G12Err = Column(REAL, nullable=True)
    u_H_u_G12_Cov = Column(REAL, nullable=True)
    u_nObsUsed = Column(Integer, nullable=True)
    u_Chi2 = Column(REAL, nullable=True)
    u_phaseAngleMin = Column(REAL, nullable=True)
    u_phaseAngleMax = Column(REAL, nullable=True)
    u_slope_fit_failed = Column(Boolean, nullable=True)

    g_nObs = Column(Integer, nullable=True)
    g_H = Column(REAL, nullable=True)
    g_HErr = Column(REAL, nullable=True)
    g_G12 = Column(REAL, nullable=True)
    g_G12Err = Column(REAL, nullable=True)
    g_H_g_G12_Cov = Column(REAL, nullable=True)
    g_nObsUsed = Column(Integer, nullable=True)
    g_Chi2 = Column(REAL, nullable=True)
    g_phaseAngleMin = Column(REAL, nullable=True)
    g_phaseAngleMax = Column(REAL, nullable=True)
    g_slope_fit_failed = Column(Boolean, nullable=True)

    r_nObs = Column(Integer, nullable=True)
    r_H = Column(REAL, nullable=True)
    r_HErr = Column(REAL, nullable=True)
    r_G12 = Column(REAL, nullable=True)
    r_G12Err = Column(REAL, nullable=True)
    r_H_r_G12_Cov = Column(REAL, nullable=True)
    r_nObsUsed = Column(Integer, nullable=True)
    r_Chi2 = Column(REAL, nullable=True)
    r_phaseAngleMin = Column(REAL, nullable=True)
    r_phaseAngleMax = Column(REAL, nullable=True)
    r_slope_fit_failed = Column(Boolean, nullable=True)

    i_nObs = Column(Integer, nullable=True)
    i_H = Column(REAL, nullable=True)
    i_HErr = Column(REAL, nullable=True)
    i_G12 = Column(REAL, nullable=True)
    i_G12Err = Column(REAL, nullable=True)
    i_H_i_G12_Cov = Column(REAL, nullable=True)
    i_nObsUsed = Column(Integer, nullable=True)
    i_Chi2 = Column(REAL, nullable=True)
    i_phaseAngleMin = Column(REAL, nullable=True)
    i_phaseAngleMax = Column(REAL, nullable=True)
    i_slope_fit_failed = Column(Boolean, nullable=True)

    z_nObs = Column(Integer, nullable=True)
    z_H = Column(REAL, nullable=True)
    z_HErr = Column(REAL, nullable=True)
    z_G12 = Column(REAL, nullable=True)
    z_G12Err = Column(REAL, nullable=True)
    z_H_z_G12_Cov = Column(REAL, nullable=True)
    z_nObsUsed = Column(Integer, nullable=True)
    z_Chi2 = Column(REAL, nullable=True)
    z_phaseAngleMin = Column(REAL, nullable=True)
    z_phaseAngleMax = Column(REAL, nullable=True)
    z_slope_fit_failed = Column(Boolean, nullable=True)

    y_nObs = Column(Integer, nullable=True)
    y_H = Column(REAL, nullable=True)
    y_HErr = Column(REAL, nullable=True)
    y_G12 = Column(REAL, nullable=True)
    y_G12Err = Column(REAL, nullable=True)
    y_H_y_G12_Cov = Column(REAL, nullable=True)
    y_nObsUsed = Column(Integer, nullable=True)
    y_Chi2 = Column(REAL, nullable=True)
    y_phaseAngleMin = Column(REAL, nullable=True)
    y_phaseAngleMax = Column(REAL, nullable=True)
    y_slope_fit_failed = Column(Boolean, nullable=True)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (PrimaryKeyConstraint("oid", name="pk_lsstssobject_oid"),)


class LsstMpcOrbits(Base):
    __tablename__ = "lsst_mpc_orbits"

    # Primary key
    ssObjectId = Column(BigInteger, primary_key=True)

    designation = Column(String, nullable=False)  #!!!!!!!
    packed_primary_provisional_designation = Column(String, nullable=False)  #!!!!!!!
    unpacked_primary_provisional_designation = Column(String, nullable=False)  #!!!!!!!

    mpc_orb_jsonb = Column(JSONB, nullable=True)  #!!!!!!!

    created_at = Column(Date, nullable=True)
    updated_at = Column(Date, nullable=True)

    orbit_type_int = Column(Integer, nullable=True)
    u_param = Column(Integer, nullable=True)
    nopp = Column(Integer, nullable=True)

    arc_length_total = Column(DOUBLE_PRECISION, nullable=True)
    arc_length_sel = Column(DOUBLE_PRECISION, nullable=True)

    nobs_total = Column(Integer, nullable=True)
    nobs_total_sel = Column(Integer, nullable=True)

    a = Column(DOUBLE_PRECISION, nullable=True)
    q = Column(DOUBLE_PRECISION, nullable=True)
    e = Column(DOUBLE_PRECISION, nullable=True)
    i = Column(DOUBLE_PRECISION, nullable=True)
    node = Column(DOUBLE_PRECISION, nullable=True)
    argperi = Column(DOUBLE_PRECISION, nullable=True)
    peri_time = Column(DOUBLE_PRECISION, nullable=True)

    yarkovsky = Column(DOUBLE_PRECISION, nullable=True)
    srp = Column(DOUBLE_PRECISION, nullable=True)
    a1 = Column(DOUBLE_PRECISION, nullable=True)
    a2 = Column(DOUBLE_PRECISION, nullable=True)
    a3 = Column(DOUBLE_PRECISION, nullable=True)
    dt = Column(DOUBLE_PRECISION, nullable=True)

    mean_anomaly = Column(DOUBLE_PRECISION, nullable=True)
    period = Column(DOUBLE_PRECISION, nullable=True)
    mean_motion = Column(DOUBLE_PRECISION, nullable=True)

    a_unc = Column(DOUBLE_PRECISION, nullable=True)
    q_unc = Column(DOUBLE_PRECISION, nullable=True)
    e_unc = Column(DOUBLE_PRECISION, nullable=True)
    i_unc = Column(DOUBLE_PRECISION, nullable=True)
    node_unc = Column(DOUBLE_PRECISION, nullable=True)
    argperi_unc = Column(DOUBLE_PRECISION, nullable=True)
    peri_time_unc = Column(DOUBLE_PRECISION, nullable=True)
    yarkovsky_unc = Column(DOUBLE_PRECISION, nullable=True)
    srp_unc = Column(DOUBLE_PRECISION, nullable=True)
    a1_unc = Column(DOUBLE_PRECISION, nullable=True)
    a2_unc = Column(DOUBLE_PRECISION, nullable=True)
    a3_unc = Column(DOUBLE_PRECISION, nullable=True)
    dt_unc = Column(DOUBLE_PRECISION, nullable=True)
    mean_anomaly_unc = Column(DOUBLE_PRECISION, nullable=True)
    period_unc = Column(DOUBLE_PRECISION, nullable=True)
    mean_motion_unc = Column(DOUBLE_PRECISION, nullable=True)

    epoch_mjd = Column(DOUBLE_PRECISION, nullable=True)

    h = Column(DOUBLE_PRECISION, nullable=True)
    g = Column(DOUBLE_PRECISION, nullable=True)

    not_normalized_rms = Column(DOUBLE_PRECISION, nullable=True)
    normalized_rms = Column(DOUBLE_PRECISION, nullable=True)

    earth_moid = Column(DOUBLE_PRECISION, nullable=True)

    fitting_datetime = Column(Date, nullable=True)

    created_date = Column(Date, server_default=func.now())
    updated_date = Column(Date, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ssObjectId", name="pk_lsstmpcorbits_ss_object_id"),
    )


class Detection(Base):
    __tablename__ = "detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    ra = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    dec = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    band = Column(SmallInteger)  # int2, nulleable because  original schema

    created_date = Column(Date, server_default=func.now())

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
    magpsf_corr = Column(REAL)  # float4,
    sigmapsf_corr = Column(REAL)  # float4,
    sigmapsf_corr_ext = Column(REAL)  # float4,
    corrected = Column(Boolean)  # bool,
    dubious = Column(Boolean)  # bool,
    parent_candid = Column(BigInteger)  # int8,
    has_stamp = Column(Boolean)  # bool,
    created_date = Column(Date, server_default=func.now())

    created_date = Column(Date, server_default=func.now())

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

    visit = Column(BigInteger, nullable=False)
    detector = Column(Integer, nullable=False)
    diaObjectId = Column(BigInteger)  # From these two we select measurement_id
    ssObjectId = Column(BigInteger)
    # ra = Column(DOUBLE_PRECISION, nullable=False) goes into detection
    raErr = Column(REAL)
    # dec = Column(DOUBLE_PRECISION, nullable=False) goes into detection
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
    timeProcessedMjdTai = Column(DOUBLE_PRECISION, nullable=False)
    timeWithdrawnMjdTai = Column(DOUBLE_PRECISION, nullable=True)
    bboxSize = Column(Integer)
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
    has_stamp = Column(Boolean)  # bool,

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstdetection_oid_measurementid"
        ),
        Index("ix_lsstdetection_oid", "oid", postgresql_using="hash"),
    )


class LsstSsDetection(Base):
    __tablename__ = "lsst_ss_detection"

    measurement_id = Column(BigInteger, nullable=False)
    ssObjectId = Column(BigInteger, nullable=False)

    designation = Column(String)  #!!!!!!!
    eclLambda = Column(DOUBLE_PRECISION, nullable=False)
    eclBeta = Column(DOUBLE_PRECISION, nullable=False)
    galLon = Column(DOUBLE_PRECISION, nullable=False)
    galLat = Column(DOUBLE_PRECISION, nullable=False)
    elongation = Column(REAL)
    phaseAngle = Column(REAL)
    topoRange = Column(REAL)
    topoRangeRate = Column(REAL)
    helioRange = Column(REAL)
    helioRangeRate = Column(REAL)
    ephRa = Column(DOUBLE_PRECISION)
    ephDec = Column(DOUBLE_PRECISION)
    ephVmag = Column(REAL)
    ephRate = Column(REAL)
    ephRateRa = Column(REAL)
    ephRateDec = Column(REAL)
    ephOffset = Column(REAL)
    ephOffsetRa = Column(DOUBLE_PRECISION)
    ephOffsetDec = Column(DOUBLE_PRECISION)
    ephOffsetAlongTrack = Column(REAL)
    ephOffsetCrossTrack = Column(REAL)
    helio_x = Column(REAL)
    helio_y = Column(REAL)
    helio_z = Column(REAL)
    helio_vx = Column(REAL)
    helio_vy = Column(REAL)
    helio_vz = Column(REAL)
    helio_vtot = Column(REAL)
    topo_x = Column(REAL)
    topo_y = Column(REAL)
    topo_z = Column(REAL)
    topo_vx = Column(REAL)
    topo_vy = Column(REAL)
    topo_vz = Column(REAL)
    topo_vtot = Column(REAL)
    diaDistanceRank = Column(Integer)

    created_date = Column(Date, server_default=func.now())

    #! CHECK
    __table_args__ = (
        PrimaryKeyConstraint(
            "ssObjectId",
            "measurement_id",
            name="pk_lsstssdetection_ssobjectid_measurementid",
        ),
        Index("ix_lsstssdetection_ssobjectid", "ssObjectId", postgresql_using="hash"),
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

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid",
            "measurement_id",
            "sid",
            name="pk_forcedphotometry_oid_measurementid_sid",
        ),
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfForcedPhotometry(Base):
    __tablename__ = "ztf_forced_photometry"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)  # int2,
    measurement_id = Column(BigInteger, nullable=False)  # int8,
    pid = Column(BigInteger)  # int8
    mag = Column(REAL, nullable=False)  # float4,
    e_mag = Column(REAL, nullable=False)  # float4,
    mag_corr = Column(REAL)  # float4,
    e_mag_corr = Column(REAL)  # float4,
    e_mag_corr_ext = Column(REAL)  # float4,
    isdiffpos = Column(Integer, nullable=False)  # int4 NOT NULL,
    corrected = Column(Boolean, nullable=False)  # bool NOT NULL,
    dubious = Column(Boolean, nullable=False)  # bool NOT NULL,
    parent_candid = Column(BigInteger)  # varchar,
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

    created_date = Column(Date, server_default=func.now())

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
    timeProcessedMjdTai = Column(DOUBLE_PRECISION, nullable=False)
    timeWithdrawnMjdTai = Column(DOUBLE_PRECISION, nullable=True)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "measurement_id", name="pk_lsstforcedphotometry_oid_measurementid"
        ),
        Index("ix_lsst_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class ZtfNonDetection(Base):
    __tablename__ = "ztf_non_detection"

    oid = Column(BigInteger, nullable=False)  # int8,
    sid = Column(SmallInteger, nullable=False)
    band = Column(SmallInteger, nullable=False)  # int2,
    mjd = Column(DOUBLE_PRECISION, nullable=False)  # float8,
    diffmaglim = Column(REAL, nullable=False)  # float4,

    created_date = Column(Date, server_default=func.now())

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

    created_date = Column(Date, server_default=func.now())

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
    nmtchps = Column(Integer)

    created_date = Column(Date, server_default=func.now())

    created_date = Column(Date, server_default=func.now())

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

    created_date = Column(Date, server_default=func.now())

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
    scorr = Column(DOUBLE_PRECISION)
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

    created_date = Column(Date, server_default=func.now())

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

    created_date = Column(Date, server_default=func.now())

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
    ndubious = Column(Integer)  # int4
    dmdt_first = Column(REAL)  # float4
    dm_first = Column(REAL)  # float4
    sigmadm_first = Column(REAL)  # float4
    dt_first = Column(REAL)  # float4
    magmean = Column(REAL)  # float4
    magmedian = Column(REAL)  # float4
    magmax = Column(REAL)  # float4
    magmin = Column(REAL)  # float4
    magsigma = Column(REAL)  # float4
    maglast = Column(REAL)  # float4
    magfirst = Column(REAL)  # float4
    magmean_corr = Column(REAL)  # float4
    magmedian_corr = Column(REAL)  # float4
    magmax_corr = Column(REAL)  # float4
    magmin_corr = Column(REAL)  # float4
    magsigma_corr = Column(REAL)  # float4
    maglast_corr = Column(REAL)  # float4
    magfirst_corr = Column(REAL)  # float4
    step_id_corr = Column(VARCHAR)  # varchar
    n_det = Column(Integer)  # int4
    firstmjd = Column(DOUBLE_PRECISION)  # float8
    lastmjd = Column(DOUBLE_PRECISION)  # float8
    saturation_rate = Column(REAL)  # float4

    updated_date = Column(Date, onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("oid", "sid", "band", name="pk_magstat_oid_sid_band"),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 16

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"


class Classifier(Base):
    __tablename__ = "classifier"

    classifier_id = Column(Integer, nullable=False)
    classifier_name = Column(VARCHAR, nullable=False)
    classifier_version = Column(VARCHAR, nullable=False)
    tid = Column(SmallInteger, nullable=False)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("classifier_id", name="pk_classifier_classifierid"),
    )


class Taxonomy(Base):
    __tablename__ = "taxonomy"

    class_id = Column(Integer, nullable=False)
    class_name = Column(VARCHAR, nullable=False)
    order = Column(Integer, nullable=False)
    classifier_id = Column(SmallInteger, nullable=False)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (PrimaryKeyConstraint("class_id", name="pk_taxonomy_classid"),)


class Probability(Base):
    __tablename__ = "probability"

    oid = Column(BigInteger, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    classifier_id = Column(SmallInteger, nullable=False)
    classifier_version = Column(SmallInteger, nullable=False)
    class_id = Column(SmallInteger, nullable=False)
    probability = Column(REAL, nullable=False)
    ranking = Column(SmallInteger)
    lastmjd = Column(DOUBLE_PRECISION, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid",
            "sid",
            "classifier_id",
            "class_id",
            name="pk_probability_oid_classifierid_classid",
        ),
        Index("ix_probability_oid", "oid", postgresql_using="hash"),
        Index("ix_probability_probability", "probability", postgresql_using="btree"),
        Index("ix_probability_ranking", "ranking", postgresql_using="btree"),
        Index(
            "ix_classification_rank1",
            "ranking",
            postgresql_where=ranking == 1,
            postgresql_using="btree",
        ),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 16

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"


class Feature(Base):
    __tablename__ = "feature"

    oid = Column(BigInteger, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    feature_id = Column(SmallInteger, nullable=False)
    band = Column(SmallInteger, nullable=False)
    version = Column(SmallInteger, nullable=False)
    value = Column(DOUBLE_PRECISION)
    updated_date = Column(Date, onupdate=func.now())

    # Not set as pk on postgres. Necessary to define a pk-less sqlalchemy table
    __mapper_args__ = {"primary_key": ["oid", "sid", "feature_id", "band"]}

    __table_args__ = (
        Index("idx_feature_oid", "oid", postgresql_using="btree"),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 32

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"


class FeatureNameLut(Base):
    __tablename__ = "feature_name_lut"

    feature_id = Column(SmallInteger, nullable=False, autoincrement=True)
    feature_name = Column(VARCHAR, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    tid = Column(SmallInteger, nullable=False)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("feature_id", name="pk_feature_name_lut_featureid"),
    )


class FeatureVersionLut(Base):
    __tablename__ = "feature_version_lut"

    version_id = Column(SmallInteger, nullable=False, autoincrement=True)
    version_name = Column(VARCHAR, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    tid = Column(SmallInteger, nullable=False)

    
    __table_args__ = (
        PrimaryKeyConstraint("version_id", name="pk_feature_version_lut_versionid"),
    )


class SidLut(Base):
    __tablename__ = "sid_lut"

    sid = Column(SmallInteger, nullable=False, autoincrement=False)
    tid = Column(SmallInteger)
    survey_name = Column(VARCHAR, nullable=False)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (PrimaryKeyConstraint("sid", name="pk_sid_lut_sid"),)


class Band(Base):
    __tablename__ = "band"

    sid = Column(SmallInteger, nullable=False, autoincrement=False)
    tid = Column(SmallInteger, nullable=False, autoincrement=False)
    band = Column(SmallInteger, nullable=False, autoincrement=False)
    band_name = Column(VARCHAR, nullable=False)
    order = Column(Integer)

    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("sid", "tid", "band", name="pk_band_sid_tid_band"),
    )


class Xmatch(Base):
    __tablename__ = "xmatch"

    oid = Column(BigInteger, nullable=False)
    sid = Column(SmallInteger, nullable=False)
    catid = Column(SmallInteger, nullable=False)

    dist = Column(REAL, nullable=False)
    oid_catalog = Column(VARCHAR, nullable=False)

    created_date = Column(Date, server_default=func.now())
    updated_date = Column(Date, onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid",
            "sid",
            "catid",
            name="pk_xmatch_oid_sid_catid",
        ),
    )


class CatalogIdLut(Base):
    __tablename__ = "catalog_id_lut"
    catid = Column(SmallInteger, autoincrement=True)
    catalog_name = Column(VARCHAR)
    created_date = Column(Date, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("catid", name="pk_catalog_id_lut_catalog_name"),
    )


class Allwise(Base):
    __tablename__ = "allwise"
    oid_catalog = Column(String, primary_key=True)
    ra = Column(DOUBLE_PRECISION, nullable=False)
    dec = Column(DOUBLE_PRECISION, nullable=False)
    w1mpro = Column(DOUBLE_PRECISION)
    w2mpro = Column(DOUBLE_PRECISION)
    w3mpro = Column(DOUBLE_PRECISION)
    w4mpro = Column(DOUBLE_PRECISION)
    w1sigmpro = Column(DOUBLE_PRECISION)
    w2sigmpro = Column(DOUBLE_PRECISION)
    w3sigmpro = Column(DOUBLE_PRECISION)
    w4sigmpro = Column(DOUBLE_PRECISION)
    j_m_2mass = Column(DOUBLE_PRECISION)
    h_m_2mass = Column(DOUBLE_PRECISION)
    k_m_2mass = Column(DOUBLE_PRECISION)
    j_msig_2mass = Column(DOUBLE_PRECISION)
    h_msig_2mass = Column(DOUBLE_PRECISION)
    k_msig_2mass = Column(DOUBLE_PRECISION)

    __table_args__ = (
        PrimaryKeyConstraint("oid_catalog", name="pk_allwise_oid_catalog"),
    )
