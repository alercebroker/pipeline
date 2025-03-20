from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    ForeignKey,
    Float,
    Boolean,
    ARRAY,
    Index,
    UniqueConstraint,
    DateTime,
    JSON,
    ForeignKeyConstraint,
)

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Commons:
    def __getitem__(self, field):
        return self.__dict__[field]


class Step(Base):
    __tablename__ = "step"

    step_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    comments = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)


class Object(Base):
    __tablename__ = "object"

    oid = Column(String, primary_key=True)
    ndethist = Column(Integer, nullable=False)
    ncovhist = Column(Integer, nullable=False)
    mjdstarthist = Column(Float(precision=53))
    mjdendhist = Column(Float(precision=53))
    corrected = Column(Boolean, nullable=False, default=False)
    stellar = Column(Boolean, nullable=False, default=False)
    ndet = Column(Integer, nullable=False, default=1)
    g_r_max = Column(Float)
    g_r_max_corr = Column(Float)
    g_r_mean = Column(Float)
    g_r_mean_corr = Column(Float)
    meanra = Column(Float(precision=53), nullable=False)
    meandec = Column(Float(precision=53), nullable=False)
    sigmara = Column(Float(precision=53))
    sigmadec = Column(Float(precision=53))
    deltajd = Column(Float(precision=53), nullable=False)
    firstmjd = Column(Float(precision=53), nullable=False)
    lastmjd = Column(Float(precision=53), nullable=False)
    step_id_corr = Column(String, nullable=False)
    diffpos = Column(Boolean)
    reference_change = Column(Boolean)

    __table_args__ = (
        Index("ix_object_ndet", "ndet", postgresql_using="btree"),
        Index("ix_object_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("ix_object_g_r_max", "g_r_max", postgresql_using="btree"),
        Index("ix_object_g_r_mean_corr", "g_r_mean_corr", postgresql_using="btree"),
        Index("ix_object_meanra", "meanra", postgresql_using="btree"),
        Index("ix_object_meandec", "meandec", postgresql_using="btree"),
    )

    def __repr__(self):
        return "<Object(oid='%s')>" % (self.oid)


class Taxonomy(Base):
    __tablename__ = "taxonomy"
    classifier_name = Column(String, primary_key=True)
    classifier_version = Column(String, primary_key=True)
    classes = Column(ARRAY(String), nullable=False)


class Probability(Base):
    __tablename__ = "probability"
    oid = Column(String, ForeignKey(Object.oid), primary_key=True)
    class_name = Column(String, primary_key=True)
    classifier_name = Column(String, primary_key=True)
    classifier_version = Column(String, primary_key=True)
    probability = Column(Float, nullable=False)
    ranking = Column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_probabilities_oid", "oid", postgresql_using="hash"),
        Index("ix_probabilities_probability", "probability", postgresql_using="btree"),
        Index("ix_probabilities_ranking", "ranking", postgresql_using="btree"),
        Index(
            "ix_classification_rank1",
            "ranking",
            postgresql_where=ranking == 1,
            postgresql_using="btree",
        ),
    )


class Score(Base):
    __tablename__ = "score"
    oid = Column(String, ForeignKey(Object.oid), primary_key=True)
    detector_name = Column(String, primary_key=True)
    detector_version = Column(String, primary_key=True)
    category_name = Column(String, primary_key=True)
    score = Column(Float, nullable=False)
    __table_args__ = (
        Index("ix_scores_oid", "oid", postgresql_using="hash"),
        Index("ix_scores_score", "score", postgresql_using="btree"),
    )


class ScoreDistribution(Base):
    __tablename__ = "score_distribution"
    detector_name = Column(String, primary_key=True)
    distribution_version = Column(String, primary_key=True)
    creation_date = Column(DateTime)
    category_name = Column(String, primary_key=True)
    distribution_name = Column(String, primary_key=True)
    distribution_value = Column(Float, nullable=False)
    __table_args__ = (
        Index(
            "ix_scoredistribution_distribution_name",
            "distribution_name",
            postgresql_using="hash",
        ),
        Index(
            "ix_scoredistribution_category_name",
            "category_name",
            postgresql_using="hash",
        ),
        Index(
            "ix_scoredistribution_detector_name",
            "category_name",
            postgresql_using="hash",
        ),
    )


class FeatureVersion(Base):
    __tablename__ = "feature_version"
    version = Column(String, primary_key=True)
    step_id_feature = Column(String, ForeignKey(Step.step_id))
    step_id_preprocess = Column(String, ForeignKey(Step.step_id))

    # __table_args__ = (ForeignKeyConstraint([step_id_feature, step_id_preprocess],
    #                                        [Step.step_id, Step.step_id]),
    #                   {})


class Feature(Base):
    __tablename__ = "feature"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    name = Column(String, primary_key=True, nullable=False)
    value = Column(Float(precision=53))
    fid = Column(Integer, primary_key=True)
    version = Column(String, primary_key=True, nullable=False)

    __table_args__ = (Index("ix_feature_oid_2", "oid", postgresql_using="hash"),)


class Xmatch(Base):
    __tablename__ = "xmatch"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    catid = Column(String, primary_key=True)
    oid_catalog = Column(String, nullable=False)
    dist = Column(Float(precision=53), nullable=False)
    class_catalog = Column(String)
    period = Column(Float(precision=53))


class Allwise(Base):
    __tablename__ = "allwise"

    oid_catalog = Column(String, primary_key=True)
    ra = Column(Float(precision=53), nullable=False)
    dec = Column(Float(precision=53), nullable=False)
    w1mpro = Column(Float(precision=53))
    w2mpro = Column(Float(precision=53))
    w3mpro = Column(Float(precision=53))
    w4mpro = Column(Float(precision=53))
    w1sigmpro = Column(Float(precision=53))
    w2sigmpro = Column(Float(precision=53))
    w3sigmpro = Column(Float(precision=53))
    w4sigmpro = Column(Float(precision=53))
    j_m_2mass = Column(Float(precision=53))
    h_m_2mass = Column(Float(precision=53))
    k_m_2mass = Column(Float(precision=53))
    j_msig_2mass = Column(Float(precision=53))
    h_msig_2mass = Column(Float(precision=53))
    k_msig_2mass = Column(Float(precision=53))

    __table_args__ = (
        Index("ix_allwise_dec", "dec", postgresql_using="btree"),
        Index("ix_allwise_ra", "ra", postgresql_using="btree"),
    )


class MagStats(Base):
    __tablename__ = "magstat"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    fid = Column(Integer, primary_key=True)
    stellar = Column(Boolean, nullable=False)
    corrected = Column(Boolean, nullable=False)
    ndet = Column(Integer, nullable=False)
    ndubious = Column(Integer, nullable=False)
    dmdt_first = Column(Float)
    dm_first = Column(Float)
    sigmadm_first = Column(Float)
    dt_first = Column(Float)
    magmean = Column(Float)
    magmedian = Column(Float)
    magmax = Column(Float)
    magmin = Column(Float)
    magsigma = Column(Float)
    maglast = Column(Float)
    magfirst = Column(Float)
    magmean_corr = Column(Float)
    magmedian_corr = Column(Float)
    magmax_corr = Column(Float)
    magmin_corr = Column(Float)
    magsigma_corr = Column(Float)
    maglast_corr = Column(Float)
    magfirst_corr = Column(Float)
    firstmjd = Column(Float(precision=53))
    lastmjd = Column(Float(precision=53))
    step_id_corr = Column(String, nullable=False)
    saturation_rate = Column(Float(precision=53))

    __table_args__ = (
        Index("ix_magstats_dmdt_first", "dmdt_first", postgresql_using="btree"),
        Index("ix_magstats_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("ix_magstats_lastmjd", "lastmjd", postgresql_using="btree"),
        Index("ix_magstats_magmean", "magmean", postgresql_using="btree"),
        Index("ix_magstats_magmin", "magmin", postgresql_using="btree"),
        Index("ix_magstats_magfirst", "magfirst", postgresql_using="btree"),
        Index("ix_magstats_ndet", "ndet", postgresql_using="btree"),
        Index("ix_magstats_maglast", "maglast", postgresql_using="btree"),
        Index("ix_magstats_oid", "oid", postgresql_using="hash"),
    )


class NonDetection(Base, Commons):
    __tablename__ = "non_detection"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    fid = Column(Integer, primary_key=True)
    mjd = Column(Float(precision=53), primary_key=True)
    diffmaglim = Column(Float)
    __table_args__ = (Index("ix_non_detection_oid", "oid", postgresql_using="hash"),)


class Detection(Base, Commons):
    __tablename__ = "detection"

    candid = Column(BigInteger, primary_key=True)
    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    mjd = Column(Float(precision=53), nullable=False)
    fid = Column(Integer, nullable=False)
    pid = Column(Float, nullable=False)
    diffmaglim = Column(Float)
    isdiffpos = Column(Integer, nullable=False)
    nid = Column(Integer)
    ra = Column(Float(precision=53), nullable=False)
    dec = Column(Float(precision=53), nullable=False)
    magpsf = Column(Float, nullable=False)
    sigmapsf = Column(Float, nullable=False)
    magap = Column(Float)
    sigmagap = Column(Float)
    distnr = Column(Float)
    rb = Column(Float)
    rbversion = Column(String)
    drb = Column(Float)
    drbversion = Column(String)
    magapbig = Column(Float)
    sigmagapbig = Column(Float)
    rfid = Column(Integer)
    magpsf_corr = Column(Float)
    sigmapsf_corr = Column(Float)
    sigmapsf_corr_ext = Column(Float)
    corrected = Column(Boolean, nullable=False)
    dubious = Column(Boolean, nullable=False)
    parent_candid = Column(BigInteger)
    has_stamp = Column(Boolean, nullable=False)
    step_id_corr = Column(String, nullable=False)

    __table_args__ = (Index("ix_ndetection_oid", "oid", postgresql_using="hash"),)

    def __repr__(self):
        return "<Detection(candid='%i', fid='%i', oid='%s')>" % (
            self.candid,
            self.fid,
            self.oid,
        )


class ForcedPhotometry(Base):
    __tablename__ = "forced_photometry"

    pid = Column(BigInteger, primary_key=True)
    oid = Column(String, primary_key=True)
    mjd = Column(Float(precision=53), nullable=False)
    fid = Column(Integer, nullable=False)
    ra = Column(Float(precision=53), nullable=False)
    dec = Column(Float(precision=53), nullable=False)
    e_ra = Column(Float)
    e_dec = Column(Float)
    mag = Column(Float)
    e_mag = Column(Float)
    mag_corr = Column(Float)
    e_mag_corr = Column(Float)
    e_mag_corr_ext = Column(Float)
    isdiffpos = Column(Integer, nullable=False)
    corrected = Column(Boolean, nullable=False)
    dubious = Column(Boolean, nullable=False)
    parent_candid = Column(String)
    has_stamp = Column(Boolean, nullable=False)
    # extra fields
    field = Column(Integer)
    rcid = Column(Integer)
    rfid = Column(BigInteger)
    sciinpseeing = Column(Float)
    scibckgnd = Column(Float)
    scisigpix = Column(Float)
    magzpsci = Column(Float)
    magzpsciunc = Column(Float)
    magzpscirms = Column(Float)
    clrcoeff = Column(Float)
    clrcounc = Column(Float)
    exptime = Column(Float)
    adpctdif1 = Column(Float)
    adpctdif2 = Column(Float)
    diffmaglim = Column(Float)
    programid = Column(Integer)
    procstatus = Column(String)
    distnr = Column(Float)
    ranr = Column(Float(precision=53))
    decnr = Column(Float(precision=53))
    magnr = Column(Float)
    sigmagnr = Column(Float)
    chinr = Column(Float)
    sharpnr = Column(Float)

    __table_args__ = (
        Index("ix_forced_photometry_oid", "oid", postgresql_using="hash"),
    )


class Dataquality(Base):
    __tablename__ = "dataquality"

    candid = Column(BigInteger, primary_key=True)
    oid = Column(String, primary_key=True)
    fid = Column(Integer, nullable=False)
    xpos = Column(Float)
    ypos = Column(Float)
    chipsf = Column(Float)
    sky = Column(Float)
    fwhm = Column(Float)
    classtar = Column(Float)
    mindtoedge = Column(Float)
    seeratio = Column(Float)
    aimage = Column(Float)
    bimage = Column(Float)
    aimagerat = Column(Float)
    bimagerat = Column(Float)
    nneg = Column(Integer)
    nbad = Column(Integer)
    sumrat = Column(Float)
    scorr = Column(Float)
    dsnrms = Column(Float)
    ssnrms = Column(Float)
    magzpsci = Column(Float)
    magzpsciunc = Column(Float)
    magzpscirms = Column(Float)
    nmatches = Column(Integer)
    clrcoeff = Column(Float)
    clrcounc = Column(Float)
    zpclrcov = Column(Float)
    zpmed = Column(Float)
    clrmed = Column(Float)
    clrrms = Column(Float)
    exptime = Column(Float)

    __table_args__ = (
        ForeignKeyConstraint([candid, oid], [Detection.candid, Detection.oid]),
        {},
    )


class Gaia_ztf(Base):
    __tablename__ = "gaia_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    candid = Column(BigInteger, nullable=False)
    neargaia = Column(Float)
    neargaiabright = Column(Float)
    maggaia = Column(Float)
    maggaiabright = Column(Float)
    unique1 = Column(Boolean, nullable=False)


class Ss_ztf(Base):
    __tablename__ = "ss_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    candid = Column(BigInteger, nullable=False)
    ssdistnr = Column(Float)
    ssmagnr = Column(Float)
    ssnamenr = Column(String)

    __table_args__ = (
        Index("ix_ss_ztf_candid", "candid", postgresql_using="btree"),
        Index("ix_ss_ztf_ssnamenr", "ssnamenr", postgresql_using="btree"),
    )


class Ps1_ztf(Base):
    __tablename__ = "ps1_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    candid = Column(BigInteger, primary_key=True)
    objectidps1 = Column(Float)
    sgmag1 = Column(Float)
    srmag1 = Column(Float)
    simag1 = Column(Float)
    szmag1 = Column(Float)
    sgscore1 = Column(Float)
    distpsnr1 = Column(Float)
    objectidps2 = Column(Float)
    sgmag2 = Column(Float)
    srmag2 = Column(Float)
    simag2 = Column(Float)
    szmag2 = Column(Float)
    sgscore2 = Column(Float)
    distpsnr2 = Column(Float)
    objectidps3 = Column(Float)
    sgmag3 = Column(Float)
    srmag3 = Column(Float)
    simag3 = Column(Float)
    szmag3 = Column(Float)
    sgscore3 = Column(Float)
    distpsnr3 = Column(Float)
    nmtchps = Column(Integer, nullable=False)
    unique1 = Column(Boolean, nullable=False)
    unique2 = Column(Boolean, nullable=False)
    unique3 = Column(Boolean, nullable=False)


class Reference(Base):
    __tablename__ = "reference"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    rfid = Column(BigInteger, primary_key=True)
    candid = Column(BigInteger, nullable=False)
    fid = Column(Integer, nullable=False)
    rcid = Column(Integer)
    field = Column(Integer)
    magnr = Column(Float)
    sigmagnr = Column(Float)
    chinr = Column(Float)
    sharpnr = Column(Float)
    ranr = Column(Float(precision=53), nullable=False)
    decnr = Column(Float(precision=53), nullable=False)
    mjdstartref = Column(Float(precision=53), nullable=False)
    mjdendref = Column(Float(precision=53), nullable=False)
    nframesref = Column(Integer, nullable=False)

    __table_args__ = (Index("ix_reference_fid", "fid", postgresql_using="btree"),)


class Pipeline(Base):
    __tablename__ = "pipeline"

    pipeline_id = Column(String, primary_key=True)
    step_id_corr = Column(String)
    step_id_feat = Column(String)
    step_id_clf = Column(String)
    step_id_out = Column(String)
    step_id_stamp = Column(String)
    date = Column(DateTime, nullable=False)
