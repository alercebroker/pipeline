from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Table,
    ForeignKey,
    Float,
    Boolean,
    JSON,
    ARRAY,
    Index,
    DateTime,
    UniqueConstraint,
    ForeignKeyConstraint,
)
from sqlalchemy.orm import relationship
from .. import generic

from db_plugins.db.sql import Base


class Commons:
    def __getitem__(self, field):
        return self.__dict__[field]


class Object(Base, generic.AbstractObject):
    __tablename__ = "object"

    oid = Column(String, primary_key=True)
    ndethist = Column(Integer)
    ncovhist = Column(Integer)
    mjdstarthist = Column(Float(precision=53))
    mjdendhist = Column(Float(precision=53))
    corrected = Column(Boolean)
    stellar = Column(Boolean)
    ndet = Column(Integer)
    g_r_max = Column(Float)
    g_r_max_corr = Column(Float)
    g_r_mean = Column(Float)
    g_r_mean_corr = Column(Float)
    meanra = Column(Float(precision=53))
    meandec = Column(Float(precision=53))
    sigmara = Column(Float(precision=53))
    sigmadec = Column(Float(precision=53))
    deltajd = Column(Float(precision=53))
    firstmjd = Column(Float(precision=53))
    lastmjd = Column(Float(precision=53))
    step_id_corr = Column(String)

    __table_args__ = (
        Index("object_ndet", "ndet", postgresql_using="btree"),
        Index("object_firstmjd", "firstmjd", postgresql_using="btree"),
        Index("object_lastmjd", "lastmjd", postgresql_using="btree"),
    )

    # xmatches = relationship("Xmatch")
    magstats = relationship("MagStats", uselist=True)
    non_detections = relationship("NonDetection", order_by="NonDetection.mjd")
    detections = relationship("Detection", order_by="Detection.mjd")
    features = relationship("Feature")
    probabilities = relationship("Probability", uselist=True, order_by="Probability.classifier_name")

    def get_lightcurve(self):
        return {
            "detections": self.detections,
            "non_detections": self.non_detections,
        }

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


class FeatureVersion(Base):
    __tablename__ = "feature_version"
    version = Column(String, primary_key=True)
    step_id_feature = Column(String, ForeignKey("step.step_id"))
    step_id_preprocess = Column(String, ForeignKey("step.step_id"))


class Feature(Base):
    __tablename__ = "feature"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    name = Column(String, primary_key=True, nullable=False)
    value = Column(Float(precision=53), nullable=False)
    fid = Column(Integer, primary_key=True)
    version = Column(
        String, ForeignKey("feature_version.version"), primary_key=True, nullable=False
    )


# class Xmatch(Base, generic.AbstractXmatch):
#     __tablename__ = "xmatch"
#
#     oid = Column(String, ForeignKey("object.oid"), primary_key=True)
#     catalog_id = Column(String, primary_key=True)
#     catalog_oid = Column(String, primary_key=True)


class MagStats(Base, generic.AbstractMagnitudeStatistics):
    __tablename__ = "magstat"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    fid = Column(Integer, primary_key=True)
    stellar = Column(Boolean)
    corrected = Column(Boolean)
    ndet = Column(Integer)
    ndubious = Column(Integer)
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
    # step_id_corr = Column(String)

    __table_args__ = (
        Index("mag_mean", "magmean", postgresql_using="btree"),
        Index("mag_median", "magmedian", postgresql_using="btree"),
        Index("mag_min", "magmin", postgresql_using="btree"),
        Index("mag_max", "magmax", postgresql_using="btree"),
        Index("mag_first", "magfirst", postgresql_using="btree"),
        Index("mag_last", "maglast", postgresql_using="btree"),
    )


class NonDetection(Base, generic.AbstractNonDetection, Commons):
    __tablename__ = "non_detection"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    fid = Column(Integer, primary_key=True)
    mjd = Column(Float(precision=53), primary_key=True)
    diffmaglim = Column(Float)
    __table_args__ = (Index("non_det_oid", "oid", postgresql_using="hash"),)


class Detection(Base, generic.AbstractDetection, Commons):
    __tablename__ = "detection"

    candid = Column(BigInteger, primary_key=True)
    oid = Column(String, ForeignKey("object.oid"), nullable=False)
    mjd = Column(Float(precision=53))
    fid = Column(Integer)
    pid = Column(Float)
    diffmaglim = Column(Float)
    isdiffpos = Column(Integer)
    nid = Column(Integer)
    ra = Column(Float(precision=53))
    dec = Column(Float(precision=53))
    magpsf = Column(Float)
    sigmapsf = Column(Float)
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
    corrected = Column(Boolean)
    dubious = Column(Boolean)
    parent_candid = Column(BigInteger)
    has_stamp = Column(Boolean)
    step_id_corr = Column(String)

    __table_args__ = (Index("object_id", "oid", postgresql_using="hash"),)

    dataquality = relationship("Dataquality")

    def __repr__(self):
        return "<Detection(candid='%i', fid='%i', oid='%s')>" % (
            self.candid,
            self.fid,
            self.oid,
        )


class Dataquality(Base, generic.AbstractDataquality):
    __tablename__ = "dataquality"

    candid = Column(BigInteger, ForeignKey("detection.candid"), primary_key=True)
    oid = Column(String, nullable=False)
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
        Index("index_candid", "candid", postgresql_using="btree"),
        Index("index_fid", "fid", postgresql_using="btree"),
    )


class Gaia_ztf(Base, generic.AbstractGaia_ztf):
    __tablename__ = "gaia_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    neargaia = Column(Float, nullable=False)
    neargaiabright = Column(Float, nullable=False)
    maggaia = Column(Float, nullable=False)
    maggaiabright = Column(Float, nullable=False)
    unique1 = Column(Boolean)


class Ss_ztf(Base, generic.AbstractSs_ztf):
    __tablename__ = "ss_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    ssdistnr = Column(Float, nullable=False)
    ssmagnr = Column(Float, nullable=False)
    ssnamenr = Column(String)


class Ps1_ztf(Base, generic.AbstractPs1_ztf):
    __tablename__ = "ps1_ztf"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    candid = Column(BigInteger, primary_key=True)
    objectidps1 = Column(Float, nullable=False)
    sgmag1 = Column(Float, nullable=False)
    srmag1 = Column(Float, nullable=False)
    simag1 = Column(Float, nullable=False)
    szmag1 = Column(Float, nullable=False)
    sgscore1 = Column(Float, nullable=False)
    distpsnr1 = Column(Float, nullable=False)
    objectidps2 = Column(Float, nullable=False)
    sgmag2 = Column(Float, nullable=False)
    srmag2 = Column(Float, nullable=False)
    simag2 = Column(Float, nullable=False)
    szmag2 = Column(Float, nullable=False)
    sgscore2 = Column(Float, nullable=False)
    distpsnr2 = Column(Float, nullable=False)
    objectidps3 = Column(Float, nullable=False)
    sgmag3 = Column(Float, nullable=False)
    srmag3 = Column(Float, nullable=False)
    simag3 = Column(Float, nullable=False)
    szmag3 = Column(Float, nullable=False)
    sgscore3 = Column(Float, nullable=False)
    distpsnr3 = Column(Float, nullable=False)
    nmtchps = Column(Integer)
    unique1 = Column(Boolean)
    unique2 = Column(Boolean)
    unique3 = Column(Boolean)


class Reference(Base, generic.AbstractReference):
    __tablename__ = "reference"

    oid = Column(String, ForeignKey("object.oid"), primary_key=True)
    rfid = Column(BigInteger, primary_key=True)
    candid = Column(BigInteger)
    fid = Column(Integer)
    rcid = Column(Integer, nullable=False)
    field = Column(Integer, nullable=False)
    magnr = Column(Float, nullable=False)
    sigmagnr = Column(Float, nullable=False)
    chinr = Column(Float, nullable=False)
    sharpnr = Column(Float, nullable=False)
    ranr = Column(Float(precision=53))
    decnr = Column(Float(precision=53))
    mjdstartref = Column(Float(precision=53))
    mjdendref = Column(Float(precision=53))
    nframesref = Column(Integer)


class Pipeline(Base, generic.AbstractPipeline):
    __tablename__ = "pipeline"

    pipeline_id = Column(String, primary_key=True)
    step_id_corr = Column(String, nullable=False)
    step_id_feat = Column(String, nullable=False)
    step_id_clf = Column(String, nullable=False)
    step_id_out = Column(String, nullable=False)
    step_id_stamp = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)


class Step(Base, generic.AbstractStep):
    __tablename__ = "step"

    step_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    comments = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
