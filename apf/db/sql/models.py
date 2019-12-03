from . import engine, Base
from sqlalchemy import Column, Integer, String, Table, ForeignKey, Float, Boolean, JSON
from sqlalchemy.orm import relationship

taxonomy_class = Table('taxonomy_class', Base.metadata,
                       Column('class_id', Integer, ForeignKey('class.id')),
                       Column('taxonomy_id', Integer,
                              ForeignKey('taxonomy.id'))
                       )


class Class(Base):
    __tablename__ = 'class'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    taxonomies = relationship(
        "Taxonomy",
        secondary=taxonomy_class,
        back_populates="classes")


class Taxonomy(Base):
    __tablename__ = 'taxonomy'

    id = Column(integer, primary_key=True)
    name = Column(String)
    classes = relationship(
        "Class",
        secondary=taxonomy_class,
        back_populates="taxonomies"
    )
    classifiers = relationship("Classifier")


class Classifier(Base):
    __tablename__ = 'classifier'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    taxonomy_id = Column(Integer, ForeignKey('taxonomy.id'))


class AstroObject(Base):
    __tablename__ = 'astro_object'

    oid = Column(String, primary_key=True)
    xmatch = relationship("Xmatch", uselist=False,
                          back_populates='astro_object')
    magref = relationship("MagRef", uselist=False,
                          back_populates='astro_object')
    statistics = relationship(
        "Statistics", uselist=False, back_populates='astro_object')
    features = relationship("Features")
    classification = relationship("Classification")
    non_detections = relationship("NonDetection")
    detections = relationship("Detection")


class Classification(Base):
    __tablename__ = 'classification'

    id = Column(Integer, primary_key=True)
    class_name = Column(String)
    probability = Column(Float)
    astro_object = Column(String, ForeignKey('astro_object.oid'))


class Xmatch(Base):
    __tablename__ = 'xmatch'

    id = Column(Integer, primary_key=True)
    class_name = Column(String)
    oid = Column(String, ForeignKey('astro_object.oid'))
    astro_object = relationship("AstroObject", back_populates='xmatch')


class MagRef(Base):
    __tablename__ = 'magref'

    id = Column(Integer, primary_key=True)
    fid = Column(Integer)
    rcid = Column(Integer)
    field = Column(Integer)
    magref = Column(Float)
    sigmagref = Column(Float)
    corrected = Column(Boolean)
    oid = Column(String, ForeignKey('astro_object.oid'))
    astro_object = relationship("AstroObject", back_populates='magref')


class Statistics(Base):
    __tablename__ = 'statistics'

    id = Column(Integer, primary_key=True)
    nobs = Column(Integer)
    mean_magap_g = Column(Float)
    mean_magap_r = Column(Float)
    median_magap_g = Column(Float)
    median_magap_r = Column(Float)
    max_magap_g = Column(Float)
    max_magap_r = Column(Float)
    min_magap_g = Column(Float)
    min_magap_r = Column(Float)
    sigma_magap_g = Column(Float)
    sigma_magap_r = Column(Float)
    last_magap_g = Column(Float)
    last_magap_r = Column(Float)
    first_magap_g = Column(Float)
    first_magap_r = Column(Float)
    mean_magpsf_g = Column(Float)
    mean_magpsf_r = Column(Float)
    median_magpsf_g = Column(Float)
    median_magpsf_r = Column(Float)
    max_magpsf_g = Column(Float)
    max_magpsf_r = Column(Float)
    min_magpsf_g = Column(Float)
    min_magpsf_r = Column(Float)
    sigma_magpsf_g = Column(Float)
    sigma_magpsf_r = Column(Float)
    last_magpsf_g = Column(Float)
    last_magpsf_r = Column(Float)
    first_magpsf_g = Column(Float)
    first_magpsf_r = Column(Float)
    meanra = Column(Float)
    meandec = Column(Float)
    sigmara = Column(Float)
    sigmadec = Column(Float)
    deltajd = Column(Float)
    lastmjd = Column(Float)
    firstmjd = Column(Float)

    oid = Column(String, ForeignKey('astro_object.oid'))
    astro_object = relationship("AstroObject", back_populates='statistics')


class Features(Base):
    __tablename__ = 'features'

    id = Column(Integer)
    data = Column(JSON)
    oid = Column(String, ForeignKey('astro_object.oid'))


class NonDetection(Base):
    __tablename__ = 'non_detection'

    id = Column(Integer, primary_key=True)
    mjd = Column(Float)
    diffmaglim = Column(Float)
    fid = Column(Integer)
    oid = Column(String, ForeignKey('astro_object.oid'))


class Detection(Base):
    __tablename__ = 'detection'

    id = Column(Integer, primary_key=True)
    candid = Column(Integer)
    mjd = Column(Float)
    fid = Column(Integer)
    diffmaglim = Column(Float)
    magpsf = Column(Float)
    magap = Column(Float)
    sigmapsf = Column(Float)
    sigmagap = Column(Float)
    ra = Column(Float)
    dec = Column(Float)
    sigmara = Column(Float)
    sigmadec = Column(Float)
    isdiffpos = Column(Integer)
    distpsnr1 = Column(Float)
    sgscore1 = Column(Float)
    field = Column(Integer)
    rcid = Column(Integer)
    magnr = Column(Float)
    sigmagnr = Column(Float)
    rb = Column(Float)
    magpsf_corr = Column(Float)
    magap_corr = Column(Float)
    sigmapsf_corr = Column(Float)
    sigmagap_corr = Column(Float)

    oid = Column(String, ForeignKey('astro_object.oid'))
