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
    nobs = Column(Integer)
    meanra = Column(Float)
    meandec = Column(Float)
    sigmara = Column(Float)
    sigmadec = Column(Float)
    deltajd = Column(Float)
    lastmjd = Column(Float)
    firstmjd = Column(Float)

    xmatch = relationship("Xmatch", uselist=False,
                          back_populates='astro_object')
    magref = relationship("MagRef", uselist=False,
                          back_populates='astro_object')
    magnitude_statistics = relationship(
        "MagnitudeStatistics", uselist=False, back_populates='astro_object')
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
    catalog_id = Column(String)
    catalog_oid = Column(String)
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


class MagnitudeStatistics(Base):
    __tablename__ = 'magnitude_statistics'

    id = Column(Integer, primary_key=True)
    magnitude_type = Column(String)
    fid = Column(Integer)
    mean = Column(Float)
    median = Column(Float)
    max_mag = Column(Float)
    min_mag = Column(Float)
    sigma = Column(Float)
    last = Column(Float)
    first = Column(Float)
    oid = Column(String, ForeignKey('astro_object.oid'))
    astro_object = relationship(
        "AstroObject", back_populates='magnitude_statistics')


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
    magpsf = Column(Float)
    magap = Column(Float)
    sigmapsf = Column(Float)
    sigmagap = Column(Float)
    ra = Column(Float)
    dec = Column(Float)
    rb = Column(Float)
    alert = Column(JSON)
    oid = Column(String, ForeignKey('astro_object.oid'))
