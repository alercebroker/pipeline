from . import Base
from sqlalchemy import Column, Integer, String, Table, ForeignKey, Float, Boolean, JSON
from sqlalchemy.orm import relationship
from .. import generic

taxonomy_class = Table('taxonomy_class', Base.metadata,
                       Column('class_id', Integer, ForeignKey('class.id')),
                       Column('taxonomy_id', Integer,
                              ForeignKey('taxonomy.id'))
                       )


class Class(Base, generic.AbstractClass):
    __tablename__ = 'class'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    acronym = Column(String)
    taxonomies = relationship(
        "Taxonomy",
        secondary=taxonomy_class,
        back_populates="classes")

    def get_taxonomies(self):
        return self.taxonomies

    def __repr__(self):
        return "<Class(name='%s', acronym='%s')>" % (self.name, self.acronym)


class Taxonomy(Base, generic.AbstractTaxonomy):
    __tablename__ = 'taxonomy'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    classes = relationship(
        "Class",
        secondary=taxonomy_class,
        back_populates="taxonomies"
    )
    classifiers = relationship("Classifier")

    def get_classes(self):
        return self.classes

    def get_classifiers(self):
        return self.classifiers

    def __repr__(self):
        return "<Taxonomy(name='%s')>" % (self.name)


class Classifier(Base, generic.AbstractClassifier):
    __tablename__ = 'classifier'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    taxonomy_id = Column(Integer, ForeignKey('taxonomy.id'))
    features = relationship("Features")
    classifications = relationship("Classification")

    def get_features(self):
        return self.features

    def get_classifications(self):
        return self.classifications

    def __repr__(self):
        return "<Classifier(name='%s')>" % (self.name)


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

    xmatches = relationship("Xmatch", back_populates='astro_object')
    magref = relationship("MagRef", uselist=False,
                          back_populates='astro_object')
    magnitude_statistics = relationship(
        "MagnitudeStatistics", uselist=False, back_populates='astro_object')
    features = relationship("Features")
    classifications = relationship("Classification")
    non_detections = relationship("NonDetection")
    detections = relationship("Detection")

    def get_classifications(self):
        return self.classifications

    def get_magnitude_statistics(self):
        return self.magnitude_statistics

    def get_xmatches(self):
        return self.xmatches

    def get_magref(self):
        return self.magref

    def get_features(self):
        return self.features

    def get_non_detections(self):
        return self.non_detections

    def get_detections(self):
        return self.detections

    def __repr__(self):
        return "<AstroObject(oid='')>" % (self.oid)


class Classification(Base):
    __tablename__ = 'classification'

    id = Column(Integer, primary_key=True)
    class_name = Column(String)
    probability = Column(Float)
    astro_object = Column(String, ForeignKey('astro_object.oid'))

    classifier = relationship("Classifier", back_populates="classifications")


class Xmatch(Base):
    __tablename__ = 'xmatch'

    catalog_id = Column(String, primary_key=True)
    catalog_oid = Column(String, primary_key=True)
    oid = Column(String, ForeignKey('astro_object.oid'))
    astro_object = relationship("AstroObject", back_populates='xmatches')


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

    id = Column(Integer, primary_key=True)
    data = Column(JSON)
    oid = Column(String, ForeignKey('astro_object.oid'))
    classifier = relationship("Classifier", back_populates='features')


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
