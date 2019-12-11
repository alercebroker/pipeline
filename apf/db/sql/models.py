from . import Base
from sqlalchemy import Column, Integer, String, Table, ForeignKey, Float, Boolean, JSON, Index
from sqlalchemy.orm import relationship
from .. import generic

taxonomy_class = Table('taxonomy_class', Base.metadata,
                       Column('class_name', String, ForeignKey('class.name')),
                       Column('taxonomy_name', String,
                              ForeignKey('taxonomy.name'))
                       )


class Class(Base, generic.AbstractClass):
    __tablename__ = 'class'

    name = Column(String, primary_key=True)
    acronym = Column(String)
    taxonomies = relationship(
        "Taxonomy",
        secondary=taxonomy_class,
        back_populates="classes")
    classifications = relationship("Classification")

    def get_taxonomies(self):
        return self.taxonomies

    def __repr__(self):
        return "<Class(name='%s', acronym='%s')>" % (self.name, self.acronym)


class Taxonomy(Base, generic.AbstractTaxonomy):
    __tablename__ = 'taxonomy'

    name = Column(String, primary_key=True)
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
    name = Column(String, primary_key=True)
    taxonomy_name = Column(String, ForeignKey('taxonomy.name'))
    classifications = relationship("Classification")

    def get_classifications(self):
        return self.classifications

    def __repr__(self):
        return "<Classifier(name='%s')>" % (self.name)


class AstroObject(Base, generic.AbstractAstroObject):
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

    xmatches = relationship("Xmatch")
    magnitude_statistics = relationship("MagnitudeStatistics", uselist=False)
    classifications = relationship("Classification")
    non_detections = relationship("NonDetection")
    detections = relationship("Detection")
    features = relationship("FeaturesObject")

    def get_classifications(self):
        return self.classifications

    def get_magnitude_statistics(self):
        return self.magnitude_statistics

    def get_xmatches(self):
        return self.xmatches

    def get_non_detections(self):
        return self.non_detections

    def get_detections(self):
        return self.detections

    # TODO implement light curve formatting
    def get_lightcurve(self):
        pass

    def __repr__(self):
        return "<AstroObject(oid='%s')>" % (self.oid)


class Classification(Base):
    __tablename__ = 'classification'

    class_name = Column(String, ForeignKey('class.name'), primary_key=True)
    probability = Column(Float)
    astro_object = Column(String, ForeignKey(
        'astro_object.oid'), primary_key=True)
    classifier_name = Column(String, ForeignKey(
        'classifier.name'), primary_key=True)

    classes = relationship("Class", back_populates='classifications')
    objects = relationship("AstroObject", back_populates='classifications')
    classifiers = relationship("Classifier", back_populates='classifications')

    def __repr__(self):
        return "<Classification(class_name='%s', probability='%s', astro_object='%s', classifier_name='%s')>" % (self.class_name,
                                                                                                                 self.probability, self.astro_object, self.classifier_name)


class Xmatch(Base, generic.AbstractXmatch):
    __tablename__ = 'xmatch'

    catalog_id = Column(String, primary_key=True)
    catalog_oid = Column(String, primary_key=True)
    oid = Column(String, ForeignKey('astro_object.oid'))


class MagnitudeStatistics(Base, generic.AbstractMagnitudeStatistics):
    __tablename__ = 'magnitude_statistics'

    magnitude_type = Column(String)
    fid = Column(Integer, primary_key=True)
    mean = Column(Float)
    median = Column(Float)
    max_mag = Column(Float)
    min_mag = Column(Float)
    sigma = Column(Float)
    last = Column(Float)
    first = Column(Float)
    oid = Column(String, ForeignKey('astro_object.oid'), primary_key=True)


class Features(Base, generic.AbstractFeatures):
    __tablename__ = 'features'

    version = Column(String, primary_key=True)


class FeaturesObject(Base):
    __tablename__ = 'features_object'

    features_version = Column(String, ForeignKey(
        'features.version'), primary_key=True)
    object_id = Column(String, ForeignKey(
        'astro_object.oid'), primary_key=True)
    data = Column(JSON)
    features = relationship("Features")


class NonDetection(Base, generic.AbstractNonDetection):
    __tablename__ = 'non_detection'

    mjd = Column(Float, primary_key=True)
    diffmaglim = Column(Float, nullable=False)
    fid = Column(Integer, primary_key=True)
    oid = Column(String, ForeignKey('astro_object.oid'), primary_key=True)


class Detection(Base, generic.AbstractDetection):
    __tablename__ = 'detection'

    candid = Column(String, primary_key=True)
    mjd = Column(Float, nullable=False)
    fid = Column(Integer, nullable=False)
    magpsf = Column(Float, nullable=False)
    magap = Column(Float, nullable=False)
    sigmapsf = Column(Float, nullable=False)
    sigmagap = Column(Float, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    rb = Column(Float)
    alert = Column(JSON, nullable=False)
    oid = Column(String, ForeignKey('astro_object.oid'), nullable=False)

    __table_args__ = (Index('object_id', 'oid', postgresql_using='btree'),)
