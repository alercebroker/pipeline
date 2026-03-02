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
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, REAL
from sqlalchemy.orm import DeclarativeBase, relationship


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

class ClassifierVersion(Base):
    __tablename__ = "classifier_version"

    id = Column(Integer, primary_key=True, autoincrement=True)
    classifier_name = Column(VARCHAR, nullable=False)
    classifier_version = Column(VARCHAR, nullable=False)
    features_version = Column(VARCHAR, nullable=True)
    release_date = Column(Date, server_default=func.now())

    taxonomies = relationship("Taxonomy", back_populates="classifier_version")
    probabilities = relationship("ProbabilityArchive", back_populates="classifier_version")


class Taxonomy(Base):
    __tablename__ = "taxonomy"

    id = Column(Integer, primary_key=True, autoincrement=True)

    classifier_version_id = Column(
        Integer,
        ForeignKey("classifier_version.id"),
        nullable=False,
    )

    class_name = Column(VARCHAR, nullable=False)
    order = Column(Integer, nullable=False)

    classifier_version = relationship("ClassifierVersion", back_populates="taxonomies")
    probabilities = relationship("ProbabilityArchive", back_populates="taxonomy")


class SidLut(Base):
    __tablename__ = "sid_lut"

    sid = Column(SmallInteger, primary_key=True)
    tid = Column(SmallInteger)
    survey_name = Column(VARCHAR, nullable=False)

    created_date = Column(Date, server_default=func.now())


class ProbabilityArchive(Base):
    __tablename__ = "probability_archive"

    oid = Column(BigInteger, nullable=False)
    sid = Column(SmallInteger, nullable=False)

    classifier_version_id = Column(
        Integer,
        ForeignKey("classifier_version.id"),
        nullable=False,
    )

    class_id = Column(
        Integer,
        ForeignKey("taxonomy.id"),
        nullable=False,
    )

    probability = Column(REAL, nullable=False)
    ranking = Column(SmallInteger)

    creation_date = Column(Date, server_default=func.now())
    update_date = Column(Date, onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid",
            "sid",
            "classifier_version_id",
            "class_id",
            name="pk_probability_archive",
        ),
        Index("ix_probability_archive_oid", "oid", postgresql_using="hash"),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 16

    @classmethod
    def __partition_on__(cls, idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {idx})"

    classifier_version = relationship("ClassifierVersion", back_populates="probabilities")
    taxonomy = relationship("Taxonomy", back_populates="probabilities")