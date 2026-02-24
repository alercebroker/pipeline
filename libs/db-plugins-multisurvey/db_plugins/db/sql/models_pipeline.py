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
    # __mapper_args__ = {"primary_key": ["oid", "sid", "feature_id", "band"]}

    __table_args__ = (
        PrimaryKeyConstraint(
            "oid", "sid", "feature_id", "band", name="pk_feature_oid_featureid_band"
        ),
        {"postgresql_partition_by": "HASH (oid)"},
    )

    __n_partitions__ = 32

    @classmethod
    def __partition_on__(cls, partition_idx: int):
        return f"FOR VALUES WITH (MODULUS {cls.__n_partitions__}, REMAINDER {partition_idx})"