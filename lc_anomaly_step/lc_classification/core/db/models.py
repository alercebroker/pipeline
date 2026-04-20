from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func
from sqlalchemy.schema import PrimaryKeyConstraint
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class AnomalyScoreTop(Base):
    __tablename__ = "scores_top"

    oid = Column(String, primary_key=True)
    update_date = Column(
        DateTime, primary_key=False, server_default=func.now()
    )
    created_date = Column(DateTime, primary_key=False, onupdate=func.now())
    Transient = Column(Float, primary_key=False)
    Stochastic = Column(Float, primary_key=False)
    Periodic = Column(Float, primary_key=False)


class AnomalyScore(Base):
    __tablename__ = "scores"

    oid = Column(String, primary_key=True)
    update_date = Column(DateTime, primary_key=False, onupdate=func.now())
    created_date = Column(
        DateTime, primary_key=False, server_default=func.now()
    )
    SNIa = Column(Float, primary_key=False)
    SNIbc = Column(Float, primary_key=False)
    SNIIb = Column(Float, primary_key=False)
    SNII = Column(Float, primary_key=False)
    SNIIn = Column(Float, primary_key=False)
    SLSN = Column(Float, primary_key=False)
    TDE = Column(Float, primary_key=False)
    Microlensing = Column(Float, primary_key=False)
    QSO = Column(Float, primary_key=False)
    AGN = Column(Float, primary_key=False)
    Blazar = Column(Float, primary_key=False)
    YSO = Column(Float, primary_key=False)
    CVNova = Column(Float, primary_key=False)
    LPV = Column(Float, primary_key=False)
    EA = Column(Float, primary_key=False)
    EBEW = Column(Float, primary_key=False)
    PeriodicOther = Column(Float, primary_key=False)
    RSCVn = Column(Float, primary_key=False)
    CEP = Column(Float, primary_key=False)
    RRLab = Column(Float, primary_key=False)
    RRLc = Column(Float, primary_key=False)
    DSCT = Column(Float, primary_key=False)


class AnomalyDistributions(Base):
    __tablename__ = "distributions"

    name = Column(String, primary_key=True)
    category = Column(String, primary_key=True)
    value = Column(Float)
    update_date = Column(DateTime, primary_key=False, onupdate=func.now())
    created_date = Column(
        DateTime, primary_key=False, server_default=func.now()
    )
    __table_args__ = (PrimaryKeyConstraint("name", "category"),)


class AnomalyEmbeddings(Base):
    __tablename__ = "embeddings"

    oid = Column(String, primary_key=True)
    update_date = Column(DateTime, primary_key=False, onupdate=func.now())
    created_date = Column(
        DateTime, primary_key=False, server_default=func.now()
    )
    embedding = Column(Vector(64))
