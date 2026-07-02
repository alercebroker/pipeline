"""Legacy ZTF crossmatch ORM models, vendored for metadata_step.

These five models (`reference`, `ss_ztf`, `ps1_ztf`, `dataquality`, `gaia_ztf`)
used to live in `db_plugins.db.sql.models`, but pipeline commit 09587bc rewrote
`libs/db-plugins` into the unified multi-survey schema, renaming them
(`ztf_reference`, `ztf_ss`, `ztf_ps1`, `ztf_dataquality`) and remapping them to
different tables. metadata_step is the only step that still writes the original
legacy tables (which are what the AWS-legacy and on-prem/quimal databases
physically have), so the definitions are pinned here verbatim from db-plugins at
09587bc~1 (version 27.5.7a25) to decouple this step from the rewritten lib.

Definitions are copied exactly except that the foreign keys to the `object` and
`detection` tables are dropped. They are never used here (no DDL, no
relationships, no joins; metadata_step only does select-by-oid and
insert-with-on_conflict against pre-existing tables), and keeping them as string
refs makes SQLAlchemy try to resolve `object`/`detection` in this standalone
MetaData during the ORM bulk-insert table-sort, raising NoReferencedTableError.
The physical tables retain their FKs regardless.
"""

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    Index,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


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


class Gaia_ztf(Base):
    __tablename__ = "gaia_ztf"

    oid = Column(String, primary_key=True)
    candid = Column(BigInteger, nullable=False)
    neargaia = Column(Float)
    neargaiabright = Column(Float)
    maggaia = Column(Float)
    maggaiabright = Column(Float)
    unique1 = Column(Boolean, nullable=False)


class Ss_ztf(Base):
    __tablename__ = "ss_ztf"

    oid = Column(String, primary_key=True)
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

    oid = Column(String, primary_key=True)
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

    oid = Column(String, primary_key=True)
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
