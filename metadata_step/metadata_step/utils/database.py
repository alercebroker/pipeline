import numpy as np
from contextlib import contextmanager
from typing import Callable, ContextManager, Dict, List
import logging

from sqlalchemy import create_engine, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker, Session
from db_plugins.db.sql.models import Reference, Gaia_ztf, Ss_ztf, Dataquality, Ps1_ztf


class PSQLConnection:
    def __init__(self, config: dict, echo: bool = False) -> None:
        url = self.__format_db_url(config)
        self._engine = create_engine(url, echo=echo)
        self._session_factory = sessionmaker(
            self._engine,
        )

    def __format_db_url(self, config):
        return (
            f"postgresql://{config['USER']}:{config['PASSWORD']}@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
        )

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception as e:
            logging.error(e)
            session.rollback()
            raise
        finally:
            session.close()


def _none_to_nan(value):
    if value is None:
        return np.nan
    return value


def _nan_to_none(value):
    if value == np.nan:
        return None
    return value


def _parse_sql_dict(d: Dict):
    return {k: _none_to_nan(v) for k, v in d.items() if not k.startswith("_")}


def _parse_ps1(d: Dict):
    parsed = _parse_sql_dict(d)
    parsed["updated"] = False
    return parsed


def _create_hashmap(catalog: List[Dict]):
    hashmap: Dict[List] = {}
    for item in catalog:
        catalog_of_oid: List = hashmap.get(item["oid"], [])
        catalog_of_oid.append(item)
        hashmap[item["oid"]] = catalog_of_oid

    return hashmap


def get_ps1_catalog(session: Session, oids: List):
    stmt = select(Ps1_ztf).where(Ps1_ztf.oid.in_(oids))
    catalog = session.execute(stmt).all()
    return _create_hashmap([_parse_ps1(c[0].__dict__) for c in catalog])


def get_gaia_catalog(session: Session, oids: List):
    stmt = select(Gaia_ztf).where(Gaia_ztf.oid.in_(oids))
    catalog = session.execute(stmt).all()
    return _create_hashmap([_parse_sql_dict(c[0].__dict__) for c in catalog])


def _accumulate(data: List):
    acc = {"ss": [], "reference": [], "dataquality": [], "gaia": [], "ps1": []}
    for d in data:
        acc["ss"].append(d["ss"])
        acc["dataquality"].append(d["dataquality"])
        acc["reference"].append(d["reference"])
        acc["gaia"].append(d["gaia"])
        acc["ps1"].append(d["ps1"])

    return acc


def insert_metadata(session: Session, data: List, ps1_updates: List):
    accumulated_metadata = _accumulate(data)
    # Reference
    reference_stmt = insert(Reference)
    reference_stmt = reference_stmt.on_conflict_do_update(
        constraint="reference_pkey", set_=dict(oid=reference_stmt.excluded.oid)
    )
    session.execute(reference_stmt, accumulated_metadata["reference"])
    session.commit()

    # SS
    ss_stmt = insert(Ss_ztf)
    ss_stmt = ss_stmt.on_conflict_do_update(constraint="ss_ztf_pkey", set_=dict(oid=ss_stmt.excluded.oid))
    session.execute(ss_stmt, accumulated_metadata["ss"])
    session.commit()

    # Dataquality
    dataquality_stmt = insert(Dataquality)
    dataquality_stmt = dataquality_stmt.on_conflict_do_update(
        constraint="dataquality_pkey", set_=dict(oid=dataquality_stmt.excluded.oid)
    )
    session.execute(dataquality_stmt, accumulated_metadata["dataquality"])
    session.commit()

    # GAIA
    gaia_stmt = insert(Gaia_ztf)
    gaia_stmt = gaia_stmt.on_conflict_do_update(
        constraint="gaia_ztf_pkey", set_=dict(unique1=gaia_stmt.excluded.unique1)
    )
    session.execute(gaia_stmt, accumulated_metadata["gaia"])
    session.commit()

    # PS1
    ps1_data = list({el["candid"]: el for el in accumulated_metadata["ps1"]}.values())
    ps1_stmt = insert(Ps1_ztf)
    ps1_stmt = ps1_stmt.on_conflict_do_update(constraint="ps1_ztf_pkey", set_=dict(oid=ps1_stmt.excluded.oid))
    session.execute(ps1_stmt, ps1_data)
    session.commit()
