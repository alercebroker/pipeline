"""Persist offline ZTF DB-ready feature rows into <schema>.feature.

Writing lives here, not in db.py (which is read-only). Takes the DataFrame
produced by lc_features.compute_db_features and upserts it one row at a time
via ON CONFLICT (oid, sid, feature_id, band) DO UPDATE.
"""
import logging

import pandas as pd
from sqlalchemy import text

from features.offline import db

log = logging.getLogger(__name__)


def _records(rows: pd.DataFrame) -> list:
    """Sanitize the frame into native-typed dict records for the upsert.

    - Drop rows with NaN feature_id (cannot satisfy the NOT-NULL PK); warn.
    - Warn if any version == -1 (version_name absent from feature_version_lut).
    - Cast to native Python types so the >2**53 oid bigint and the smallints are
      not coerced to float by the driver; NaN/None value -> SQL NULL.
    """
    n_before = len(rows)
    rows = rows[rows["feature_id"].notna()]
    dropped = n_before - len(rows)
    if dropped:
        log.warning("Dropping %d feature row(s) with unmapped (NaN) feature_id", dropped)

    if len(rows) and (rows["version"] == -1).any():
        log.warning("%d row(s) have version=-1 (version_name not in feature_version_lut)",
                    int((rows["version"] == -1).sum()))

    records = []
    for r in rows.to_dict("records"):
        value = r["value"]
        records.append({
            "oid": int(r["oid"]),
            "sid": int(r["sid"]),
            "feature_id": int(r["feature_id"]),
            "band": int(r["band"]),
            "version": int(r["version"]),
            "value": None if value is None or pd.isna(value) else float(value),
        })
    return records


def write_features(rows: pd.DataFrame, credentials: str, schema: str = None,
                   execute: bool = False) -> dict:
    """Upsert DB-ready feature rows into <schema>.feature.

    Dry-run by default (execute=False): returns {"executed": False,
    "would_write": N} and opens no DB connection. With execute=True, upserts all
    records in one transaction. schema defaults to db.SCHEMA.
    """
    schema = schema or db.SCHEMA
    records = _records(rows)
    n = len(records)

    if not execute:
        return {"executed": False, "would_write": n}

    sql = text(
        f"INSERT INTO {schema}.feature (oid, sid, feature_id, band, version, value) "
        "VALUES (:oid, :sid, :feature_id, :band, :version, :value) "
        "ON CONFLICT (oid, sid, feature_id, band) "
        "DO UPDATE SET value = EXCLUDED.value, version = EXCLUDED.version, updated_date = now()"
    )
    engine = db._make_engine(credentials)
    with engine.begin() as conn:
        if records:
            conn.execute(sql, records)
    return {"executed": True, "written": n}
