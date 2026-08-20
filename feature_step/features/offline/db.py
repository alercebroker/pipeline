"""SQL readers against the multisurvey_ztf.* schema for ZTF (sid = 0).

All readers take a credentials path + oid list and return DataFrames whose
columns already carry the production parser field names — the DB→parser
mapping lives in the SQL aliases (see the query table in
docs/superpowers/specs/2026-06-04-features-ztf-design.md).
"""
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

# multisurvey_ztf is the real/backfilled ZTF dataset; plain `multisurvey` was a
# short ~3-month slice. Override via OFFLINE_DB_SCHEMA if needed.
SCHEMA = os.getenv("OFFLINE_DB_SCHEMA", "multisurvey_ztf")
ALERCE_SCHEMA = "alerce"
SID = 0  # ZTF, per multisurvey_ztf.sid_lut (verified 2026-06-23)
ZTF_BAND_MAP = {1: "g", 2: "r", 3: "i"}
# From multisurvey_ztf.catalog_id_lut (verified 2026-06-23): catid of AllWISE.
ALLWISE_CATID = 0

log = logging.getLogger(__name__)


# Engine cache, keyed by (pid, credentials path). Every reader calls
# _make_engine, so without this a batched run pays a fresh TCP+auth handshake per
# reader per oid. Keying on the pid makes it fork-safe: a forked child gets a new
# pid and therefore builds its own engine instead of reusing the parent's
# sockets (psycopg2 connections are NOT fork-safe).
_ENGINES: dict = {}


def _make_engine(credentials_json: str) -> sa.Engine:
    key = (os.getpid(), os.path.abspath(credentials_json))
    engine = _ENGINES.get(key)
    if engine is not None:
        return engine
    with open(credentials_json, "r", encoding="utf-8") as f:
        params = json.load(f)
    engine = sa.create_engine(
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}/{params['dbname']}",
        # ONE connection per process, no overflow: a worker issues its queries
        # serially, so a second connection is never in use -- and with ~126
        # workers the pool size is multiplied by 126 against the server's
        # max_connections. pool_recycle drops connections the server may have
        # closed under a multi-hour run, without paying pool_pre_ping's extra
        # round trip on every checkout.
        pool_size=1, max_overflow=0, pool_recycle=1800,
    )
    _ENGINES[key] = engine
    return engine


def dispose_engines() -> None:
    """Close this process's pooled connections (call before forking workers)."""
    for key in [k for k in _ENGINES if k[0] == os.getpid()]:
        _ENGINES.pop(key).dispose()


def _py_oids(oids) -> list:
    """Cast to plain Python scalars — psycopg2 won't adapt numpy types."""
    return [o.item() if hasattr(o, "item") else o for o in oids]


# Storage variants seen across ZTF ingestions; bools hash to 1/0 and hit
# those entries. 'f'/0 means negative difference -> -1 (parser convention).
# multisurvey stores integer ±1 (recon 2026-06-05), so passthrough is the
# live branch; the rest are defensive.
_ISDIFFPOS_MAP = {1: 1, -1: -1, 0: -1, "t": 1, "f": -1, "1": 1, "-1": -1, "0": -1}


def normalize_isdiffpos(series: pd.Series) -> pd.Series:
    """Normalize isdiffpos to the +1/-1 ints the parser logic expects."""

    def _one(v):
        if pd.isna(v):
            return np.nan
        try:
            return _ISDIFFPOS_MAP[v]
        except (KeyError, TypeError):
            raise ValueError(f"Unrecognized isdiffpos value: {v!r}") from None

    return series.map(_one)


def _postprocess_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-read normalization for detections / forced photometry."""
    bad_bands = set(df["band"].dropna().unique()) - set(ZTF_BAND_MAP)
    if bad_bands:
        raise ValueError(
            f"DB returned band integer(s) not in ZTF_BAND_MAP={ZTF_BAND_MAP}: {bad_bands}"
        )
    df["isdiffpos"] = normalize_isdiffpos(df["isdiffpos"])
    return df


def fetch_detections(credentials_json: str, oids: list) -> pd.DataFrame:
    """Per-epoch ZTF detections: detection ⋈ ztf_detection on (oid, measurement_id).

    ORDER BY is load-bearing, not cosmetic: without it Postgres returns rows in
    plan order, which differs between a single-oid lookup and a 500-oid
    `ANY(...)` batch. Same rows, different order -- and the fitted features
    (SPM/Harmonics/period) are order-sensitive, so the feature values moved with
    the batch size. Pinning (oid, mjd, measurement_id) makes a read reproducible
    regardless of how many oids it asks for.

    Returns one row per detection with columns
    `oid, sid, measurement_id, mjd, ra, dec, band(int), mag, e_mag,
    mag_corr, e_mag_corr_ext, isdiffpos(±1), distnr, rb, rfid`.
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT
            d.oid                   AS oid,
            d.sid                   AS sid,
            d.measurement_id        AS measurement_id,
            d.mjd                   AS mjd,
            d.ra                    AS ra,
            d.dec                   AS dec,
            d.band                  AS band,
            z.magpsf                AS mag,
            z.sigmapsf              AS e_mag,
            z.magpsf_corr           AS mag_corr,
            z.sigmapsf_corr_ext     AS e_mag_corr_ext,
            z.isdiffpos             AS isdiffpos,
            z.distnr                AS distnr,
            z.rb                    AS rb,
            z.rfid                  AS rfid
        FROM {SCHEMA}.detection d
        JOIN {SCHEMA}.ztf_detection z
          ON d.oid = z.oid AND d.measurement_id = z.measurement_id
        WHERE d.oid = ANY(:oids) AND d.sid = :sid
        ORDER BY d.oid, d.mjd, d.measurement_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"oids": _py_oids(oids), "sid": SID})
    return _postprocess_epochs(df)


def fetch_forced_photometry(credentials_json: str, oids: list) -> pd.DataFrame:
    """Per-epoch ZTF forced photometry: forced_photometry ⋈ ztf_forced_photometry.

    Returns one row per epoch with columns
    `oid, sid, measurement_id, mjd, ra, dec, band(int), mag, e_mag,
    mag_corr, e_mag_corr_ext, isdiffpos(±1), procstatus, distnr, rfid,
    sharpnr, chinr`.
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT
            fp.oid                  AS oid,
            fp.sid                  AS sid,
            fp.measurement_id       AS measurement_id,
            fp.mjd                  AS mjd,
            fp.ra                   AS ra,
            fp.dec                  AS dec,
            fp.band                 AS band,
            z.mag                   AS mag,
            z.e_mag                 AS e_mag,
            z.mag_corr              AS mag_corr,
            z.e_mag_corr_ext        AS e_mag_corr_ext,
            z.isdiffpos             AS isdiffpos,
            z.procstatus            AS procstatus,
            z.distnr                AS distnr,
            z.rfid                  AS rfid,
            z.sharpnr               AS sharpnr,
            z.chinr                 AS chinr
        FROM {SCHEMA}.forced_photometry fp
        JOIN {SCHEMA}.ztf_forced_photometry z
          ON fp.oid = z.oid AND fp.measurement_id = z.measurement_id
        WHERE fp.oid = ANY(:oids) AND fp.sid = :sid
        ORDER BY fp.oid, fp.mjd, fp.measurement_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"oids": _py_oids(oids), "sid": SID})
    return _postprocess_epochs(df)


def fetch_ps1(credentials_json: str, oids: list) -> pd.DataFrame:
    """PS1 crossmatch metadata, one row per oid.

    `DISTINCT ON (oid) ... ORDER BY oid, measurement_id` keeps the earliest
    row — mirrors production, which takes these keys from the first
    detection that carries them.
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT DISTINCT ON (oid)
            oid, sgscore1, sgmag1, srmag1, simag1, szmag1, distpsnr1
        FROM {SCHEMA}.ztf_ps1
        WHERE oid = ANY(:oids)
        ORDER BY oid, measurement_id
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"oids": _py_oids(oids)})


def fetch_allwise(credentials_json: str, oids: list) -> pd.DataFrame:
    """AllWISE W1–W4 metadata via the xmatch table, one row per oid.

    `ORDER BY x.oid, x.dist` keeps the nearest match when an oid has
    several AllWISE crossmatches.
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT DISTINCT ON (x.oid)
            x.oid       AS oid,
            a.w1mpro    AS "W1",
            a.w2mpro    AS "W2",
            a.w3mpro    AS "W3",
            a.w4mpro    AS "W4"
        FROM {SCHEMA}.xmatch x
        JOIN {SCHEMA}.allwise a
          ON x.oid_catalog = a.oid_catalog
        WHERE x.oid = ANY(:oids) AND x.sid = :sid AND x.catid = :catid
        ORDER BY x.oid, x.dist
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn,
            params={"oids": _py_oids(oids), "sid": SID, "catid": ALLWISE_CATID},
        )


def fetch_references(credentials_json: str, oids: list) -> pd.DataFrame:
    """ZTF reference-image rows; chinr >= 0 mirrors production's validity filter."""
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT oid, rfid, sharpnr, chinr
        FROM {SCHEMA}.ztf_reference
        WHERE oid = ANY(:oids) AND chinr >= 0
        ORDER BY oid, rfid
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params={"oids": _py_oids(oids)})


def fetch_alerce_features(
    credentials_json: str,
    ztf_oid: str,
    version: str | None = None,
) -> pd.DataFrame:
    """Stored legacy ALeRCE features for one ZTF string oid, optionally one version.

    Queries alerce.feature WHERE oid = :oid (and version = :version when given).
    Returns columns [name, value, fid, version].

    Note: alerce.feature has no timestamp column; versions are identified by the
    'version' string (e.g. '26.0.0', '27.5.7a32.dev1'). Multiple versions can
    coexist for the same oid. Use list_alerce_feature_versions to discover them.
    """
    engine = _make_engine(credentials_json)
    if version is not None:
        query = text(f"""
            SELECT name, value, fid, version
            FROM {ALERCE_SCHEMA}.feature
            WHERE oid = :oid AND version = :version
        """)
        params: dict = {"oid": ztf_oid, "version": version}
    else:
        query = text(f"""
            SELECT name, value, fid, version
            FROM {ALERCE_SCHEMA}.feature
            WHERE oid = :oid
        """)
        params = {"oid": ztf_oid}
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def list_alerce_feature_versions(credentials_json: str, ztf_oid: str) -> list[str]:
    """Return the distinct feature versions stored in alerce.feature for one oid.

    The returned list is sorted lexicographically. Returns an empty list if the
    oid is not found. Useful for defaulting to the latest version when running
    offline_compare_vs_alerce.py.
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT DISTINCT version
        FROM {ALERCE_SCHEMA}.feature
        WHERE oid = :oid
        ORDER BY version
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"oid": ztf_oid})
    return df["version"].tolist()


def fetch_stored_probabilities(
    credentials_json: str,
    ztf_oid: str,
    classifier_names: list,
    version: str,
) -> pd.DataFrame:
    """Stored legacy probabilities for one ZTF string oid, given classifiers+version.

    Reads `alerce.probability` — the legacy ZTF probability table (string-keyed,
    LIST-partitioned by classifier_name; ~1.47B rows). The filter
    `classifier_name = ANY(:names)` prunes to just those partitions and the
    per-partition PK `(oid, classifier_name, classifier_version, class_name)`
    turns each into a cheap index lookup (no scan of the huge partitions).

    NOTE: `classifier_name` values are mixed-case (e.g.
    `lc_classifier_BHRF_forced_phot`) even though the child partition table names
    are lowercased by Postgres — pass the column values, not the table names.
    `classifier_version` here is a VARCHAR (e.g. '2.1.0'), unlike the smallint in
    the multisurvey scheme.

    Returns columns [classifier_name, class_name, probability, ranking].
    """
    engine = _make_engine(credentials_json)
    query = text(f"""
        SELECT classifier_name, class_name, probability, ranking
        FROM {ALERCE_SCHEMA}.probability
        WHERE oid = :oid
          AND classifier_version = :version
          AND classifier_name = ANY(:names)
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn,
            params={"oid": ztf_oid, "version": version, "names": list(classifier_names)},
        )


def fetch_taxonomy_maps(credentials_json: str, classifier_ids: list,
                        schema: Optional[str] = None) -> dict:
    """Return {classifier_id: {class_name: class_id}} from <schema>.taxonomy.

    The authoritative class_name -> class_id mapping for writing probabilities
    (mirrors production's get_taxonomy_by_classifier_id). Read-only. Ordered by
    "order" per classifier (cosmetic for the dict, matches production).
    """
    schema = schema or SCHEMA
    engine = _make_engine(credentials_json)
    # schema is trusted operator input (env / CLI), same f-string convention as the
    # other readers; classifier_ids are bound as an expanding parameter.
    sql = text(
        f'SELECT classifier_id, class_id, class_name FROM {schema}.taxonomy '
        'WHERE classifier_id IN :cids ORDER BY classifier_id, "order"'
    ).bindparams(sa.bindparam("cids", expanding=True))

    maps: dict = {}
    with engine.connect() as conn:
        for row in conn.execute(sql, {"cids": _py_oids(classifier_ids)}).mappings():
            maps.setdefault(int(row["classifier_id"]), {})[row["class_name"]] = int(row["class_id"])
    return maps
