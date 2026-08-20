"""Offline AllWISE crossmatch — mirror the live feature_step xmatch path.

The live `feature_step` (`features/step.py::pre_execute`) computes the AllWISE
crossmatch *itself* by calling the internal Xwave microservice through
`libs/xmatch_client` (`XmatchClient.conesearch_with_metadata`), then (a) attaches
the result to the message so `detections_to_astro_object` can read W1–W4 for the
features, and (b) produces the match link back to the scribe so it lands in
`multisurvey_ztf.xmatch`.

The offline tooling used to read the crossmatch straight from
`multisurvey_ztf.xmatch ⋈ allwise`, but those tables are EMPTY for ZTF (the
crossmatch is produced by the feature step, which never ran with USE_XMATCH for
ZTF). So — exactly like the step — the offline path must compute the crossmatch
against Xwave. This module wraps that call and converts its result into the same
`[oid, W1, W2, W3, W4]` frame `db.fetch_allwise` returns, so the downstream
`_xmatches`/`compute_astro_object` path is unchanged.

Persistence differs from the live step: we do NOT send to scribe. Instead we save
directly to the DB (see `persist_matches`, currently a placeholder).

Xwave URL: not committed to the repo (injected via secret at deploy time). Pass
it explicitly or via the `XMATCH_URL` env var.
"""
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

log = logging.getLogger(__name__)

# ZTF, per multisurvey_ztf.sid_lut; catid 0 = AllWISE per catalog_id_lut
# (matches features/offline/db.py). Kept here so persist_matches can shape rows.
SID_ZTF = 0
ALLWISE_CATID = 0
ALLWISE_CATALOG = "allwise"
# catalog name -> catid, mirroring scribe_multisurvey parse_xmatch exactly
# (allwise -> 0, anything else -> -999).
_CATALOG_TO_CATID = {ALLWISE_CATALOG: ALLWISE_CATID}
_UNKNOWN_CATID = -999

# Xwave conesearch defaults. NOTE: the offline radius is intentionally tighter
# than the live step / xmatch_client default (1.5"): 1.005" scopes the AllWISE
# cone to a ~1" match. (The live step passes no radius, so it uses the client's
# 1.5"; offline deviates here on purpose.)
DEFAULT_RADIUS = 1.005   # arcsec
DEFAULT_BATCH_SIZE = 500
# Internal Xwave crossmatch service. Used only as the CLI default
# (env-overridable via XMATCH_URL); the library `_resolve_url` stays strict.
DEFAULT_XMATCH_URL = "http://quimal-db1.alerce.online:8081"

# The parser (detections_to_astro_object) reads the WISE magnitudes from the
# match metadata as metadata["w{n}mpro"]["Float64"]; the Xwave metadata payload
# uses these keys. Map catalog metadata key -> our column name.
_WISE_META_KEYS = {"W1": "w1mpro", "W2": "w2mpro", "W3": "w3mpro", "W4": "w4mpro"}


def _resolve_url(base_url):
    url = base_url or os.getenv("XMATCH_URL")
    if not url:
        raise ValueError(
            "no Xwave xmatch URL: pass base_url or set the XMATCH_URL env var "
            "(the service URL is not committed to the repo)"
        )
    return url


def _make_client(base_url, batch_size=DEFAULT_BATCH_SIZE):
    # Imported lazily so importing this module never requires xmatch_client to be
    # installed/on-path (offline scripts add libs/xmatch_client to sys.path).
    from xmatch_client import XmatchClient

    return XmatchClient(base_url=_resolve_url(base_url), batch_size=batch_size)


def compute_matches(oids, ras, decs, base_url=None, radius=DEFAULT_RADIUS,
                    batch_size=DEFAULT_BATCH_SIZE, catalogs=(ALLWISE_CATALOG,),
                    client=None):
    """Cone-search Xwave for `oids` at (ras, decs); return MatchWithMetadata list.

    Mirrors `feature_step.step.get_xmatch_info`: one `conesearch_with_metadata`
    request **per catalog** (the client's `catalog` arg scopes the search on the
    server *before* the KNN, so each catalog yields its own nearest match rather
    than only the single global nearest across all catalogs). `catalogs=None`
    means one global-nearest call (`catalog=None`, the client default).

    `client` is injectable for tests (defaults to a real XmatchClient). `oids`
    are passed as strings, as the client expects."""
    if not (len(oids) == len(ras) == len(decs)):
        raise ValueError("oids, ras and decs must have the same length")
    if len(oids) == 0:
        return []
    client = client or _make_client(base_url, batch_size)
    str_oids = [str(o) for o in oids]
    ras, decs = list(ras), list(decs)
    cats = list(catalogs) if catalogs is not None else [None]
    matches = []
    for cat in cats:
        matches += client.conesearch_with_metadata(
            ras=ras, decs=decs, oids=str_oids, radius=radius, catalog=cat,
        )
    return matches


def _meta_float(match, meta_key):
    """Read match['metadata'][meta_key]['Float64'], tolerant of shape/absence."""
    meta = match.get("metadata") or {}
    cell = meta.get(meta_key)
    if cell is None:
        return np.nan
    if isinstance(cell, dict):
        cell = cell.get("Float64", cell.get("value"))
    return np.nan if cell is None else float(cell)


def matches_to_allwise_df(matches):
    """MatchWithMetadata list -> the `[oid, W1, W2, W3, W4]` frame the offline
    features path expects (same columns/shape as `db.fetch_allwise`).

    Keeps the nearest AllWISE match per oid (smallest `distance`), so it is a
    drop-in for `db.fetch_allwise`'s `DISTINCT ON (oid) ... ORDER BY dist`.
    Non-AllWISE matches are ignored."""
    cols = ["oid", "W1", "W2", "W3", "W4"]
    allwise = [m for m in matches if m.get("catalog") == ALLWISE_CATALOG]
    if not allwise:
        return pd.DataFrame(columns=cols)

    nearest = {}
    for m in allwise:
        oid = int(m["oid"])
        dist = m.get("distance")
        dist = np.inf if dist is None else float(dist)
        if oid not in nearest or dist < nearest[oid][0]:
            nearest[oid] = (dist, m)

    rows = []
    for oid, (_dist, m) in nearest.items():
        rows.append({
            "oid": oid,
            **{col: _meta_float(m, key) for col, key in _WISE_META_KEYS.items()},
        })
    return pd.DataFrame(rows, columns=cols)


def compute_allwise(oids, ras, decs, base_url=None, client=None, **kwargs):
    """Convenience: cone-search + reduce to the `db.fetch_allwise` frame."""
    matches = compute_matches(oids, ras, decs, base_url=base_url, client=client, **kwargs)
    return matches_to_allwise_df(matches)


def build_xmatch_rows(matches):
    """MatchWithMetadata list -> the exact `multisurvey_ztf.xmatch` rows the
    crossmatch link would persist.

    Faithful to the live write path: the step produces an xmatch scribe command
    (`step.produce_xmatch_to_scribe`) that `scribe_multisurvey`'s XmatchCommand
    turns into an upsert of the `Xmatch` model columns
    `(oid, sid, catid, oid_catalog, dist)` — the link table ONLY (the AllWISE
    catalog rows in `multisurvey_ztf.allwise` are loaded by a separate process,
    NOT by the feature step). `catid` follows scribe's `parse_xmatch`
    (allwise -> 0, else -999); `sid == 2` rows are dropped."""
    rows = []
    for m in matches:
        if SID_ZTF == 2:  # mirror parse_xmatch's sid==2 skip (never true for ZTF)
            continue
        catalog = m.get("catalog", ALLWISE_CATALOG)
        rows.append({
            "oid": int(m["oid"]),
            "sid": SID_ZTF,
            "catid": _CATALOG_TO_CATID.get(catalog, _UNKNOWN_CATID),
            "oid_catalog": m.get("match_id"),
            "dist": m.get("distance"),
        })
    return rows


def _xmatch_records(rows):
    """Sanitize build_xmatch_rows output into native-typed dict records.

    `multisurvey_ztf.xmatch.dist` and `.oid_catalog` are both NOT NULL, so drop
    (with a warning) any match missing a distance or a catalog id — it cannot be
    persisted. Cast to native Python types so the >2**53 oid bigint and the
    smallints are not coerced to float by the driver."""
    records = []
    dropped = 0
    for r in rows:
        if r.get("dist") is None or r.get("oid_catalog") is None:
            dropped += 1
            continue
        records.append({
            "oid": int(r["oid"]),
            "sid": int(r["sid"]),
            "catid": int(r["catid"]),
            "oid_catalog": str(r["oid_catalog"]),
            "dist": float(r["dist"]),
        })
    if dropped:
        log.warning("Dropping %d xmatch row(s) with NULL dist/oid_catalog "
                    "(both are NOT NULL in %s.xmatch)", dropped, "multisurvey_ztf")
    return records


def _db_allwise(credentials_json, oids):
    """Read AllWISE from <schema>.xmatch join <schema>.allwise.

    A thin indirection so `db` stays a lazy import here (importing this module
    must not require db) and so the DB read is injectable in tests."""
    from features.offline import db
    return db.fetch_allwise(credentials_json, oids)


def allwise_for_oid(oid, ra, dec, credentials_json, xmatch_url=None):
    """AllWISE colors for one oid -> (allwise_df, matches).

    With `xmatch_url`, cone-search Xwave live, exactly like the deployed step;
    `matches` is the raw crossmatch, so the caller can also persist it. Without
    it, read the precomputed `xmatch join allwise` and return no matches.

    The two are NOT interchangeable today: <schema>.allwise is empty for ZTF —
    the catalog rows are bulk-loaded by a separate process, and that load never
    ran for multisurvey_ztf — so the DB read yields no WISE at all and every
    WISE colour comes out NaN, which biases BHRF toward Stochastic
    (WISE_NULL_CLASSIFICATION_IMPACT.md). Pass a URL whenever the result matters.
    """
    if xmatch_url:
        matches = compute_matches([oid], [ra], [dec], base_url=xmatch_url)
        return matches_to_allwise_df(matches), matches
    return _db_allwise(credentials_json, [oid]), []


def persist_matches(matches, write_credentials=None, schema=None, execute=False):
    """Upsert the crossmatch link rows directly into <schema>.xmatch (no scribe).

    Sending the crossmatch to the DB is the feature step's responsibility: the
    live step produces it to the scribe, which upserts `multisurvey_ztf.xmatch`
    on conflict `(oid, sid, catid)`, updating `oid_catalog, dist, updated_date`.
    Offline we write the same rows straight into the table with write-capable
    creds — like `probability_writer.write_probabilities` — skipping the scribe.
    Link table only: the AllWISE catalog rows in `multisurvey_ztf.allwise` are
    loaded by a separate process, NOT here.

    Dry-run by default (execute=False): returns {"executed": False,
    "would_write": N} and opens no DB connection. With execute=True, upserts all
    records in one transaction and returns {"executed": True, "written": N}.
    `schema` defaults to db.SCHEMA."""
    from features.offline import db  # lazy: importing this module must not need db

    schema = schema or db.SCHEMA
    records = _xmatch_records(build_xmatch_rows(matches))
    n = len(records)
    log.info("persist_matches: %d %s.xmatch row(s) prepared (execute=%s)",
             n, schema, execute)

    if not execute:
        return {"executed": False, "would_write": n}

    # schema is a trusted operator-supplied identifier (db.SCHEMA env / CLI), not
    # user input — same f-string convention as db.py's read queries. Mirrors
    # scribe_multisurvey XmatchCommand.db_operation.
    sql = text(
        f"INSERT INTO {schema}.xmatch (oid, sid, catid, oid_catalog, dist, created_date) "
        "VALUES (:oid, :sid, :catid, :oid_catalog, :dist, now()) "
        "ON CONFLICT (oid, sid, catid) "
        "DO UPDATE SET oid_catalog = EXCLUDED.oid_catalog, dist = EXCLUDED.dist, "
        "updated_date = now()"
    )
    engine = db._make_engine(write_credentials)
    with engine.begin() as conn:
        if records:
            conn.execute(sql, records)
    return {"executed": True, "written": n}
