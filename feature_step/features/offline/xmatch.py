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

# Xwave conesearch defaults (mirror the step / xmatch_client defaults).
DEFAULT_RADIUS = 1.5   # arcsec
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

    Mirrors `feature_step.step.get_xmatch_info` -> `conesearch_with_metadata`.
    `client` is injectable for tests (defaults to a real XmatchClient). `oids`
    are passed as strings, as the client expects."""
    if not (len(oids) == len(ras) == len(decs)):
        raise ValueError("oids, ras and decs must have the same length")
    if len(oids) == 0:
        return []
    client = client or _make_client(base_url, batch_size)
    return client.conesearch_with_metadata(
        ras=list(ras), decs=list(decs), oids=[str(o) for o in oids],
        radius=radius, catalogs=list(catalogs) if catalogs is not None else None,
    )


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


def persist_matches(matches, write_credentials=None, execute=False):
    """Save the crossmatch directly to the DB — PLACEHOLDER (does not scribe).

    Sending the crossmatch to the DB is the feature step's responsibility: the
    live step produces it to the scribe, which upserts `multisurvey_ztf.xmatch`
    on conflict `(oid, sid, catid)`, updating `oid_catalog, dist, updated_date`.
    Offline we intend to write the same rows straight into the table (write-
    capable creds, like `probability_writer.write_probabilities`), skipping the
    scribe.

    The direct-write SQL is intentionally not implemented yet: this builds and
    returns the exact `Xmatch` rows that would be written (see
    `build_xmatch_rows`) and, on `execute`, raises NotImplementedError so nothing
    is silently dropped."""
    rows = build_xmatch_rows(matches)
    log.info("persist_matches: %d %s.xmatch row(s) prepared (execute=%s)",
             len(rows), "multisurvey_ztf", execute)
    if not execute:
        return rows
    # TODO(placeholder): upsert `rows` into multisurvey_ztf.xmatch using
    # write_credentials — INSERT ... ON CONFLICT (oid, sid, catid) DO UPDATE SET
    # oid_catalog=EXCLUDED.oid_catalog, dist=EXCLUDED.dist, updated_date=now().
    # Mirrors scribe_multisurvey XmatchCommand.db_operation. Link table only; the
    # AllWISE catalog rows are loaded separately, not here.
    raise NotImplementedError(
        "persist_matches: direct DB write is a placeholder — not yet implemented"
    )
