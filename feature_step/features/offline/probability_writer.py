"""Persist offline BHRF probabilities into <schema>.probability.

Mirrors feature_writer.py (writing lives here, not in db.py). Turns the BHRF
OutputDTO (flat + hierarchical frames) into DB-ready rows across all 5 seeded
classifiers (ids 5-9). class_name -> class_id uses a {classifier_id: {class_name:
class_id}} map READ FROM THE DB taxonomy table (the authority for the
probability.class_id FK; see db.fetch_taxonomy_maps), passed in so this stays a
pure function. Upsert is ON CONFLICT (oid, sid, classifier_id, class_id) DO UPDATE.
Dry-run unless execute=True.
"""
import logging
from typing import Optional

from psycopg2.extras import execute_values

from features.offline import db
from features.offline.classifier_taxonomy_lut import CLASSIFIER_VERSION

log = logging.getLogger(__name__)

# Rows per INSERT statement; see feature_writer.PAGE_SIZE.
PAGE_SIZE = 1000


def _assert_no_duplicate_keys(records, key_fields, table) -> None:
    """Refuse a batch that carries the same primary key twice.

    execute_values puts many rows in ONE statement, and Postgres rejects an
    ON CONFLICT DO UPDATE that would touch the same row twice ("cannot affect
    row a second time"). Under the old per-row executemany this was impossible,
    so the batching introduces the failure mode -- and it always means the
    caller assembled the rows wrong (the same oid twice in a unit), which is
    worth naming rather than passing to the driver.
    """
    seen = set()
    for r in records:
        key = tuple(r[f] for f in key_fields)
        if key in seen:
            raise ValueError(
                f"duplicate {table} key {dict(zip(key_fields, key))} in one write batch"
            )
        seen.add(key)

# The 5 seeded BHRF classifiers, in id order (flat, top, transient, stochastic, periodic).
CLASSIFIER_IDS = [5, 6, 7, 8, 9]


def classifier_version_to_smallint(version: str) -> int:
    """'2.1.0' -> 210 (production rule). Strips a '_suffix' on the patch part.

    Raises ValueError on anything else. Returning a 0 sentinel here is not safe:
    <schema>.probability has no FK or CHECK on classifier_version, so the row is
    accepted and the whole run is stamped with a version nothing can resolve.
    """
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"classifier version {version!r} is not MAJOR.MINOR.PATCH — refusing to "
            "stamp probabilities with an unresolvable version"
        )
    parts[-1] = parts[-1].split("_")[0]
    try:
        return int("".join(parts))
    except ValueError:
        raise ValueError(
            f"classifier version {version!r} is not MAJOR.MINOR.PATCH — refusing to "
            "stamp probabilities with an unresolvable version"
        ) from None


def _iter_frames(output_dto):
    """Return (classifier_id, frame) for the 5 BHRF classifiers, in id order."""
    hierarchical = output_dto.hierarchical or {}
    children = hierarchical.get("children", {}) or {}
    return [
        (5, output_dto.probabilities),
        (6, hierarchical.get("top")),
        (7, children.get("Transient")),
        (8, children.get("Stochastic")),
        (9, children.get("Periodic")),
    ]


def build_probability_rows(output_dto, oid: int, lastmjd: float,
                           taxonomy_by_classifier: dict, *,
                           version: str = CLASSIFIER_VERSION, sid: int = 0) -> list:
    """BHRF OutputDTO (single oid) -> DB-ready probability row dicts (all 5 classifiers).

    taxonomy_by_classifier: {classifier_id: {class_name: class_id}} from the DB
    (db.fetch_taxonomy_maps). A class name not in its classifier's map, or a
    classifier with no map at all, raises (mirrors the -1 miss that would store
    garbage). Returns [] for an empty/can't-predict OutputDTO. ranking =
    per-classifier dense rank descending.
    """
    if output_dto is None or output_dto.probabilities is None or len(output_dto.probabilities) == 0:
        return []
    if lastmjd is None:
        raise ValueError("lastmjd is required (probability.lastmjd is NOT NULL)")

    version_smallint = classifier_version_to_smallint(version)
    rows = []
    for classifier_id, frame in _iter_frames(output_dto):
        if frame is None or len(frame) == 0:
            continue
        if len(frame) != 1:
            raise ValueError(
                f"expected a single-oid frame for classifier_id={classifier_id}, "
                f"got {len(frame)} rows (build_probability_rows is per-oid)"
            )
        class_id_of = taxonomy_by_classifier.get(classifier_id)
        if not class_id_of:
            raise ValueError(
                f"no taxonomy map for classifier_id={classifier_id} "
                "(fetch_taxonomy_maps returned nothing for it — is the taxonomy seeded?)"
            )
        series = frame.iloc[0]  # single oid -> Series: class_name -> probability
        ranks = series.rank(ascending=False, method="dense").astype(int)
        for class_name in series.index:
            if class_name not in class_id_of:
                raise ValueError(
                    f"class '{class_name}' not in taxonomy for classifier_id={classifier_id} "
                    "(model/taxonomy mismatch — cannot map to class_id)"
                )
            rows.append({
                "oid": int(oid),
                "sid": int(sid),
                "classifier_id": int(classifier_id),
                "classifier_version": int(version_smallint),
                "class_id": int(class_id_of[class_name]),
                "probability": float(series[class_name]),
                "ranking": int(ranks[class_name]),
                "lastmjd": float(lastmjd),
            })
    return rows


def write_probabilities(rows: list, credentials: str, schema: Optional[str] = None,
                        execute: bool = False) -> dict:
    """Upsert DB-ready probability rows into <schema>.probability.

    Dry-run by default (execute=False): returns {"executed": False,
    "would_write": N} and opens no DB connection. With execute=True, upserts all
    rows in one transaction. schema defaults to db.SCHEMA.
    """
    schema = schema or db.SCHEMA
    n = len(rows)
    _assert_no_duplicate_keys(
        rows, ("oid", "sid", "classifier_id", "class_id"), "probability")
    if not execute:
        return {"executed": False, "would_write": n}

    # schema is a trusted operator-supplied identifier (db.SCHEMA env / CLI), not
    # user input — same f-string convention as db.py / feature_writer.py.
    sql = (
        f"INSERT INTO {schema}.probability "
        "(oid, sid, classifier_id, classifier_version, class_id, probability, ranking, lastmjd) "
        "VALUES %s "
        "ON CONFLICT (oid, sid, classifier_id, class_id) "
        "DO UPDATE SET probability = EXCLUDED.probability, "
        "classifier_version = EXCLUDED.classifier_version, "
        "ranking = EXCLUDED.ranking, lastmjd = EXCLUDED.lastmjd"
    )
    rows_out = [(r["oid"], r["sid"], r["classifier_id"], r["classifier_version"],
                 r["class_id"], r["probability"], r["ranking"], r["lastmjd"])
                for r in rows]

    engine = db._make_engine(credentials)
    raw = engine.raw_connection()
    try:
        if rows_out:
            with raw.cursor() as cur:
                execute_values(cur, sql, rows_out, page_size=PAGE_SIZE)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return {"executed": True, "written": n}
