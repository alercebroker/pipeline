"""Backfill the four colour columns of <schema>.ztf_object from <schema>.feature.

The live feature step (features/utils/parsers.py::parse_scribe_payload) emits,
next to the feature upsert, an `update-ztf-object-features` command per object.
scribe_multisurvey turns it into

    UPDATE ztf_object SET g_r_max, g_r_mean, g_r_max_corr, g_r_mean_corr

taking each value from the band-12 (g,r pair) feature of the same name. The
offline run (BHRF_RUN_RESULTS.md) wrote `feature`, `probability` and `xmatch`
but never touched `ztf_object`, so those four columns are stale on every object
it processed. The values are already in `feature`; this module copies them
across, one UPDATE ... FROM per oid range.

WHY OID RANGES. `feature` is HASH partitioned on oid with PK
(oid, sid, feature_id, band) and no other index. A range on oid is a btree range
scan on every partition's PK that touches only the four colour entries per
object; anything keyed on feature_id alone would be a sequential scan of ~1.4B
rows. The ranges are slices of the run's own sorted oid array, so they follow
the real (very sparse) oid distribution rather than a uniform split of int64.

WHAT MIRRORS THE STEP. The step sends None for a colour missing from the
object's feature list. `feature` keeps a row from an older run when the latest
run did not recompute it (the PK has no run id -- see feature-rows-superseded),
so only rows carrying the object's max(updated_date) count; an older colour
becomes NULL instead of leaking into the object. Objects with no colour rows
are left untouched.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import text

log = logging.getLogger(__name__)

# ztf_object column -> feature_name_lut name. Same four the scribe writes.
COLOR_FEATURES = {
    "g_r_max": "g-r_max",
    "g_r_mean": "g-r_mean",
    "g_r_max_corr": "g-r_max_corr",
    "g_r_mean_corr": "g-r_mean_corr",
}
# band of a (g, r) pair feature, as written by feature_writer / the step.
PAIR_BAND = 12


def resolve_color_feature_ids(feature_name_lut: dict) -> dict:
    """{column: feature_id} for the four colours, from a {feature_id: name} LUT.

    The LUT must be the one read from the DB (db.fetch_feature_name_lut): the
    feature_lut.py fixture carries offline's own ids and `feature` has no
    foreign key that would catch a mismatch. Raises LookupError naming the
    first colour absent from the LUT.
    """
    by_name = {name: fid for fid, name in feature_name_lut.items()}
    ids = {}
    for column, name in COLOR_FEATURES.items():
        if name not in by_name:
            raise LookupError(f"feature_name_lut has no entry for {name!r}")
        ids[column] = int(by_name[name])
    return ids


def make_ranges(oids: np.ndarray, chunk_size: int) -> list:
    """[(lo, hi), ...] closed ranges: consecutive slices of the sorted unique oids."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    uniq = np.unique(np.asarray(oids, dtype=np.int64))
    return [(int(uniq[i]), int(uniq[min(i + chunk_size, len(uniq)) - 1]))
            for i in range(0, len(uniq), chunk_size)]


def build_backfill_sql(schema: str, ids: dict) -> str:
    """The UPDATE for one range. Bound params: :lo, :hi, :sid.

    schema is a trusted operator-supplied identifier, same f-string convention
    as db.py; the feature ids are ints resolved from the LUT.
    """
    id_list = ", ".join(str(i) for i in sorted(ids.values()))
    picks = ",\n               ".join(
        f"max(value) FILTER (WHERE feature_id = {ids[col]}) AS {col}"
        for col in COLOR_FEATURES)
    sets = ",\n            ".join(f"{col} = c.{col}" for col in COLOR_FEATURES)
    return f"""
        WITH rows AS (
            SELECT oid, feature_id, value,
                   updated_date = max(updated_date) OVER (PARTITION BY oid) AS current
            FROM {schema}.feature
            WHERE oid BETWEEN :lo AND :hi
              AND sid = :sid
              AND band = {PAIR_BAND}
              AND feature_id IN ({id_list})
        ),
        c AS (
            SELECT oid,
               {picks}
            FROM rows
            WHERE current
            GROUP BY oid
        )
        UPDATE {schema}.ztf_object AS o
        SET {sets}
        FROM c
        WHERE o.oid = c.oid
          AND o.oid BETWEEN :lo AND :hi
    """


def _read_progress(progress_path: Path) -> dict:
    """{chunk: record} of ranges already applied."""
    done = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[int(rec["chunk"])] = rec
    return done


def run_backfill(conn, sql: str, ranges: list, sid: int, progress_path,
                 execute: bool = False, max_chunks: Optional[int] = None) -> dict:
    """Apply `sql` to each range in order, one transaction per range.

    Dry-run by default: nothing is executed and the progress file is not
    created. With execute=True every finished range is appended to
    `progress_path` (jsonl) and a rerun with the same ranges skips them; a
    record whose (lo, hi) does not match the range at that index means a
    different oid array and is refused. `max_chunks` caps how many NEW ranges
    this call applies, for a first small pass.
    """
    progress_path = Path(progress_path)
    done = _read_progress(progress_path) if execute else {}
    for chunk, rec in done.items():
        if chunk >= len(ranges) or (rec["lo"], rec["hi"]) != ranges[chunk]:
            raise ValueError(
                f"{progress_path}: chunk {chunk} covers ({rec['lo']}, {rec['hi']}), "
                f"which is not range {chunk} of this oid array -- different plan, "
                "use a fresh --out-dir")

    summary = {"executed": execute, "ranges": len(ranges), "skipped": len(done),
               "updated": 0}
    if not execute:
        return summary

    applied = 0
    for chunk, (lo, hi) in enumerate(ranges):
        if chunk in done:
            continue
        if max_chunks is not None and applied >= max_chunks:
            break
        t0 = time.monotonic()
        result = conn.execute(text(sql), {"lo": lo, "hi": hi, "sid": sid})
        conn.commit()
        seconds = time.monotonic() - t0
        updated = int(result.rowcount)
        rec = {"chunk": chunk, "lo": lo, "hi": hi, "updated": updated,
               "seconds": round(seconds, 3)}
        with progress_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        summary["updated"] += updated
        applied += 1
        log.info("chunk %d/%d [%d, %d]: %d object(s) updated in %.1fs",
                 chunk + 1, len(ranges), lo, hi, updated, seconds)
    return summary
