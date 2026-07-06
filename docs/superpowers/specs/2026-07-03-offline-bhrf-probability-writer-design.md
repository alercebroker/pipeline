# Persist offline BHRF probabilities to `multisurvey_ztf.probability`

**Date:** 2026-07-03
**Status:** Design notes — grounded in the production write path, pending an
implementation plan. Depends on the now-applied BHRF `classifier` + `taxonomy`
seed (ids 5–9).
**Related:**
- `docs/superpowers/specs/2026-07-02-ztf-bhrf-classifier-taxonomy-seed-design.md` (the seed, done + applied)
- `feature_step/features/offline/FLOW.md` §3c, §3d, §7 item 4
- Production reference: `stamp_classifier_2025_multisurvey_step/stamp_classifier_2025_multisurvey_step/{step.py,db/db.py}`

## Problem

The offline classify path (`classify.py`) stops at the in-memory `OutputDTO` —
it prints probabilities but never writes them. To persist them we need a
`probability_writer.py` analogous to `feature_writer.py` (§5 of FLOW.md): DB-ready
rows, dry-run by default, `--execute` to write, wired into
`offline_classify.py --save`.

The BHRF `classifier` (ids 5–9) and `taxonomy` (45 rows, `SESN`) LUTs are now
**seeded and committed to live** (2026-07-02/03), which unblocks this.

## Correction to FLOW.md §7 item 4 (important)

FLOW.md §7 assumed the name→id / smallint-version / ranking logic lived in a
**downstream scribe consumer** we'd have to locate. That is **not** how the
multisurvey path works. `stamp_classifier_2025_multisurvey_step` **writes to
`probability` directly** via SQLAlchemy (`db.py::store_probability`), no scribe in
the loop. So the offline writer should mirror that direct-write step exactly —
which is also what `feature_writer.py` already does. **The "locate the consumer"
task is resolved: the consumer is `store_probability` / `format_probability_records`
in that step.**

## Source of the probabilities: the BHRF `OutputDTO` → 5 classifier_ids

BHRF is hierarchical, so one `predict()` yields **five** probability frames, one
per seeded classifier. From `alerce_classifiers/squidward/{model.py,mapper.py}` and
`base/dto.py`:

```python
out = model.predict(input_dto)            # SquidwardFeaturesClassifier.predict
out.probabilities                         # flat 21-class frame  (oid × class cols)
out.hierarchical["top"]                   # 3-class  (Periodic/Stochastic/Transient)
out.hierarchical["children"]["Transient"] # 6-class
out.hierarchical["children"]["Stochastic"]# 6-class
out.hierarchical["children"]["Periodic"]  # 9-class
```

Mapping each frame to the seeded `classifier_id`:

| classifier_id | classifier_name | source frame |
|---|---|---|
| 5 | `lc_classifier_BHRF_forced_phot` (flat) | `out.probabilities` |
| 6 | `lc_classifier_BHRF_forced_phot_top` | `out.hierarchical["top"]` |
| 7 | `lc_classifier_BHRF_forced_phot_transient` | `out.hierarchical["children"]["Transient"]` |
| 8 | `lc_classifier_BHRF_forced_phot_stochastic` | `out.hierarchical["children"]["Stochastic"]` |
| 9 | `lc_classifier_BHRF_forced_phot_periodic` | `out.hierarchical["children"]["Periodic"]` |

The column names of each frame are the model's own class labels — they must match
the seeded `taxonomy.class_name` exactly (that's the whole point of the `SESN` lock
and the `offline_verify_taxonomy.py` guard).

## The `probability` table (`models_pipeline.py:968`)

- Columns: `oid (bigint)`, `sid (smallint)`, `classifier_id (smallint)`,
  `classifier_version (smallint)`, `class_id (smallint)`, `probability (real)`,
  `ranking (smallint, nullable)`, `lastmjd (double, NOT NULL)`.
- **PK / conflict target = `(oid, sid, classifier_id, class_id)`**
  (`pk_probability_oid_classifierid_classid`). Note `classifier_version` and
  `probability` are **not** in the PK.
- **Hash-partitioned on `oid`** into 16 partitions. Always filter/insert by oid;
  never full-scan (FLOW.md §3c).

## Per-row rules (mirror `format_probability_records`, with two offline deltas)

For each of the 5 (classifier_id, frame) pairs:

1. `melt` the frame `id_vars=["oid"]` → `(oid, class_name, probability)`.
2. `ranking` = per-**oid** dense rank of `probability` descending, as int
   (`groupby("oid")["probability"].rank(ascending=False, method="dense").astype(int)`).
   Computed **within each classifier's frame** (rank 1 = argmax of that classifier).
3. `sid = 0` (ZTF).
4. `classifier_id` = the id for this frame (5–9).
5. `classifier_version` = **smallint** via `classifier_version_str_to_small_integer`
   — `"2.1.0" → 210` (strip a `_suffix` on the patch, join the 3 parts). **Lock
   this convention** (this was the "decide when the writer is built" open point).
6. `class_id` = `class_name → class_id` via the **seeded taxonomy for that
   classifier_id**, fetched with `get_taxonomy_by_classifier_id` = `{class_name:
   class_id}` (`SELECT class_id, class_name FROM taxonomy WHERE classifier_id=:id
   ORDER BY "order"`). Exact string match; **`-1` on miss** — a miss means garbage,
   so the writer should treat any `class_id == -1` as a hard error, not persist it.
7. `lastmjd` — **OFFLINE DELTA.** Production stamp uses `msg["jd"] - 2400000.5`
   (JD→MJD of the single alert). The offline pipeline is **already in MJD** (`db.py`
   reads `mjd` directly), and BHRF is a light-curve classifier, so:
   `lastmjd = max(mjd)` over the object's **real** detections. **Do NOT subtract
   2400000.5** — the value is already MJD; subtracting would produce a nonsense
   negative epoch.
8. drop `class_name`; the row is `(oid, sid, classifier_id, classifier_version,
   class_id, probability, ranking, lastmjd)`.

### Second offline delta — the taxonomy source

Production reads the taxonomy once from the DB at step startup. Offline can do the
same (read the seeded rows), OR reuse the local fixture `classifier_taxonomy_lut.py`
directly (`{name: enumerate index}`), avoiding a DB round-trip. **Recommendation:**
read from the **DB** (`get_taxonomy_by_classifier_id`) so the writer persists exactly
what the live catalog will map by — the fixture is the seed *source*, the DB is the
*authority* at write time. Guard: if the fetched mapping is empty (taxonomy not
seeded in the target schema), fail loudly.

## Deliverable (proposed)

1. **`feature_step/features/offline/probability_writer.py`** — mirrors
   `feature_writer.py`:
   - `build_probability_rows(output_dto, oid, lastmjd, taxonomy_by_classifier, *, version="2.1.0", sid=0)`
     → list of DB-ready dicts across all 5 classifier_ids (pure, unit-testable, no DB).
   - `write_probabilities(rows, credentials, schema=db.SCHEMA, execute=False)` →
     upsert into `<schema>.probability`; **dry-run by default**, opens no connection
     unless `execute=True`.
   - **Conflict policy — decision needed (see below).**
2. Wire `offline_classify.py --save [--execute] [--write-credentials]` (same flag
   shape as `offline_compute_features.py --save`).
3. Docs: FLOW.md §3c/§7 item 4 → Done; README file map.
4. Tests: pure `build_probability_rows` (5 classifiers, ranking, version 210,
   class_id mapping incl. a `-1`-raises case, lastmjd is max-mjd-not-JD).

## Open decisions

1. **Conflict policy: `DO UPDATE` vs `DO NOTHING`.** Production stamp uses
   `on_conflict_do_nothing()`. `feature_writer.py` uses `DO UPDATE` (refresh on
   re-run). Since PK excludes `classifier_version` and `probability`, `DO NOTHING`
   would make a re-run with a *new* model version silently keep stale probabilities.
   **Recommendation: `ON CONFLICT (oid, sid, classifier_id, class_id) DO UPDATE`**
   (refresh `probability, classifier_version, ranking, lastmjd`) — matches
   `feature_writer` and offline's reproducibility goal. Flag if the team wants to
   mirror production's DO NOTHING instead.
2. **Write all 5 classifiers, or only the flat (id 5)?** Recommendation: **all 5**
   — we seeded all 5, and the hierarchical rows are what make the catalog useful.
3. **`classifier_version` smallint for future non-`2.1.0` models** — the
   `str_to_small_integer` rule (`210`) is adopted; revisit only if a version needs
   >3 parts or a non-numeric tag.

## Out of scope
- Backfilling probabilities for many oids (this is the single-oid writer; batch is
  a later loop, same as features).
- Any change to the live scribe / streaming step.
