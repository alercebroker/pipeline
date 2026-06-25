# Offline ZTF feature writer — persist DB-ready rows to `multisurvey_ztf.feature`

**Date:** 2026-06-25
**Status:** design, approved
**Related:** `docs/superpowers/specs/2026-06-25-offline-ztf-db-ready-features-design.md`
(the upstream piece that produces the rows), `feature_step/features/offline/FLOW.md`.

---

## 1. Problem

`compute_db_features` produces DB-ready rows
(`[oid, sid, feature_id, band, version, value]`) but nothing persists them — the
offline tooling stops before the INSERT. With the ZTF `feature_name_lut` /
`feature_version_lut` now seeded, we add the writer so an offline run can save
features into `multisurvey_ztf.feature` exactly as production would.

## 2. Target table

`Feature` (`libs/db-plugins-multisurvey/.../models_pipeline.py:1007`):
`(oid bigint, sid smallint, feature_id smallint, band smallint, version smallint,
value double, updated_date date)`. PK `(oid, sid, feature_id, band)`,
**HASH-partitioned on `oid`**, **no foreign keys**. `updated_date` has
`server_default now()` and ORM `onupdate now()` (the latter does NOT fire for raw
SQL).

## 3. Decisions (approved)

- **Upsert semantics:** per-row `ON CONFLICT (oid, sid, feature_id, band) DO UPDATE`
  (refresh `value`, `version`, `updated_date`). Matches production's scribe upsert.
  Known caveat: a feature that becomes NaN/absent on recompute leaves its prior row
  behind (stale) — accepted, same as production.
- **Interface:** a `--save` flag on the existing `offline_compute_features.py`
  (single oid). The writer itself is a new module taking a DataFrame, so batch can
  be added later without touching it.
- **Safety:** dry-run by default; `--execute` required to actually write.
- Writing lives **outside `db.py`** (which stays read-only) in a new
  `feature_writer.py`.

## 4. Design

### 4a. `feature_step/features/offline/feature_writer.py` (new)

```python
write_features(rows, credentials, schema=db.SCHEMA, execute=False) -> dict
```

1. **Sanitize** `rows` (the `compute_db_features` frame):
   - Drop rows with NaN `feature_id` (can't satisfy NOT-NULL PK); `log.warning` the
     dropped feature count if any.
   - `log.warning` if any `version == -1` (version_name absent from the LUT).
   - Cast to native Python types per record: `int(oid)` (preserves the >2^53
     bigint), `int` for `sid/feature_id/band/version`, and `value` → `None` when
     NaN/None else `float`.
2. **Dry-run** (`execute=False`): return `{"executed": False, "would_write": N}`.
   Must NOT open a DB connection.
3. **Execute** (`execute=True`): `engine = db._make_engine(credentials)`; inside
   `with engine.begin() as conn:` run ONE parameterized statement over
   `records` (list of dicts):
   ```sql
   INSERT INTO {schema}.feature (oid, sid, feature_id, band, version, value)
   VALUES (:oid, :sid, :feature_id, :band, :version, :value)
   ON CONFLICT (oid, sid, feature_id, band)
   DO UPDATE SET value = EXCLUDED.value, version = EXCLUDED.version, updated_date = now()
   ```
   Return `{"executed": True, "written": N}`.

`N` is the post-sanitize row count. The transaction is all-or-nothing per call.

### 4b. CLI wiring (`feature_step/scripts/offline_compute_features.py`)

Add args (compute path unchanged):
- `--save` — after computing the DB-ready frame, pass it to `write_features`.
- `--execute` — actually write (only meaningful with `--save`). Without it: dry-run.
- `--write-credentials PATH` — credentials used for writing; **required when
  `--execute`** (default `credentials.json` is `readonly_user`). Error clearly if
  `--execute` is given without it.

Behavior:
- no `--save` → unchanged (print the frame).
- `--save` without `--execute` → compute, then `write_features(..., execute=False)`,
  print `would_write` count.
- `--save --execute --write-credentials P` → write, print `written` count.

## 5. Error handling

- No `--execute` ⇒ zero DB connection (safe by construction).
- NaN `feature_id` rows dropped + warned.
- `version == -1` warned.
- One transaction per oid (all-or-nothing).
- `--execute` without `--write-credentials` ⇒ argparse-level error, no compute.

## 6. Testing (`feature_step/tests/unittest/test_offline_feature_writer.py`, no real DB)

- **Dry-run:** `write_features(df, "ignored", execute=False)` → `{"executed": False,
  "would_write": N}`; monkeypatch `db._make_engine` to raise → proves no connection.
- **Execute:** monkeypatch `db._make_engine` to a fake engine whose `begin()` yields
  a recording connection. Assert: SQL contains `ON CONFLICT (oid, sid, feature_id,
  band) DO UPDATE`; the passed records carry **native `int`** (a big oid like
  `36028941624528297` is intact, not float); a `None` value passes through;
  `{"executed": True, "written": N}`.
- **Sanitize:** a frame with one NaN `feature_id` row → that row dropped, warning
  emitted, remaining rows written.

## 7. Out of scope

- Batch / multi-oid CLI (writer takes a DataFrame, so it's a later CLI-only add).
- A live-DB integration test (verified manually via `--execute`).
- Removing stale rows (now-absent features) — accepted divergence, matches production.
- Probability persistence (separate milestone).

## 8. File map

| File | Change |
|---|---|
| `feature_step/features/offline/feature_writer.py` | **new** — `write_features`. |
| `feature_step/tests/unittest/test_offline_feature_writer.py` | **new** — unit tests. |
| `feature_step/scripts/offline_compute_features.py` | add `--save` / `--execute` / `--write-credentials`. |
| `feature_step/features/offline/FLOW.md`, `README.md` | document the writer. |
