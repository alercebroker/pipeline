# Server runbook — offline ZTF features + BHRF classification at scale

Everything that has to happen on a **freshly cloned repo** to get from nothing to
a full offline run over `multisurvey_ztf`. For what the pipeline *does*, read
[`FLOW.md`](./FLOW.md); this file is only the operational sequence.

**What the run produces:** parquet shards on local disk — probability rows always,
feature rows with `--features`. With `--load-db` it *also* upserts each finished
unit into `<schema>.probability`, `<schema>.xmatch` and (with `--features`)
`<schema>.feature`, one batched statement per table per unit. The shards stay
the primary output either way; the database load is opt-in so a probe run cannot
touch production.

`xmatch` is in that list because writing the crossmatch link is the *feature
step's* job in the live pipeline; this run replaces that step. It is skipped
when `--xmatch-url` is empty, since then the AllWISE frames were read from the
DB and there is nothing new to record.

**Do not change `--load-db` within one `--out-dir`.** Resume skips a unit when
its manifest exists and never looks at the database, so units finished while the
flag was off stay on disk and are never loaded — and no loader exists to pick
them up later (§11). Switching between *different* `--out-dir`s is fine and is
what the probes do: a fresh directory has no manifests, so the real run simply
redoes those oids.

**Shortcut:** `scripts/offline_setup.py` checks everything §3–§7 asks for and
does the two expensive parts itself — downloading the model with its md5 (§5)
and materializing the oid list (§8 step 1) — skipping whatever is already in
place. It does not install anything (§1–§3 stay manual), cannot write the
credentials files (they carry live passwords) and cannot issue the grants (§4).
Read the sections below for what the steps mean; run the script to do them.

```bash
python scripts/offline_setup.py --check-only   # what is missing, changes nothing
python scripts/offline_setup.py                # ...and fix what it can
```

Exit code is 0 only when the machine can start a run.

---

## 1. Clone

```bash
git clone --recurse-submodules git@github.com:alercebroker/pipeline.git
cd pipeline
```

`--recurse-submodules` is not optional: `alerce_classifiers/` is a submodule, and
without it the BHRF model cannot be loaded at all. On an existing clone:

```bash
git submodule update --init --recursive
```

## 2. System packages

```bash
sudo apt-get update && sudo apt-get install -y git build-essential
```

`build-essential` is required: `P4J` and `mhps` are Cython/C extensions compiled
from source. On **Linux x86-64 nothing from `LOCAL_DEV_NOTES.md` applies** — that
document describes arm64/macOS workarounds (the `-march=x86-64-v3` flag, the
`fastavro` wheel). On the target platform the original code builds as-is.

Python **3.10** specifically — `feature_step/pyproject.toml` pins
`python = ">=3.10,<3.11"`.

## 3. Install

Order matters; this mirrors `feature_step/Dockerfile`, which is the authority.

```bash
cd feature_step
pip install poetry

# The C extensions need these present BEFORE they build. Cython is pinned:
# modern Cython fails on P4J/mhps.
poetry run python -m pip install setuptools wheel Cython==0.29.36 numpy
poetry run python -m pip install ../mhps
poetry run python -m pip install -r ../P4J/requirements.txt

poetry install --without=test --no-root   # add --with=test to run the suite
```

Verify the interpreter resolves the offline package:

```bash
poetry run python -c "from features.offline import db; print(db.SCHEMA)"
# -> multisurvey_ztf
```

## 4. Database credentials

`features/offline/credentials.json` is **gitignored** and must be created by hand:

```json
{"user": "...", "password": "...", "host": "quimal-db1.alerce.online",
 "port": 5432, "dbname": "ztf"}
```

Without `--load-db` the run only **reads**, so a read-only user is enough. With
`--load-db` you need a second file for `--write-credentials`, and that user must
hold `INSERT`/`UPDATE` on `multisurvey_ztf.feature`, `.probability` and `.xmatch`.

`write_user` was granted these on 2026-08-20 (as `postgres` on quimal-db1):

```sql
GRANT USAGE ON SCHEMA multisurvey_ztf TO write_user;
GRANT SELECT, INSERT, UPDATE ON multisurvey_ztf.feature     TO write_user;
GRANT SELECT, INSERT, UPDATE ON multisurvey_ztf.probability TO write_user;
GRANT SELECT, INSERT, UPDATE ON multisurvey_ztf.xmatch      TO write_user;
```

The `USAGE` line is the one that is easy to miss: `write_user` inherits table
grants on the `alerce` schema through `write_role` but has no `USAGE` on that
schema, which makes them unusable — so "it already writes to `alerce`" was never
true. Grants on the 32 + 16 partitions are deliberately NOT given: an INSERT
routed through the partitioned parent is checked against the parent only.

Verify with:

```sql
SELECT has_schema_privilege('write_user','multisurvey_ztf','USAGE'),
       has_table_privilege ('write_user','multisurvey_ztf.feature','INSERT'),
       has_table_privilege ('write_user','multisurvey_ztf.probability','INSERT'),
       has_table_privilege ('write_user','multisurvey_ztf.xmatch','INSERT');
```

The runner opens the write connection in the parent and fails there, before
forking workers, so a wrong user costs seconds rather than a unit.

Check connectivity and that the LUTs the run depends on are present:

```bash
poetry run python -c "
from features.offline import db
C='features/offline/credentials.json'
print('feature names :', len(db.fetch_feature_name_lut(C)))          # 127
print('version 27.5.7a31 ->', db.fetch_feature_version_id(C,'27.5.7a31'))  # 1
print('taxonomy heads:', sorted(db.fetch_taxonomy_maps(C,[5,6,7,8,9])))    # [5..9]
"
```

If any of those is empty you are pointing at a DB that was never seeded — apply
`ztf_feature_luts_seed.sql` and `ztf_classifier_taxonomy_seed.sql` first. They are
idempotent. Note there are **no FK constraints** on `feature`/`probability`, so a
missing LUT does not raise on write; it silently produces rows that resolve to
nothing. That is why the runner resolves these ids up front and refuses to start
without them.

## 5. The model

Download once, verify the md5, point `MODEL_PATH` at it. The location does not
matter:

```bash
mkdir -p /data/models
curl -o /data/models/hierarchical_random_forest_model.pkl \
  https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl

md5sum /data/models/hierarchical_random_forest_model.pkl
# must be 95e8e9f18fde62f22025e31a88ad81fa  (1,720,755,396 bytes)

export MODEL_PATH=/data/models/hierarchical_random_forest_model.pkl
```

**Use a local file, not the URL.** A URL `MODEL_PATH` is downloaded into
`/tmp/SquidwardFeaturesClassifier/` and whatever already sits there is reused —
that cache has held a stale SNIbc pickle. On a pristine server the URL is safe,
but a verified local file removes the question permanently.

`alerce_classifiers` derives the model version by scanning the path for a
version-shaped component, so a local file reports `"no_version"`.
`classify.resolve_model_version` handles that: it stamps the pinned
`MODEL_VERSION` and **refuses to run** if the path claims a version that is not
2.1.0 — the feature list, the seeded taxonomy and `classifier_version` all
assume it. No directory naming convention is required.

## 6. Crossmatch service

Every offline CLI already defaults to `http://quimal-db1.alerce.online:8081`, so
there is nothing to export. Just check it answers:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://quimal-db1.alerce.online:8081/   # expect 200
```

Override with `--xmatch-url` or `XMATCH_URL` to point at a different Xwave.

**Never pass `--xmatch-url ''`** unless you mean it: that forces the DB read, and
`multisurvey_ztf.allwise` is empty — the catalog rows are bulk-loaded by a
separate process that never ran for this schema. Every WISE colour would come out
NaN and the classifications would carry the Stochastic bias documented in
[`WISE_NULL_CLASSIFICATION_IMPACT.md`](./WISE_NULL_CLASSIFICATION_IMPACT.md).
A live Xwave that fails is different: those retry with backoff and then fail the
unit, rather than degrading silently.

## 7. Verify before committing 26.3M objects

Three checks, minutes each. Do not skip them; each has caught a real defect.

```bash
# a) the model's 199 features are all produced, and it predicts without KeyError
poetry run python scripts/offline_verify_model_features.py --smoke

# b) the seeded taxonomy matches the deployed pickle's classes (SESN, not SNIbc)
poetry run python scripts/offline_verify_taxonomy.py

# c) the batched reads hand the extractor the same inputs as the validated
#    single-oid path — a grouping bug would shift predicted classes silently
poetry run python scripts/offline_verify_batch_equivalence.py --n 12 --min-n-det 20
```

## 8. The run

```bash
# 1. materialize the oid list once (n_det >= 2 -> ~26.3M of the 130M).
#    offline_setup.py already did this; the explicit form is here for a
#    different cut (n_det >= 6 keeps ~7.5M, n_det >= 20 keeps ~2.5M).
poetry run python scripts/offline_run_batch.py --min-n-det 2 \
    --save-oids /data/oids/run.npy --plan-only

# 2. a small unit against the real database — the first end-to-end write.
#    --unit-size, NOT --max-units: a unit is one worker's task and is never split
#    across cores, so a default 5000-oid unit is 5000 objects in series.
poetry run python -c "import numpy as np; \
    np.savetxt('/data/oids/smoke.txt', np.load('/data/oids/run.npy')[:200], fmt='%d')"
poetry run python scripts/offline_run_batch.py \
    --oid-file /data/oids/smoke.txt --out-dir /data/bhrf_one \
    --unit-size 200 --minibatch 200 --workers 1 --features \
    --load-db --write-credentials features/offline/write_credentials.json
# then cross-check disk against DB for that unit:
jq '{db_prob_rows, prob_rows, db_feat_rows, feat_rows, db_xmatch_rows}' /data/bhrf_one/manifests/unit_0000000.json

# 3a. quick probe (~10 min): 64 small units over 16 workers, so there is a real
#     queue (4 units each) and the first checkpoint lands in minutes.
poetry run python scripts/offline_run_batch.py \
    --oid-file /data/oids/run.npy --out-dir /data/bhrf_probe1 \
    --unit-size 500 --max-units 64 --workers 16 --features
du -sh /data/bhrf_probe1

# 3b. full width, only once 3a looks sane: 128 real units over 64 workers.
poetry run python scripts/offline_run_batch.py \
    --oid-file /data/oids/run.npy --out-dir /data/bhrf_probe2 \
    --unit-size 5000 --max-units 128 --workers 64 --features
du -sh /data/bhrf_probe2       # extrapolate: this is 640k of 26.3M oids

# 4. the real run — rerun the SAME command after any interruption to resume
poetry run python scripts/offline_run_batch.py \
    --oid-file /data/oids/run.npy \
    --out-dir /data/bhrf_run --workers 64 --features \
    --load-db --write-credentials features/offline/write_credentials.json
```

Step 2 was run on 2026-08-20 over 20 oids against `multisurvey_ztf`: 900
probability rows, 3,319 feature rows spread across the hash partitions, and 16
`xmatch` rows all landed, and the manifest counters matched the database
exactly. Two things that check confirmed and that are worth re-checking on the
server: the upsert does not disturb the *other* classifiers' rows for the same
oids (the key includes `classifier_id`), and partition routing needs no grants
on the children. Repeat it there anyway — it is also the first real test of the
`write_user` login and of Xwave from that host.

Two probes, not one, because a small *fraction* of the catalogue is not the
same as a short run. `--max-units` truncates the unit list, but a unit is one
worker's task and its oids run in series inside it: 16 units of 5000 over 16
workers is one unit per worker, so the wall clock is however long 5000 objects
take back to back — an hour or more — while measuring nothing about queueing,
because no worker ever picks up a second unit. 3a fixes both by shrinking the
unit rather than the unit count.

Neither probe passes `--load-db`: a scaling test has no business writing to
production. Each needs its own `--out-dir` because `--unit-size` is part of the
`run.json` fingerprint, and neither leaves anything the real run will trip over
— it starts in a fresh directory and redoes those oids.

You do not have to read them by hand — every unit records its timing, row
counts and the worker's peak RSS in its manifest, and `offline_estimate.py`
turns those into a projection for the real run:

```bash
poetry run python scripts/offline_estimate.py /data/bhrf_probe1 \
    --oid-file /data/oids/run.npy --workers 64
```

```
measured: 64 units, 32,000 oids
  per oid        : 1.403 core-s
  per unit       : p50 14s  p90 15s  max 15s
  no AllWISE     : 14.2%  (~14% expected)
  peak RSS/worker: 664 MB
projected: 26,300,000 oids on 64 workers
  elapsed        : 160.2 h  (6.7 days)
  probability    : 1.18e9 rows
  feature        : 4.14e9 rows
  RSS all workers: 41 GB  <- compare against the host's RAM before scaling up
```

Four things decide whether to go ahead. **`per oid`** sets the duration, and it
divides by workers, so it is also what says whether more cores are worth it.
**`RSS all workers`** is the one that ends a run rather than slowing it: the
1.7 GB model is shared copy-on-write under `fork` and refcounting dirties the
pages it touches, so if that sharing degrades the projection climbs toward one
private copy per worker — check it against the host's RAM before scaling up.
**`per unit` p90 vs max** matters because a unit is one worker's task run in
series: the slowest unit decides when the last worker finishes, and the mean
hides it. **`no AllWISE`** should sit near 14%; much higher means Xwave is
returning empty, not that the sky is.

Run step 4 under `tmux`/`screen` or `nohup`: it is a multi-day job, and a
dropped SSH session kills the parent. If it is interrupted, rerun the identical
command — finished units are checkpointed and it picks up where it stopped.

**Size the disk from the probes, not from arithmetic.** Order of magnitude: ~45
probability rows and ~193 feature rows per object, so ~1.2e9 and ~5.1e9 rows
respectively across the run. `--features` is opt-in for exactly this reason.

**Defaults worth knowing:** `--workers` defaults to *physical* cores − 2 (not
`os.cpu_count()`, which counts hyperthreads and oversubscribes a CPU-bound run);
`--unit-size 5000` is the checkpoint granularity; `--minibatch 500` is the
round-trip granularity; `--start-method` is `fork` on Linux, which is what lets
all workers share one copy-on-write model.

## 9. While it runs

Per-unit progress goes to stdout; the durable record is on disk:

```
/data/bhrf_run/
├─ probabilities/unit_NNNNNNN.parquet
├─ features/unit_NNNNNNN.parquet
├─ errors/unit_NNNNNNN.jsonl      # only for units that hit per-oid failures
├─ manifests/unit_NNNNNNN.json    # written LAST — its presence means "done"
└─ run.json                       # oid-list fingerprint
```

Watch these in the summary:

- **`no AllWISE`** — expect ~14%. A much higher rate means Xwave is returning
  empty, not that the sky is empty.
- **`errors`** — now reported separately from "no detections" and
  "unclassifiable", which are expected outcomes and not worth chasing.

**Exit code:** 0 only when no unit was lost. Per-oid errors do *not* fail the
run — at this scale a handful of bad objects is expected, they are named in
`errors/unit_*.jsonl`, and folding them in would make the code non-zero on
essentially every run. A lost unit is different: it produced nothing, and the
only thing that recovers it is somebody rerunning the command. Ctrl-C exits
130, a worker dying abruptly exits non-zero with a diagnosis.

With `--load-db` the summary also prints `upserted to DB`, and each manifest
carries `db_prob_rows` / `db_feat_rows` / `db_xmatch_rows` beside `prob_rows` /
`feat_rows`. Those pairs matching is the only check that disk and database
agree; they are written in the same manifest precisely so the comparison needs
no query. `db_xmatch_rows` has no disk counterpart — the crossmatch is not
sharded — so it is the only record that the link rows were written at all.

Interrupting is safe. Finished units are checkpointed and the same command
resumes; the fingerprint in `run.json` refuses a resume against a different oid
list, because unit *N* only means anything relative to one specific array.

## 10. Retrying failed oids

A per-oid failure does not kill its unit — but the unit then completes and marks
itself done, so a plain rerun will never revisit it. Retry them explicitly:

```bash
cat /data/bhrf_run/errors/*.jsonl | jq -r .oid > /data/oids/retry.txt
poetry run python scripts/offline_run_batch.py \
    --oid-file /data/oids/retry.txt --out-dir /data/bhrf_run-retry --features
```

A separate `--out-dir` is required: the oid list differs, so the fingerprint will
not match the original shards. That is the guard working, not an obstacle.

## 11. Known gaps

| Gap | Consequence |
|---|---|
| **`--load-db` has never run a full 5000-oid unit.** It was exercised end to end over 20 oids (§8), not at unit scale. | The per-statement paging is untested against ~19k feature rows at once; step 2 on the server is what closes it. |
| **No parquet → DB backfill loader.** `--load-db` writes during the run; there is nothing that loads shards afterwards. | Units finished before the flag was turned on can only be redone into a fresh `--out-dir`. |
| `multisurvey_ztf.allwise` is empty. | `XMATCH_URL` is mandatory (§6). `--load-db` writes the `xmatch` link rows, but with no catalog rows to join against, the features still cannot be recomputed from the DB alone. |
| Objects with no AllWISE counterpart are indistinguishable from never-crossmatched ones in the stored data. | Only the per-unit `n_no_allwise` count records the difference. |
