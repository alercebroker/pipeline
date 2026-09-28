# Quickstart — rerun on `quimal-cpu1` for the reprocessed ZTF objects

The commands, in order, for classifying the objects of the reprocessing
campaign on the server that already ran the full catalogue and the tail run.
Nothing is installed here: the checkout at
`/home/alerce/features_batch_processing/pipeline`, the poetry venv, the model,
the credentials, `$RUN/bhrf_run` and `$RUN/bhrf_tail` are all as
[`SERVER_QUICKSTART.md`](./SERVER_QUICKSTART.md) steps 1 to 12 left them, and the
list of names is already at `$RUN/oids/ztf_reprocesados_oids.txt.gz`. For *why*
the offline run works the way it does, read
[`SERVER_RUNBOOK.md`](./SERVER_RUNBOOK.md).

**What this run is.** The campaign's manifest, `ztf_reprocesados.parquet`, is a
table of **alerts** (one row per `candid`), 29.7M rows, of which 9.2M carry no
`oid`. The 20.5M that do belong to **15,453,749 distinct objects**,
`ZTF17aaaaaak` to `ZTF26abtnngf`. This run recomputes features, classification
and crossmatch for those objects, from the detections the database holds today.
It is step 12 of the quickstart with the list handed in instead of selected:
a new oid array, a fresh `--out-dir`, the same flags as the full run.

**What it assumes.** The reprocessed alerts are already ingested into
`multisurvey_ztf` — that is what makes the rerun produce anything new. The
runner reads `detection` and `forced_photometry`; it does not read the parquet.
If the ingestion is not done, the run recomputes the same light curves the
August run saw and overwrites each row with itself. Step 0 checks this on a
handful of objects before anything else is spent.

```bash
# 0 — is the reprocessed data in the database? Pick three oids from the
#     parquet and compare what the catalogue has now with what the full run
#     classified them at. probability.lastmjd is the MJD the run saw; if
#     object.lastmjd or n_det moved past it, the ingestion happened.
poetry run python -c "
from features.offline import db; import pandas as pd; from sqlalchemy import text
oids=[36028933559736971, 36028933559755080]   # <- three oids from the list, step 3
e=db._make_engine('features/offline/credentials.json')
with e.connect() as c:
    print(pd.read_sql(text(f'''
      SELECT o.oid, o.n_det, o.lastmjd, o.updated_date,
             (SELECT max(lastmjd) FROM {db.SCHEMA}.probability p WHERE p.oid=o.oid) AS run_lastmjd
      FROM {db.SCHEMA}.object o WHERE o.oid = ANY(:oids) AND o.sid=:sid'''),
      c, params={'oids':oids,'sid':db.SID}).to_string())"

# 1 — the code. The list builder (scripts/offline_reprocessed_oids.py) and the
#     colour backfill (scripts/offline_backfill_object_colors.py) landed after
#     the tail run; pull the working branch. A dirty tree here means someone
#     edited the server checkout by hand: look before you pull. The venv is
#     the checkout's own `.venv` (poetry is at ~/.venvs/poetry/bin, already on
#     PATH); `poetry run` from this directory uses it. Nothing to activate or
#     install.
cd /home/alerce/features_batch_processing/pipeline/feature_step
poetry env info --path                              # -> .../feature_step/.venv
git status --short
git pull --recurse-submodules
poetry run python scripts/offline_reprocessed_oids.py --help | head -3
poetry run python scripts/offline_backfill_object_colors.py --help | head -3

# 2 — the names are already on the server: 15,453,749 newline-separated ZTF
#     names, 40 MB gzipped, at $RUN/oids. (Built on the laptop from the
#     parquet's oid column with nulls dropped and duplicates removed; the
#     parquet itself never leaves the laptop.) Unpack and count.
export RUN=$HOME/bhrf
gunzip -k $RUN/oids/ztf_reprocesados_oids.txt.gz
wc -l $RUN/oids/ztf_reprocesados_oids.txt          # -> 15453749

# 3 — the oid array. --eligible keeps what the catalogue can classify (present
#     in object, n_det >= 2 — the run's cut) and drops the rest: an oid that is
#     not there costs a round trip and lands in "no detections". --run-dir
#     checks run.npy against the finished run's fingerprint, so the split into
#     "never processed / processed before" is trustworthy. Dry run first:
#     minutes (the catalogue selection is a bitmap scan), writes nothing.
poetry run python scripts/offline_reprocessed_oids.py \
    --ztf-list $RUN/oids/ztf_reprocesados_oids.txt \
    --eligible --run-dir $RUN/bhrf_run --dry-run
poetry run python scripts/offline_reprocessed_oids.py \
    --ztf-list $RUN/oids/ztf_reprocesados_oids.txt \
    --eligible --run-dir $RUN/bhrf_run --out $RUN/oids/reprocesados.npy
cat $RUN/oids/reprocesados.npy.json                 # the counts, kept next to the array

# 4 — the environment, again. A fresh shell or tmux window has none of it
#     (same as before the tail run: RUN and MODEL_PATH are per shell).
export RUN=$HOME/bhrf
export MODEL_PATH=$PWD/features/offline/models/hierarchical_random_forest_model.pkl
curl -s -o /dev/null -w '%{http_code}\n' http://quimal-db1.alerce.online:8081/   # 200
df -h $RUN | tail -1                                 # manifests only with --no-shards; MBs

# 5 — smoke: two units, no database writes, a throwaway --out-dir. Confirms
#     the checkout, the model, the credentials and Xwave still work together.
#     Read `no AllWISE` in the manifests: far above ~14% means Xwave is empty.
poetry run python scripts/offline_run_batch.py \
    --oid-file $RUN/oids/reprocesados.npy --out-dir $RUN/bhrf_reproc_smoke \
    --unit-size 500 --max-units 2 --workers 8 --features
jq '{n_oids, n_ok, n_unclassifiable, n_no_detections, n_errors, n_no_allwise}' \
    $RUN/bhrf_reproc_smoke/manifests/unit_*.json

# 6 — the run. Under tmux; a FRESH --out-dir; no --max-units. Same flags as
#     the full run. Every write is an upsert, so objects the full run already
#     classified get their feature / probability / xmatch rows overwritten.
tmux new -s reproc          # the tail run used `tail`; a new name, a new window
cd /home/alerce/features_batch_processing/pipeline/feature_step
export RUN=$HOME/bhrf
export MODEL_PATH=$PWD/features/offline/models/hierarchical_random_forest_model.pkl
poetry run python scripts/offline_run_batch.py \
    --oid-file $RUN/oids/reprocesados.npy --out-dir $RUN/bhrf_reproc \
    --workers 64 --features \
    --load-db --write-credentials features/offline/credentials.json --no-shards

# 7 — progress, from another window. Unit count against the plan the run
#     printed; totals from the manifests, never from the end-of-run summary
#     (BHRF_RUN_RESULTS.md, caveats).
ls $RUN/bhrf_reproc/manifests/unit_*.json | wc -l
jq -s '{units:length, oids:(map(.n_oids)|add), ok:(map(.n_ok)|add),
        unclassifiable:(map(.n_unclassifiable)|add), no_det:(map(.n_no_detections)|add),
        errors:(map(.n_errors)|add)}' $RUN/bhrf_reproc/manifests/unit_*.json

# 8 — when it ends: the disk-vs-database counters must match to the row, on
#     every unit. This is the only check that the database got what was computed.
jq -s '{prob:(map(.prob_rows)|add), db_prob:(map(.db_prob_rows)|add),
        feat:(map(.feat_rows)|add), db_feat:(map(.db_feat_rows)|add),
        xmatch:(map(.db_xmatch_rows)|add)}' $RUN/bhrf_reproc/manifests/unit_*.json

# 9 — the object colours. Its own --out-dir: the array defines the ranges.
poetry run python scripts/offline_backfill_object_colors.py \
    --oid-file $RUN/oids/reprocesados.npy --out-dir $RUN/object_colors_reproc \
    --credentials features/offline/credentials.json --execute --max-chunks 2
poetry run python scripts/offline_backfill_object_colors.py \
    --oid-file $RUN/oids/reprocesados.npy --out-dir $RUN/object_colors_reproc \
    --credentials features/offline/credentials.json --execute
```

## Before starting step 6

**How long.** The full run measured 156 oid/s on 64 workers (0.559 core-s per
oid, `BHRF_RUN_RESULTS.md` §3). Step 3 on 2026-09-27 kept **6,779,444** of the
15.45M names (5,163,575 processed before, 1,615,869 never processed), so the
run is **~12 h**. The 8.67M dropped are all present in `object` with a single
detection: `scripts/offline_reprocessed_probe.py` on a 2,000-name sample found
0 absent, 56% under the cut, 44% eligible, and 46% of the sample last written on
2026-09-22 -- the ingestion did land. The oids are sorted, so throughput should
match the full run's, not the probe's.

**Memory.** Nothing about the run changed, so `RSS all workers` from the
original step 10 estimate still applies. If the host has less free memory than
it had in August (another job running), lower `--workers`; it may change between
resumes, `--unit-size` may not.

**If it is interrupted, rerun step 6 unchanged.** It resumes from the manifests
in `$RUN/bhrf_reproc`. `run.json` there pins `reprocesados.npy` by SHA-1; a
rebuilt array (step 3 run again with a different list or cut) will be refused,
correctly. Do not `--force-resume` past that.

**Read the split from step 3.** `processed before` objects are the ones whose
rows this run overwrites; `never processed` are objects the full run did not
have (born since, or under the cut in August). If `never processed` is close to
zero and step 0 showed no movement in `lastmjd`, the ingestion has not happened
and this run changes nothing — stop and check that first.

**The tail run is done.** It ran in the tmux session `tail` over `tail.npy`
into `$RUN/bhrf_tail`; this run is the same procedure over a third array into
its own `--out-dir`. The two lists may overlap; upserts make that harmless. The
split step 3 prints is against `run.npy` only, so an object the tail run
classified but the full run did not shows up as `never processed`. Do not start
this run while another one is still writing — both use the same tables through
the same account, and the full run's throughput was measured with the database
to itself.

## What the list builder does

`scripts/offline_reprocessed_oids.py` turns ZTF names into the bigint array the
runner consumes:

| step | what | why |
|---|---|---|
| encode | `ZTFyyxxxxxxx` → `1<<55 + yy·26⁷ + base-26(xxxxxxx)`, idmapper's formula, vectorised | a mistyped name is not an error downstream, it is an oid with no detections; every name is validated first and the first offender is named |
| unique + sort | ascending int64 | unit N is `oids[N·unit_size:…]` of one array; ascending keeps a unit's rows on adjacent index pages |
| `--eligible` | intersect with `object WHERE sid=0 AND n_det >= 2` | same query as the full run's `select_oids`; drops what would only cost round trips |
| baseline | `np.isin` against `run.npy`, fingerprint-checked with `--run-dir` | labels only; it never adds or removes an oid |

It writes `<out>.npy` and `<out>.npy.json` with the counts and the array's
SHA-1, the same value `run.json` will pin.

## When it does not work

| Symptom | Cause |
|---|---|
| `Invalid ZTF object ID: '…'` from step 3 | A name in the list is not `ZTF` + 2 digits + 7 lowercase letters. The offending line is quoted; fix the list, not the script. |
| `BASELINE MISMATCH` from step 3 | `features/offline/oids/run.npy` is not the array `$RUN/bhrf_run/run.json` pinned — someone rebuilt it. Point `--baseline` at the array the run used. The list is unaffected; only the label is. |
| step 3 drops most of the list as ineligible | `scripts/offline_reprocessed_probe.py <list>` samples the names and splits the drop into absent / under the cut. Absent means the campaign's objects are not in `multisurvey_ztf.object` yet (ingestion pending — see step 0); under the cut is single-detection objects, expected to be most of an alert manifest (56% on 2026-09-27). |
| `run.json` refuses the resume in step 6 | `--out-dir` reused from another array (`bhrf_run`, `bhrf_tail`) or `--unit-size` changed. Fresh directory, default unit size. |
| `MODEL_PATH env var is required to load the model` | Step 4 skipped in this shell — every tmux window needs it. |
| `no AllWISE` far above ~14% in step 5 | Xwave returning empty, not the sky. Runbook §6 before trusting anything. |
| db counters differ from disk counters in step 8 | A unit committed partially. Its manifest names it; rerun step 6 — a unit without a manifest is redone from scratch and its upserts overwrite the partial rows. |
