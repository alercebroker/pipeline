# Quickstart — offline ZTF run on a fresh server

The commands, in order, with one line on what each is for. Target is **Ubuntu
24.04**. For *why* any of it is the way it is, read
[`SERVER_RUNBOOK.md`](./SERVER_RUNBOOK.md); this file is only the sequence.

```bash
# 1 — clone (working branch). Without --recurse-submodules the model cannot load.
git clone --recurse-submodules -b fix/ztf-feature-parser-extra-fields \
    git@github.com:alercebroker/pipeline.git
cd pipeline/feature_step

# 2 — Python 3.10 from deadsnakes: 24.04 ships 3.12 and pyproject pins <3.11.
sudo apt-get update
sudo apt-get install -y git build-essential software-properties-common python3-venv
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
python3.10 --version                        # -> 3.10.x

# 3 — Poetry in its own venv: a global pip install is refused (PEP 668) on 24.04.
python3 -m venv ~/.venvs/poetry && ~/.venvs/poetry/bin/pip install poetry
export PATH="$HOME/.venvs/poetry/bin:$PATH"
export POETRY_VIRTUALENVS_IN_PROJECT=true
poetry env use python3.10
poetry run python --version                 # -> 3.10.x, NOT 3.12

# 4 — install in this order: Cython and numpy before the C extensions.
poetry run python -m pip install setuptools wheel Cython==0.29.36 numpy
poetry run python -m pip install ../mhps
poetry run python -m pip install -r ../P4J/requirements.txt
poetry install --without=test --no-root
# poetry.lock says scikit-learn 1.7.2; alerce_classifiers asks for ~1.4.2 and
# imbalanced-learn imports _safe_tags, gone since 1.6. The lock wins, so undo it.
poetry run python -m pip install "scikit-learn==1.4.2"
poetry run python -c "import P4J, mhps; from features.offline import db; print(db.SCHEMA)"

# 5 — credentials. The only manual step: it carries a password, so it is neither
#     generated nor committed. One account reads and writes everything.
cat > features/offline/credentials.json <<'EOF'
{"user":"write_user","password":"...","host":"quimal-db1.alerce.online","port":5432,"dbname":"ztf"}
EOF
chmod 600 features/offline/credentials.json

# 6 — setup. Checks deps, privileges and seeds; downloads the model and builds
#     the oid list. Exits non-zero until the machine can start a run.
poetry run python scripts/offline_setup.py --check-only     # what is missing, changes nothing
poetry run python scripts/offline_setup.py                  # ...and fixes what it can

# 6b — point the loader at the model setup just downloaded. Everything from
#      here on loads the pickle, and none of it falls back to a default.
export MODEL_PATH=$PWD/features/offline/models/hierarchical_random_forest_model.pkl

# 7 — three verifications. Each has caught a real defect; minutes each.
poetry run python scripts/offline_verify_model_features.py --smoke                    # all 199 features produced
poetry run python scripts/offline_verify_taxonomy.py                                  # seeded taxonomy == the pickle's
poetry run python scripts/offline_verify_batch_equivalence.py --n 12 --min-n-det 20   # batched == single-oid path

# 8 — 200 oids against the database. First real write from this machine.
export RUN=$HOME/bhrf && mkdir -p $RUN/oids
poetry run python -c "import numpy as np; np.savetxt('$RUN/oids/smoke.txt', \
    np.load('features/offline/oids/run.npy')[:200], fmt='%d')"
poetry run python scripts/offline_run_batch.py \
    --oid-file $RUN/oids/smoke.txt --out-dir $RUN/bhrf_one \
    --unit-size 200 --minibatch 200 --workers 1 --features \
    --load-db --write-credentials features/offline/credentials.json

# 8b — the db_* counters must match their on-disk pair. The only check that the
#      database got what the run computed.
jq '{db_prob_rows, prob_rows, db_feat_rows, feat_rows, db_xmatch_rows}' \
    $RUN/bhrf_one/manifests/unit_0000000.json

# 9 — probe. Measures real throughput and memory. No --load-db: it must not
#     touch production.
poetry run python scripts/offline_run_batch.py \
    --oid-file features/offline/oids/run.npy --out-dir $RUN/bhrf_probe1 \
    --unit-size 500 --max-units 64 --workers 16 --features

# 10 — the estimate: duration, rows and RSS projected onto all 26.3M objects.
poetry run python scripts/offline_estimate.py $RUN/bhrf_probe1 \
    --oid-file features/offline/oids/run.npy --workers 64

# 11 — the run. Under tmux: it takes days, and a dropped SSH session kills the parent.
tmux new -s bhrf
export MODEL_PATH=$PWD/features/offline/models/hierarchical_random_forest_model.pkl
poetry run python scripts/offline_run_batch.py \
    --oid-file features/offline/oids/run.npy --out-dir $RUN/bhrf_run \
    --workers 64 --features \
    --load-db --write-credentials features/offline/credentials.json --no-shards
```

## Before starting step 11

**Read two numbers from step 10.** `RSS all workers` against the host's RAM —
that is the one that ends a run rather than slowing it. And `per oid`, which is
the duration: if it is more than you want to spend, that is the moment to raise
the `n_det` cut (`offline_setup.py --min-n-det 6` keeps ~7.5M instead of
~26.3M), not after the run has started.

**If it is interrupted, rerun step 11 unchanged.** It resumes from the
manifests. `--workers` may change between reruns; `--unit-size` may not — it is
part of the `run.json` fingerprint.

**`--no-shards` saves ~68 GB** and costs nothing with `--load-db`: the database
holds the same rows and nothing ever reads the parquet back. Drop the flag if
you want the local copy anyway.

**`--load-db` cannot be turned on midway** through one `--out-dir`. Units that
finished without it stay on disk and are never loaded, and there is no loader
that could pick them up.

## When it does not work

| Symptom | Cause |
|---|---|
| `Unable to locate package python3.10-dev` | deadsnakes not added yet — step 2, third line. |
| `Python.h: No such file or directory` while building mhps | `python3.10-dev` missing. `build-essential` gives the compiler, not the headers. |
| `error: externally-managed-environment` | `pip install poetry` run globally. Use the venv in step 3. |
| `poetry run python --version` says 3.12 | `poetry env use python3.10` was skipped. Everything after it will fail with unrelated-looking errors. |
| `ModuleNotFoundError: No module named 'wget'` | `poetry install` did not finish. It is a dependency of `alerce_classifiers`, pulled in by the path dep. Re-run step 4's `poetry install` and read its output; `poetry run python -m pip install wget` unblocks you meanwhile. |
| `cannot import name '_safe_tags' from 'sklearn.utils._tags'` | `poetry.lock` installed scikit-learn 1.7.2. `imbalanced-learn` needs `_safe_tags`, removed in 1.6. Run the pin in step 4. |
| setup reports `Permission denied: '/data'` | A stale `MODEL_PATH` is exported in your shell. `unset MODEL_PATH`, then re-export it as in step 6b. `/data` is root-owned. |
| `MODEL_PATH env var is required to load the model` | Step 6b was skipped, or you are in a shell that never had it — a fresh `tmux` window included. |
| setup reports missing privileges | See the `GRANT`s in §4 of the runbook. `USAGE` on the schema is separate from the table grants and its absence makes them dead. |
| `no AllWISE` far above ~14% | Xwave is returning empty, not the sky. Check §6 of the runbook before trusting the classifications. |

## Step 12 — the tail run (what the full run did not cover)

`run.npy` is a **snapshot**, not a window: `select_oids` has no date filter, so
objects that entered `object` or crossed the `n_det` cut after that file was
built are absent from the run and nothing reports their absence
([`BHRF_RUN_RESULTS.md` §5](./BHRF_RUN_RESULTS.md)). To process only those,
diff the catalogue against the baseline and run the difference.

```bash
# 12a — what the tail would be. Queries the DB, writes nothing. Two table scans
#       if --since-date is given, one if not.
poetry run python scripts/offline_tail_oids.py --dry-run --run-dir $RUN/bhrf_run

# 12b — the list. --run-dir checks run.npy against the finished run's run.json
#       fingerprint: diffing against a rebuilt baseline gives the wrong tail.
poetry run python scripts/offline_tail_oids.py \
    --baseline features/offline/oids/run.npy --run-dir $RUN/bhrf_run \
    --out $RUN/oids/tail.npy

# 12c — the run. A FRESH --out-dir, and no --max-units.
export MODEL_PATH=$PWD/features/offline/models/hierarchical_random_forest_model.pkl
poetry run python scripts/offline_run_batch.py \
    --oid-file $RUN/oids/tail.npy --out-dir $RUN/bhrf_tail \
    --workers 64 --features \
    --load-db --write-credentials features/offline/credentials.json --no-shards
```

**New objects, or also updated ones?** By default the tail is *new* oids only —
over the cut today, absent from the baseline. Objects the run already processed
whose light curve has since grown carry stale features and probabilities, and
they are a different (much larger) set. Add `--since-date 2026-08-14` — the
run's data horizon — to fold in every baseline oid with `object.lastmjd` past
that date. Reprocessing them is safe: `feature`, `probability` and `xmatch` all
write `ON CONFLICT ... DO UPDATE`, so the second pass overwrites its own rows.

**The fresh `--out-dir` is not optional.** Unit index N means
`oids[N*unit_size : (N+1)*unit_size]` of *one specific array*. Pointing the tail
at `$RUN/bhrf_run` makes `run.json` refuse the resume (correctly) — and
`--force-resume` there would mark tail units as done against the full run's
indices.

**`--min-n-det` must match the baseline's cut.** The default is 2, the same as
`offline_setup.py`. A different value compares two different questions and the
diff is meaningless.

**If step 12a reports `0 new`, stop.** `object` had received no rows since
2026-08-14 as of the run; an empty tail means the table still has not moved, and
the script writes nothing rather than leaving an empty list behind.
