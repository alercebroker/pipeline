# Quickstart — reclassify the squidward 2.1.0 objects with 2.2.0 on `quimal-cpu1`

The commands, in order, for rerunning the BHRF classifier **from the features
already stored** in `multisurvey.feature`, over the 20,620,133 objects that
hold squidward 2.1.0 (`classifier_version = 210`) probabilities, and writing the
2.2.0 probabilities (`classifier_version = 220`) over them. No feature
computation, no crossmatch: `scripts/offline_reclassify_batch.py` reads ~190
feature rows per object, rebuilds the model's 199-column frame, classifies the
minibatch in one call and upserts the 45 probability rows per object.

The 2.2.0 pickle was checked against 2.1.0 before this was written: same 21
classes in the same order, same 4 branches with the same class order, same 199
feature names. Only the trees differ. That is why the taxonomy rows
(classifier_ids 5 to 9) and the feature list are reused unchanged.

**What it assumes.** The 2.1.0 rows were dumped first (`~/dumps/prob_part_*_v210.parquet`
on cpu1, 927.9M rows); the upsert replaces them in place, and the dump is the only
copy afterwards. The oid list is the distinct oids of that dump.

```bash
# 0 — the code. The runner, its module and tests live on branch
#     reclass/squidward-2.2.0 (off fix/ztf-feature-parser-extra-fields, which the
#     server checkout is on). Same venv, nothing to install.
cd /home/alerce/features_batch_processing/pipeline/feature_step
git status --short                                   # look before pulling
git fetch origin && git switch reclass/squidward-2.2.0
poetry run python scripts/offline_reclassify_batch.py --help | head -3

# 1 — the oid list, built on the laptop from the 16 dump partitions
#     (np.unique over the oid column; 20,620,133 int64, sorted, SHA-1 b2e02525…).
#     From the laptop:
#       rsync -avh --progress ~/Desktop/dumps/squidward_v210_oids.npy* quimal-cpu1:bhrf/oids/
export RUN=$HOME/bhrf
cat $RUN/oids/squidward_v210_oids.npy.json           # n_oids 20620133

# 2 — the database side, read-only. The runner stamps classifier_ids 5-9 and
#     needs their rows in multisurvey.taxonomy (it refuses to start otherwise).
#     multisurvey.classifier still says 2.1.0 for those ids: that string is
#     bumped by hand, as superuser, once the run is done (step 7).
poetry run python -c "
from features.offline import db; import pandas as pd; from sqlalchemy import text
e=db._make_engine('features/offline/credentials.json')
with e.connect() as c:
    print(pd.read_sql(text('SELECT classifier_id, classifier_name, classifier_version FROM multisurvey.classifier ORDER BY 1'), c).to_string())
    print(pd.read_sql(text('SELECT classifier_id, count(*) AS classes FROM multisurvey.taxonomy WHERE classifier_id BETWEEN 5 AND 9 GROUP BY 1 ORDER BY 1'), c).to_string())"
# expect classes 21, 3, 6, 6, 9 for ids 5..9

# 3 — the environment. MODEL_PATH is the 2.2.0 URL; the runner refuses a pickle
#     that reports any other version. Per shell / tmux window.
export RUN=$HOME/bhrf
export MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.2.0/hierarchical_random_forest_model.pkl

# 4 — smoke: one unit of 500 oids, no database writes, a throwaway --out-dir.
#     Confirms the checkout, the model download, the taxonomy and the feature
#     read work together. n_no_features and n_no_lastmjd should be ~0: every
#     oid in the list has 210 rows by construction.
poetry run python scripts/offline_reclassify_batch.py \
    --oid-file $RUN/oids/squidward_v210_oids.npy --out-dir $RUN/reclass_220_smoke \
    --schema multisurvey --model-version 2.2.0 \
    --credentials features/offline/credentials.json \
    --unit-size 500 --max-units 1 --workers 4
jq '{n_oids, n_ok, n_no_features, n_no_lastmjd, n_errors, prob_rows, elapsed_s, t_fetch_s, t_compute_s}' \
    $RUN/reclass_220_smoke/manifests/unit_*.json
poetry run python -c "
import pandas as pd; d=pd.read_parquet('$RUN/reclass_220_smoke/probabilities/unit_0000000.parquet')
print(d.classifier_version.unique(), d.groupby('oid').size().value_counts().to_dict())"
# expect [220] and {45: 500}

# 5 — the run. Under tmux; a FRESH --out-dir; --load-db writes. Every write is
#     an upsert on (oid, sid, classifier_id, class_id), so the 210 rows become
#     220 rows in place; lastmjd is copied from the 210 row of classifier 5.
tmux new -s reclass
cd /home/alerce/features_batch_processing/pipeline/feature_step
export RUN=$HOME/bhrf
export MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.2.0/hierarchical_random_forest_model.pkl
poetry run python scripts/offline_reclassify_batch.py \
    --oid-file $RUN/oids/squidward_v210_oids.npy --out-dir $RUN/reclass_220 \
    --schema multisurvey --model-version 2.2.0 \
    --credentials features/offline/credentials.json \
    --workers 32 --stall-timeout 7200 \
    --load-db --write-credentials features/offline/credentials.json --no-shards
tmux pipe-pane -t reclass -o "cat >> $RUN/reclass_pane.log"
# Ctrl-b d to detach. A resume is the same command: run.json pins the oid
# array, the unit size, the schema and the model version.

# 6 — progress, from another window. 4,125 units of 5,000 oids.
ls $RUN/reclass_220/manifests/unit_*.json | wc -l
grep -E '^\[' $RUN/reclass_pane.log | tail -5
jq -s '{units:length, oids:(map(.n_oids)|add), ok:(map(.n_ok)|add),
        no_features:(map(.n_no_features)|add), no_lastmjd:(map(.n_no_lastmjd)|add),
        errors:(map(.n_errors)|add), prob:(map(.prob_rows)|add), db_prob:(map(.db_prob_rows)|add)}' \
    $RUN/reclass_220/manifests/unit_*.json
# prob and db_prob must match to the row on every unit; ok should approach 20,620,133.

# 7 — after the run, as superuser, the version string the API reads:
#   UPDATE multisurvey.classifier SET classifier_version = '2.2.0' WHERE classifier_id BETWEEN 5 AND 9;
```

## Sizing

Per 500-oid minibatch the unit does two indexed reads (~95k feature rows and 500
probability rows by primary key) and one model call over a 500 × 199 frame
(four forests of 500 trees). There is no light-curve compute, so the cost per
object is a small fraction of the feature runs'; read `t_fetch_s` and
`t_compute_s` in the smoke manifest and scale from there before picking
`--workers`. The DB read is the part that scales with workers; the model call
runs single-threaded per worker (BLAS pinned to 1).

## What the run leaves on disk

Under `$RUN/reclass_220`: `run.json` (fingerprint, with `model_version` and
`schema`), `manifests/unit_*.json` (one per finished unit, written after the
upsert commits), `errors/unit_*.jsonl` (per-oid exceptions, if any). No shards
with `--no-shards`: the database holds the rows.
