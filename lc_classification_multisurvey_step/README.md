# LC Classification Multisurvey Step

Consumes the multisurvey `feature_step` output topic, runs the ZTF BHRF
(Squidward 2.1.0) classifier, and produces probabilities for its five heads to
`scribe_multisurvey`, which owns the upsert into `multisurvey_ztf.probability`.

The step writes nothing to the database. It reads `classifier` and `taxonomy`
once at startup to resolve classifier names to ids and class names to class ids,
and refuses to start if either is unseeded (see the design doc, §8).

Design: `docs/superpowers/specs/2026-08-16-multisurvey-lc-classification-step-design.md`

## Tests

The unit suite has no model dependency:

    python -m pytest tests/unittest -v

The offline-equivalence test is opt-in. It needs the offline checkout at
`~/desktop/pipeline/feature_step` — note the `feature_step` subdirectory, since
`features.offline` does not resolve from the checkout root. It needs neither the
`alerce_classifiers` submodule nor `MODEL_PATH`: both row builders are pure and
no classifier is run.

    RUN_EQUIVALENCE_TEST=1 python -m pytest tests/integration -v

If it reports `SKIPPED`, the offline checkout was not found — the skip message
names the path it looked for. A skip here means the test did not run at all, so
treat it as no coverage rather than as a pass.

This is the only test that checks the port against the implementation it was
ported from; the unit suite checks the port's internal consistency. It compares
one single-oid `OutputDTO` through both row builders. It cannot cover the
multi-oid melt path, because the offline reference raises on a multi-row frame
by design.

### Against real data

The synthetic harness below proves the wiring; it cannot say the port
classifies *correctly*, because its features are random floats. That is what
`tests/integration/test_real_data_equivalence.py` is for: it takes objects the
production pipeline has already classified, recomputes their probabilities
through the step's own path, and requires the result to match the rows in
`multisurvey_ztf.probability`.

    MODEL_PATH=/path/to/model/2.1.0 \
    python -m pytest tests/integration/test_real_data_equivalence.py -v

The objects live in `tests/integration/data/real_examples.json.gz` (992 of them,
1.5 MiB), so the test needs neither the VPN nor credentials — only the model.
Regenerate the fixture after a model bump or a change to the feature set:

    REAL_DB_CONFIG=$(pwd)/local_config.yaml python scripts/dump_real_examples.py --count 1000

`REAL_DB_CONFIG` points at any yaml with a `PSQL_CONFIG` block — the local run
config already has one — so no credentials live in this repo.

Objects are taken as they come. The only condition is that the object has BHRF
rows to compare against; there is no filter on feature completeness, so the set
spans 19 to 189 of the model's 199 features and exercises the NaN handling on
real sparsity. It is drawn evenly from all 32 `feature` partitions.

2026-09-03: 44640 rows over 992 objects, **all matching exactly** — same
probabilities, same rankings, same top class, and the row sets are identical, so
the step writes exactly what production wrote, no more and no fewer. This is
also the only test that melts a realistic batch — 992 oids at once, where the
offline reference raises on a multi-row frame and the synthetic harness only
reaches five.

### Superseded feature rows

Reaching exact equality needed one rule, in `_current_features`: keep only the
rows carrying each object's most recent `updated_date`.

`feature` is upserted `ON CONFLICT (oid, sid, feature_id, band) DO UPDATE ...
updated_date = now()`, so a pass only touches the rows it computed. A feature
computed in an earlier pass but *not* produced by the latest one keeps its old
row, its old value and its old date, while its ~190 siblings move on. The
classifier, working from that latest computation, saw NaN for it — so reading
the row back as a value feeds the model something production never had.

Three of the 992 objects carried exactly that (3–4 rows dated a day before the
rest), and they were the only three whose recomputed probabilities disagreed.
With the superseded rows dropped, every object matches.

This is a property of the table, not of this test: **anything** that
reconstructs a feature vector from `feature` has to apply the same rule, or it
silently mixes two generations of a computation. It is easy to miss because
`updated_date` is a DATE — same-day generations are indistinguishable — and
because `probability` carries no timestamp at all to compare against.

Reading features back out of the database needs a name translation, and getting
it wrong is silent — around 31 of the 199 features land as NaN and the model
returns plausible but different probabilities. `feature_name_lut` spells colours
with hyphens (`W1-W2`) and ratios with slashes (`Power_rate_1/2`) where the
model's `feature_list` uses underscores, and `band` 12 is the (g,r) pair, so
`g-r_mean` at band 12 is the model's `g_r_mean_12`. The step itself is
unaffected: its features arrive in the Kafka message, and
`features_record_ztf.avsc` already uses the model's spelling.

## Running the step without an upstream

The multisurvey `feature_step` lives on another branch and its topic is empty,
so there is nothing to consume and nothing to read the taxonomy from. The
integration harness supplies both locally: `tests/integration/docker-compose.yml`
brings up Kafka and Postgres, `tests/integration/fake_features.py` generates
schema-valid feature messages, and `tests/integration/taxonomy_seed.py` holds the
BHRF seed the step refuses to start without.

`tests/integration/test_local_pipeline.py` runs the whole thing: it needs Docker
and `MODEL_PATH` (the BHRF 2.1.0 pickle is not in this repo, and is the only
piece the harness cannot fake), and skips without the latter.

    MODEL_PATH=/path/to/model/2.1.0 python -m pytest tests/integration/test_local_pipeline.py -v

If a run reports `Timeout reached while waiting on service!`, the broker is not
up: check `docker logs` for the kafka container before suspecting the fixtures.
A crashed run also leaves its containers behind, and they hold ports 9092/5432
so the next run cannot bind them — clear them with

    docker rm -f $(docker ps -aq --filter name=pytest)

The probabilities are meaningless — the features are random floats — so the
assertions are on row counts, identifiers and envelope shape, never on values.
What this covers that the unit suite cannot: the apf consumer/producer wiring,
the §8 startup queries against a real Postgres, and the scribe envelope as it
lands on a topic.

To drive the step by hand instead, bring the same compose up, seed the database
the way `conftest.py` does, produce some messages, and point `CONFIG_YAML_PATH`
at a local yaml with `bootstrap.servers: localhost:9092` and the Postgres
credentials from the compose file:

    docker compose -f tests/integration/docker-compose.yml up -d
    python scripts/produce_fake_features.py --count 20
    CONFIG_FROM_YAML=yes CONFIG_YAML_PATH=$(pwd)/local_config_docker.yaml \
        python scripts/run_step.py

### The seed is not in db-plugins

`_initial_data_pipeline.py` seeds classifier ids 1-4 for the stamp classifiers
only; the five BHRF heads are not there (design §12 tracks the back-port). Until
that lands, `taxonomy_seed.py` is the definition, and its class names were read
off the model pickle rather than written by hand — `build_probability_rows`
raises when one class name is missing from the taxonomy, so a seed guessed from
the class hierarchy hardcoded in `alerce_classifiers` would crash the step on
its first batch: that hierarchy is stale (it lists `SNIbc`/`SNIIb` and `RRL`,
while 2.1.0 emits `SESN` and `RRLab`/`RRLc`). After a model bump, regenerate:

    python scripts/dump_model_taxonomy.py /path/to/model/2.1.0
