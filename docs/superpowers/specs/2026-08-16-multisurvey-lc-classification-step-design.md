# Multisurvey LC classification step — design

**Branch:** `feat/multisurvey-lc-classification-step` (off `multisurvey`)
**Date:** 2026-08-16
**Status:** design approved, not yet implemented

A new `lc_classification_multisurvey_step/` that consumes the multisurvey
`feature_step` output, runs the ZTF BHRF (Squidward 2.1.0) classifier, and writes
probabilities into `multisurvey_ztf.probability` through `scribe_multisurvey`.

---

## 1. Why a new step

The deployed `lc_classification_step/` is a legacy, multi-survey, multi-model step:
it carries eight model families through `models_settings.configurator`, builds a
detections DTO the ZTF model never reads, and writes probabilities to the scribe
with **string** class names keyed by a **string** `_id`. The multisurvey database
keys probabilities by `(oid bigint, sid, classifier_id, class_id)` with an integer
`class_id` resolved from the `taxonomy` table.

Bridging those two inside one step means a config-switched fork through every
parser. The repo has already answered this shape question three times —
`correction_multisurvey_step`, `magstats_multisurvey_step`,
`stamp_classifier_2025_multisurvey_step`, `scribe_multisurvey` are all standalone
sibling directories. This follows that convention.

The offline pipeline in `feature_step/features/offline/` (in the `~/desktop/pipeline`
checkout, branch `fix/ztf-feature-parser-extra-fields`) already implements and
validates the BHRF → multisurvey-probability path end to end. **This step is the
streaming port of that offline logic**, not a new derivation. §5 tracks the port
line by line.

## 2. Decisions

| # | Decision | Chosen |
|---|---|---|
| 1 | Structure | **New `lc_classification_multisurvey_step/` dir**, copy-and-adapt. Legacy step untouched. |
| 2 | Model scope | **ZTF BHRF only** (`SquidwardFeaturesClassifier`). No elasticc/balto/messi/toretto/barney/mbappe/anomaly. A one-entry `models_settings.configurator` is kept so the model/mapper/path stay env-driven, matching the stamp step. |
| 3 | Persistence | **Scribe-only.** No DB writes from the step. `scribe_multisurvey` owns the upsert. |
| 4 | `class_name → class_id` | **Read-only DB lookup at startup**, cached. DB `taxonomy` is the authority. |
| 5 | Downstream Kafka topic | **New multisurvey output schema — placeholder for now.** Shape deferred. |
| 6 | `classifier_name → classifier_id` | **Read-only DB lookup at startup**, cached. DB `classifier` is the authority; ids are never hardcoded. |

### Model and mapper live in `alerce_classifiers`

`alerce_classifiers` is a git submodule (`.gitmodules`: `alerce_classifiers` →
`alercebroker/alerce_classifiers`). `SquidwardFeaturesClassifier` and
`SquidwardMapper` live there. **The step owns no model code** — it resolves both
from config via `apf.core.get_class`, exactly as
`stamp_classifier_2025_multisurvey_step/step.py:83` does.

## 3. Module layout

```
lc_classification_multisurvey_step/
├── pyproject.toml
├── Dockerfile
├── README.md
├── .gitignore
├── settings.py                     # env → config dict
├── models_settings.py              # squidward_params + configurator (one entry)
├── scripts/
│   └── run_step.py
├── lc_classification_multisurvey_step/
│   ├── __init__.py
│   ├── step.py                     # LateClassifierMultisurvey(GenericStep)
│   ├── input_dto.py                # messages → features-only InputDTO
│   ├── probabilities.py            # OutputDTO → scribe-ready rows (5 heads)
│   ├── db/
│   │   ├── __init__.py
│   │   └── db.py                   # PSQLConnection + get_classifier_ids_by_name
│   │                               #              + get_taxonomy_by_classifier_id
│   └── output_parser.py            # PLACEHOLDER downstream producer
└── tests/
    ├── __init__.py
    └── unittest/
        ├── test_probabilities.py
        ├── test_input_dto.py
        └── test_taxonomy.py
```

Model wiring in `step.__init__`:

```python
self.mapper = get_class(config["MODEL_CONFIG"]["CLASS_MAPPER"])()
self.model  = get_class(config["MODEL_CONFIG"]["CLASS"])(
    **{"mapper": self.mapper, **config["MODEL_CONFIG"]["PARAMS"]}
)
```

`models_settings.py` keeps a single `squidward_params` entry (mirroring the stamp
step's one-entry `configurator`) so the model class, mapper class and model path
stay env-driven rather than hardcoded:

```python
def squidward_params(model_class: str):
    return {
        "CLASS": model_class,
        "CLASS_MAPPER": os.getenv("CLASS_MAPPER"),
        "PARAMS": {"model_path": os.getenv("MODEL_PATH")},
        "NAME": model_class.split(".")[-1],
        "VERSION": os.getenv("MODEL_VERSION", "2.1.0"),
    }
```

## 4. Data flow

```
feature_step output topic
  │
  ├─ execute(messages)
  │    messages → input_dto.create_input_dto(messages)   # features-only
  │    model.can_predict(dto)  → skip batch if False
  │    model.predict(dto)      → OutputDTO (batched, multi-oid frames)
  │    probabilities.build_probability_rows(dto, lastmjd_by_oid, heads, taxonomy_maps, ...)
  │      → for each of 5 heads: melt by oid → row dicts
  │
  ├─ post_execute → produce_scribe(rows)
  │    one `update-probability` command per row → scribe_multisurvey topic
  │
  └─ pre_produce → output_parser (PLACEHOLDER)
```

**Startup (two reads, in order):**

1. `get_classifier_ids_by_name(HEAD_CLASSIFIER_NAMES)` → `{classifier_name:
   classifier_id}` from the `classifier` table.
2. `get_taxonomy_by_classifier_id` for the ids resolved in (1) →
   `{classifier_id: {class_name: class_id}}`.

Both cached on the step; the connection is opened once and not used again.
Nothing else in the step touches the DB. See §6 for the head→name mapping and §8
for the fail-fast assertions on both reads.

### `oid` is already a bigint

Unlike `stamp_classifier_2025_multisurvey_step` — which starts from raw ZTF alerts
and must call `idmapper.catalog_oid_to_masterid` — the multisurvey `feature_step`
already emits the bigint masterid (see `feature_step/features/utils/parsers.py:521`,
which does `int(oid)` on the same value). The Avro field is typed `string`, so the
step does `int(msg["oid"])` and calls no idmapper.

### Features-only InputDTO

`SquidwardFeaturesClassifier.can_predict` inspects only `input_dto.features`, and
`predict` calls `self.mapper.preprocess(input_dto)` which reads only features. So
detections / non-detections / xmatch / stamps are passed **empty**.

This is deliberate, and follows offline `classify.py`, which documents that it
"sidesteps `lc_classification_step`'s stale candid schema". The legacy
`create_detections_dto` builds a detections frame the model never reads, and
unpickles `extra_fields` bytes to do it. Dropping it removes the stale-schema
coupling and the pickle round-trip.

## 5. What is ported from offline, and what changes

The offline reference is `~/desktop/pipeline/feature_step/features/offline/`.

| Offline symbol | Port | Change |
|---|---|---|
| `probability_writer.CLASSIFIER_IDS` = `[5,6,7,8,9]` | **not ported** | ids come from the DB; the step pins the five *names* instead and resolves ids at startup (decision 6, §6) |
| `probability_writer.classifier_version_to_smallint` | verbatim | `"2.1.0"` → `210`; strips a `_suffix` on the patch part |
| `probability_writer._iter_frames` | verbatim | the 5-head mapping; see §6 |
| `probability_writer.build_probability_rows` | **adapted** | offline is strictly per-oid and raises on a multi-row frame; the step is batched, so melt by oid instead, and log-and-drop rather than raise (§8). Vectorised: one `to_dict("records")` per head, `class_id`/`lastmjd` via `.map()`, following the sibling `format_probability_records`. A per-(oid, head) `to_dict` costs ~3 s per 1000-oid batch. |
| `probability_writer.write_probabilities` | **dropped** | scribe-only (decision 3) |
| `db.fetch_taxonomy_maps` | adapted | same query, via `PSQLConnection` instead of a raw engine; called with DB-resolved ids rather than the literal `[5..9]` |
| `classify.load_squidward_model` | **dropped** | `get_class` from config instead (§2) |
| `classify.features_message_to_dto` | adapted | one row per message instead of one oid |
| `classify.classify_astro_object` | partially | keep `can_predict` → empty-OutputDTO-on-false; drop `parse_output` (feature_step already ran it) |
| `classify._lc_lastmjd` | adapted | max MJD from the message's `detections` array, not from DB |
| `classifier_taxonomy_lut` | **not ported** | the fixture stays the seed authority in the offline repo; the step reads the DB for both the classifier ids and the taxonomy |

The single most important piece the stamp step does **not** have is
`_iter_frames`: `stamp_classifier_2025_multisurvey_step` writes only
`output_dto.probabilities`, i.e. one head. BHRF emits five.

## 6. The five heads

Each head is identified by a **classifier name**, and the frame it reads out of the
`OutputDTO`. The name suffix is structural — it is what makes the head that head —
so the suffixes are pinned; only the base name is configurable.

```python
CLASSIFIER_NAME = os.getenv("CLASSIFIER_NAME", "lc_classifier_BHRF_forced_phot")

def heads(output_dto, base=CLASSIFIER_NAME):
    h = output_dto.hierarchical
    return [
        (f"{base}",             output_dto.probabilities),      # flat, 21 leaves
        (f"{base}_top",         h["top"]),                      # Periodic/Stochastic/Transient
        (f"{base}_transient",   h["children"]["Transient"]),    # 6 classes
        (f"{base}_stochastic",  h["children"]["Stochastic"]),   # 6 classes
        (f"{base}_periodic",    h["children"]["Periodic"]),     # 9 classes
    ]
```

`classifier_id` for each head is then `classifier_ids[name]`, resolved once at
startup from the `classifier` table (§6.1). In the live `multisurvey_ztf` these
happen to be 5–9 (seeded 2026-08-03), but the step never assumes that: the offline
fixture's own comment notes the ids were picked as "next-free after live max 4" and
must be re-verified at apply time, so a different environment — a fresh dev DB, a
restored dump, a deploy that claimed 5+ first — can legitimately allocate other ids
for the same names.

Class names are md5-verified against the deployed pickle; the transient class is
**`SESN`**, not `SNIbc`.

A head whose frame is `None` or empty is skipped, not written.

### 6.1 Resolving the ids

```python
def get_classifier_ids_by_name(
    classifier_names: list[str], psql_connection: PSQLConnection
) -> dict[str, dict]:
    """{classifier_name: {"classifier_id": int, "classifier_version": str}}.

    Read-only, from <schema>.classifier. Raises on a duplicated name (§8
    assertion 2) — that is the only place a duplicate is still visible, since the
    return value is keyed by name.
    """
```

Query, mirroring `get_taxonomy_by_classifier_id`'s shape (`text()` + bound
expanding param, `PSQLConnection.session()`):

```sql
SELECT classifier_id, classifier_name, classifier_version
FROM classifier
WHERE classifier_name IN :names
```

`classifier_name` is not unique in the table's constraints (the PK is
`classifier_id` alone), so two rows sharing a name is possible in principle and is
treated as a deploy error — see §8.

The row's `classifier_version` is read for the startup consistency check in §8, not
for the written value: `probability.classifier_version` stays derived from
`MODEL_VERSION`, which describes the artifact actually loaded from `MODEL_PATH`.

Neither reader swallows exceptions. The stamp step's
`get_taxonomy_by_classifier_id` wraps its query in `try/except` and returns `{}` on
error, which turns a dead DB into an empty map; here both readers let the exception
propagate so a connection failure at startup is distinguishable from a genuinely
unseeded table.

## 7. Probability row contract

`scribe_multisurvey` already accepts exactly this payload — no scribe change is
needed. `parse_probability_table` (`scribe_multisurvey/sql_scribe/sql/command/parser.py:430`):

```
{oid, sid, classifier_id, classifier_version, class_id, probability, ranking, lastmjd}
```

Command envelope, matching `stamp_classifier_2025_multisurvey_step/step.py:272`
and accepted by `decode.command_factory`:

```json
{"step": "update-probability", "survey": "ztf", "payload": { ...row... }}
```

produced as `{"payload": json.dumps(command)}` with the Kafka key set to
`str(oid)`.

`ProbabilityCommand.db_operation`
(`scribe_multisurvey/sql_scribe/sql/command/commands.py:607`) dedups by
`(oid, sid, classifier_id, class_id)` keeping the highest `lastmjd`, then upserts
with `ON CONFLICT (pk_probability_oid_classifierid_classid) DO UPDATE` on
`probability`, `ranking`, `lastmjd`. That is the update-on-reclassify behaviour
this step needs — it matches offline `write_probabilities`, and is **not** the
`DO NOTHING` the stamp step uses.

Field derivations:

- `oid` — `int(msg["oid"])`.
- `sid` — from config `SID`, default `0` (ZTF).
- `classifier_id` — `classifier_ids[head_name]`, from the startup DB lookup (§6.1).
- `classifier_version` — `classifier_version_to_smallint(MODEL_VERSION)` → `210`.
- `class_id` — `taxonomy_maps[classifier_id][class_name]`, exact string match.
- `probability` — the model's value.
- `ranking` — dense rank descending **within (oid, classifier_id)**, i.e. per head.
- `lastmjd` — `max(det["mjd"] for det in msg["detections"])`. Already MJD; **do not**
  subtract 2400000.5. The message's `detections` array carries forced photometry
  too (each entry has a `forced` flag), so this is the max over detections and
  forced together, matching offline `_lc_lastmjd`.

## 8. Error handling

`probability.class_id` is a foreign key. The stamp step's `class_name_to_id`
returns `-1` on a miss, which either violates the FK or stores a garbage class.
Offline instead raises. This step splits the difference by when the problem is
detectable:

- **Startup, fail fast.** Five assertions, all raising and refusing to start. An
  unseeded, partially-seeded, ambiguous or version-skewed `classifier`/`taxonomy`
  is a deploy error, and a step that starts anyway would silently drop every
  probability it produces or write it against the wrong classifier.
  1. `MODEL_VERSION` parses to a non-zero smallint. Checked first, because it
     needs no DB round trip. `classifier_version_to_smallint` returns `0` for
     anything that is not three dot-separated parts, so a `MODEL_VERSION` of
     `dev` — matched by an equally malformed seeded `classifier_version`, which
     passes assertion 5 — would write `classifier_version = 0` on every row for
     the life of the deploy. That is the same silent-garbage failure this section
     exists to prevent, on a different column.
  2. All five head names in §6 resolved to a row. Raise naming the missing ones.
  3. No name resolved to more than one row.
  4. Every resolved id has a non-empty taxonomy map. Raise naming the *heads*,
     not just the ids — ids are DB-resolved and not assumed to be 5–9, so an id
     alone does not tell an operator in a crashloop which head is unseeded.
  5. Each row's `classifier_version` equals `MODEL_VERSION`. A DB row saying
     `2.1.0` while `MODEL_PATH` points at a different artifact means the seeded
     taxonomy may not match the model's `classes_`, which is exactly the
     silent-garbage-class failure the class-name lookup exists to prevent.
     Checked before the taxonomy query, so a version-skewed deploy fails without
     a second round trip. (See §13 — this is the one assertion that may be too
     strict in practice.)

  These messages are the operator interface for a step that will not start, so
  each names the offending value, states what the consequence would have been,
  and ends with "Refusing to start."

  These replace the `-1`-on-miss behaviour rather than layering on it: with the ids
  themselves coming from the DB, a name that does not resolve has no safe fallback.
- **Per batch, skip and log.** If a class name coming out of the model is absent
  from its head's map, log an error identifying the classifier id and class name,
  and drop **that whole head for the batch**. Do not emit `class_id = -1`, and do
  not kill the batch — one model/taxonomy drift should not stop the consumer.

  The granularity is the head, not the oid: class names are the *columns* of a
  head's frame, so an unknown name is a frame-wide model/taxonomy drift that is
  identical for every oid in the batch. Checking per-oid would recompute the same
  answer once per oid and emit one near-identical error line per oid — thousands
  of them during exactly the incident the log exists to diagnose. Check once per
  head, log once.

  An oid missing from the `lastmjd` map *is* per-oid (`probability.lastmjd` is
  NOT NULL), so those rows drop individually, with the affected oids named in one
  log line per head. An oid whose mapped `lastmjd` is NaN drops the same way —
  the column is NOT NULL, so a NaN is as unusable as a missing key, and the
  vectorised `.isna()` check treats them alike deliberately.
- **`can_predict` false** → produce nothing, log, return an empty OutputDTO,
  matching offline `classify_astro_object` and the legacy step.
- **Messages with no features** (`msg["features"]` is `None`) are filtered out
  before the DTO is built.
- **Messages whose `oid` will not parse as an integer** are dropped and logged,
  one aggregated warning per batch naming the offending raw values. This is the
  one input-shape guard the step carries, and it exists because `oid` is the one
  field where Avro is weaker than the code's assumption: the field is typed plain
  `string` (§7), and Avro cannot constrain a string to digits, so validity rests
  on a producer convention rather than on the wire format. Every other field the
  step reads — `detections`, its `mjd`, `features` — is pinned by the schema
  (`array`, non-nullable `double`, and a fixed 209-field record), so shape guards
  there would defend against messages the deserializer already rejects, and the
  step deliberately carries none.

  Note this risk is *introduced* by the multisurvey port rather than inherited:
  the legacy `lc_classification_step` never casts `oid`, indexing its frame by
  the raw string. The cast is new here because multisurvey oids are bigints
  (§7). Without the guard, one malformed oid raises inside `int()` and kills the
  whole batch — and on a Kafka consumer it re-raises on every redelivery, so the
  partition stalls rather than merely losing a message.
- **Detections whose `mjd` is non-finite** (NaN or ±inf) are skipped when
  `lastmjd` is computed; an oid left with no usable mjd is dropped, again with
  one aggregated warning per batch. NaN matters because `max()` is order-
  sensitive with it (`max(nan, x)` is `nan`, `max(x, nan)` is `x`), which would
  otherwise make the result depend on detection ordering. `inf` matters more:
  a NaN `lastmjd` is caught downstream by the `.isna()` check above, but an
  `inf` is a valid `double precision` value that Postgres accepts and that then
  wins the scribe's highest-`lastmjd` dedup permanently — no later, correct
  message could ever displace it.
- **Duplicate oids in one batch** are collapsed in a single pass that picks one
  winning message per oid — the last, by arrival order — from which *both* the
  features frame and the `lastmjd` map are derived. Two messages for the same
  object can land in a single consume batch; left alone they produce two rows
  colliding on `(oid, sid, classifier_id, class_id)`. The stamp step does the
  same (`df[~df.index.duplicated(...)]`), as does the legacy step
  (`drop_duplicates("oid", keep="last")`). `build_probability_rows` therefore
  documents a unique-oid-index contract rather than re-checking.

  Last-wins is correct because `feature_step` produces keyed by `str(oid)`, so
  same-oid messages land on one partition in offset order and the last really is
  the newest. (It is *not* because duplicates carry the same `lastmjd` — they
  normally do not, being updates with different detection sets.)

  The collapse must happen once, not once per derived structure. Computing the
  frame and the `lastmjd` map in two independent passes lets them disagree: if
  the winning message carries an empty `detections` list — legal, the field
  defaults to `[]` — a frame-side "last wins" rule pairs the winner's features
  with the *loser's* timestamp. Deriving both from one collapsed mapping removes
  that class of bug by construction rather than by keeping two rules in sync.

## 9. Placeholder downstream output

Decision 5 is "new multisurvey output schema", but its shape is deferred. For now:

- `output_parser.py` holds a `MultisurveyOutputParser` whose `parse` returns a
  minimal per-oid message — `{oid, classifier_name, classifier_version}` plus the
  top-ranked class per head — and is marked `# PLACEHOLDER` with a pointer to this
  section.
- No `schemas/lc_classification_multisurvey_step/` Avro file is added yet.
- The producer is configured but the shape is not treated as a contract; nothing
  downstream should be pointed at it until the schema is designed.

Deferring this is safe because decision 3 makes the scribe the real output path.

## 10. Configuration

| Env var | Purpose | Default |
|---|---|---|
| `CONSUMER_TOPICS`, `CONSUMER_SERVER`, `CONSUMER_GROUP_ID` | feature_step output topic | required |
| `MODEL_CLASS` | `alerce_classifiers.squidward.model.SquidwardFeaturesClassifier` | required |
| `CLASS_MAPPER` | `alerce_classifiers.squidward.mapper.SquidwardMapper` | required |
| `MODEL_PATH` | S3 url of `squidward/2.1.0/hierarchical_random_forest_model.pkl` | required |
| `MODEL_VERSION` | version string → smallint; checked against `classifier.classifier_version` | `2.1.0` |
| `CLASSIFIER_NAME` | base classifier name; the five head names derive from it (§6) | `lc_classifier_BHRF_forced_phot` |
| `SCRIBE_SERVER`, `SCRIBE_TOPIC` | scribe_multisurvey topic | required |
| `PSQL_*` / `SCHEMA` | startup `classifier` + `taxonomy` reads | required, schema `multisurvey_ztf` |
| `SID` | survey id written into `probability.sid` | `0` |
| `MIN_DETECTIONS` | optional pre-classification filter (§13) | unset |

Classifier ids are **not** configurable and **not** constants — they are read from
the `classifier` table at startup (decision 6). The head name *suffixes*
(`_top`, `_transient`, `_stochastic`, `_periodic`) are pinned, because they are
positional against the model's hierarchical output; only the base
`CLASSIFIER_NAME` is env-driven. Making the ids an env var would let config and DB
drift silently, and hardcoding them assumes an id allocation the DB never
promised.

## 11. Testing

Pure-function unit tests, following the offline test layout
(`tests/unittest/test_offline_*.py`):

- `test_probabilities.py` — the 5-head split; melt-by-oid over a multi-oid frame;
  per-head dense ranking; version → smallint; a missing/empty head is skipped;
  an unknown class name drops that head's rows for that oid and logs rather than
  raising; `lastmjd` taken as the max over detections without JD subtraction.
- `test_input_dto.py` — features-only DTO; `oid` cast to int; messages with
  `features: None` filtered; empty batch → empty DTO.
- `test_taxonomy.py` — `get_classifier_ids_by_name` and
  `get_taxonomy_by_classifier_id` map construction against a mocked session; the
  head names derived from a non-default `CLASSIFIER_NAME`; rows returned in a
  different order than requested still map correctly; and each of the four §8
  startup assertions fires — a missing name, a duplicated name, an empty taxonomy
  map, a `classifier_version` mismatch. Ids in the fixtures are deliberately
  **not** 5–9, so a reintroduced hardcode fails the test.

**Equivalence test against the offline reference.** The step's row builder and
offline `probability_writer.build_probability_rows` must agree. For a handful of
real OIDs, run the offline classifier, feed the same `OutputDTO` through both row
builders, and assert the row sets are identical modulo ordering. Offline hardcodes
`CLASSIFIER_IDS = [5..9]`, so the comparison holds only when the target DB actually
allocated those ids — the test asserts that precondition explicitly (via
`get_classifier_ids_by_name`) and skips with a clear message otherwise, rather than
failing on an id mismatch that is not a port defect. This is the test
that actually protects the port — the unit tests above only check the port's
internal consistency. It needs the `alerce_classifiers` submodule initialised and
`MODEL_PATH` set, so it is marked as an opt-in integration test, not part of the
default unit run.

## 12. Out of scope

- Changes to `scribe_multisurvey` — its `ProbabilityCommand` already fits.
- Changes to the legacy `lc_classification_step`.
- The downstream output schema (§9).
- Seeding `classifier` / `taxonomy` rows — already applied to live `multisurvey_ztf`.
  Note this is now a hard startup dependency, not just a write-time one: an
  unseeded DB makes the step refuse to start (§8). Any environment the step runs
  in — including local and CI-integration DBs — must have the seed applied.
- Back-porting the seeds to the db-plugins authority file — tracked separately in
  the offline repo's `FLOW.md` §7.
- LSST / Rubin models.
- Helm chart (`charts/lc_classification_multisurvey_step/`) — deployment is a
  follow-up once the step runs locally.

## 13. Open questions

- **`MIN_DETECTIONS`.** The legacy step filters objects below a detection count
  before classifying. Since features are already computed upstream, it is not
  obvious this step should re-filter. **Resolved for now:** carry the config knob,
  default it to unset — every message that has features gets classified. Counts
  non-forced detections only, as the legacy step does. Revisit if the model
  produces junk on sparse light curves.
- **How strict should the version check be?** §8 assertion 4 refuses to start when
  `classifier.classifier_version` differs from `MODEL_VERSION`. That catches a real
  failure mode (model artifact and seeded taxonomy out of sync), but it also means
  a model bump becomes a two-step deploy: seed the new `classifier` rows first, or
  the step will not come up. The alternative is to log an error and continue,
  since `probability.classifier_version` is written from `MODEL_VERSION` either
  way and the ids/taxonomy are still valid. **Leaning fail-fast** for consistency
  with the rest of §8, but this is the one assertion worth revisiting once the
  deploy story for model bumps is settled.
- **Batch size vs. `classify_batch`.** The model classifies the whole batch in one
  call. Whether the feature_step consume batch size is a good predict batch size
  has not been measured.
