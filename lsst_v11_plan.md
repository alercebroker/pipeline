# Plan: migrate the multisurvey pipeline from LSST schema v10 to v11

## Context

The Rubin/LSST team is bumping their alert schema from v10.0 to v11.0. ALeRCE consumes
LSST alerts in the multisurvey pipeline. This plan describes what needs to change so the
pipeline keeps decoding LSST alerts after Rubin's hard-cutover to v11.

Source of truth for v11 schemas:
[../alert_packet/python/lsst/alert/packet/schema/11/0/](../alert_packet/python/lsst/alert/packet/schema/11/0/).

## Schema diff (v10 → v11)

The change is unusually small. After ignoring doc-string and key-ordering noise, the only
meaningful structural changes between v10 and v11 are field-type tightenings: ~22 integer
fields went from `["null","int"]` (nullable, default `null`) to plain `int`. No fields were
added, removed, or renamed.

Affected fields:

- `diaSource`: `bboxSize`, `dipoleNdata`, `psfNdata`, `trailNdata`
- `diaObject`: `{u,g,r,i,z,y}_psfFluxNdata` (6 fields)
- `ssObject`: `{u,g,r,i,z,y}_nObs`, `{u,g,r,i,z,y}_nObsUsed` (12 fields)

`diaForcedSource`, `ssSource`, `mpc_orbits` are unchanged structurally. The wrapping
`alert.avsc` only changes its namespace string `lsst.v10_0` → `lsst.v11_0`.

### Wire-compatibility implication

This is **not** transparent on the wire. The Avro union tag for the affected fields is part
of the encoded byte stream (`null`-tag vs. `int`-tag), so a v10 reader cannot decode v11
bytes for those fields. Every consumer must move to a v11 reader schema in lock-step with
Rubin's switch.

Cutover model: **hard switch** (per discussion). Rubin will simultaneously move to a
new Kafka broker URL and to schema v11; no parallel-version operation. Date TBD.

## Where v10 is wired into the pipeline today

| Surface | File | Notes |
|---|---|---|
| Avro reader schemas | [schemas/surveys/lsst_v10.0/](schemas/surveys/lsst_v10.0/) | 7 avsc files — consumed at runtime via Helm `SCHEMA_PATH`. |
| Pandas-dtype generator | [ingestion_step/scripts/generate_pd_schemas.py](ingestion_step/scripts/generate_pd_schemas.py) | Lines 91-96 hardcode `lsst_v10.0/*.avsc` paths. |
| Generated dtype map | [ingestion_step/ingestion_step/lsst/schemas.py](ingestion_step/ingestion_step/lsst/schemas.py) | Auto-generated from the above. |
| LSST ingestion strategy | [ingestion_step/ingestion_step/lsst/](ingestion_step/ingestion_step/lsst/) (`extractor.py`, `strategy.py`, `transforms.py`) | Version-agnostic; just consumes `schemas.py`. |
| Avro parser library | [libs/lsst_schema_parser/](libs/lsst_schema_parser/) | Has only `tests/unit/test_v9_v10_parser.py`. Library code itself appears version-agnostic. |
| Correction step (LSST strategy) | [correction_multisurvey_step/core/](correction_multisurvey_step/core/) | Behavior version-agnostic; only stale `# Omitting in schema v10.0` comments to clean up. |
| Test fixtures | [ingestion_step/tests/unittest/lsst/](ingestion_step/tests/unittest/lsst/) | Likely encoded with a v10 writer; need re-encoding under v11. |
| Stale references unrelated to v10→v11 | [ingestion_step/generator/main.py:15](ingestion_step/generator/main.py#L15) | Still references v9.0. Pre-existing staleness — out of scope. |

### `rubin_stamp_classifier_step`

The production ConfigMap (namespace `multisurvey-rubin-stamp-classifier-db1-20260409-step`)
confirms the live consumer config:

```yaml
CONSUMER_CONFIG:
  CLASS: apf.consumers.KafkaSchemalessConsumer
  SCHEMA_PATH: /schemas/surveys/lsst_v10.0/lsst.v10_0.alert.avsc
  TOPICS: [lsst]
```

Two implications:

1. The step currently reads v10 alerts (not v7.4 — the v7.4 references in the repo are
   doc/test residue only).
2. `KafkaSchemalessConsumer` uses the path-supplied schema as the *only* reader schema —
   no per-message writer-schema lookup. When Rubin switches to v11, this overlay's
   `SCHEMA_PATH` must change in lock-step or decoding fails on the 22 affected fields.

The committed [charts/rubin_stamp_classifier_step/values.yaml](charts/rubin_stamp_classifier_step/values.yaml)
does not pin a consumer `SCHEMA_PATH`; the value above is supplied by an uncommitted
production overlay.

Independent of the v10→v11 schema cutover, the step's source code carries v7.4-era
residue in
[rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py](rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py):

- Lines 74-79: log warnings stating that `airmass`, `magLim`, `scienceFlux/Err`, `seeing`
  are "not available in schema v7.4".
- Lines 115-124 (inside the `elif obj_ok ^ src_ok` branch — i.e., the only branch that
  builds a `processed_message` and appends): hardcodes
  `airmass=1.0`, `magLim=25.0`, `scienceFlux=0.0`, `scienceFluxErr=0.0`, `seeing=0.7`,
  while reading `psfFlux`, `psfFluxErr`, `snr` from the alert.

I checked v11: of those five hardcoded fields, **`scienceFlux` and `scienceFluxErr` exist
in v11 `diaSource`** (and have existed since at least v10 — i.e., they are present in
the bytes the step is decoding today). `airmass`, `magLim`, `seeing` do not exist in v11.

> ⚠️ **NEEDS VERIFICATION BY DOMAIN OWNER.** I'm reading the code as: the
> `processed_message["scienceFlux"] = 0.0` assignment runs unconditionally for every
> alert that reaches the message-appending branch — there is no read-from-alert and no
> "missing field" guard. The production reader being v10 (which contains both fields)
> means the v10-era reasoning that `scienceFlux` was unavailable does not apply. If my
> reading is correct, the step is silently zeroing real data on every LSST alert today.
> **Please confirm before this plan moves to implementation.**

The other v7.4 references in this step are dev-only (not production code paths):
[scripts/produce_lsst_topic.py](rubin_stamp_classifier_step/scripts/produce_lsst_topic.py),
[tests/integration/data/load_msgs_sample.py](rubin_stamp_classifier_step/tests/integration/data/load_msgs_sample.py),
[tests/integration/test_kafka_output.py](rubin_stamp_classifier_step/tests/integration/test_kafka_output.py),
and the README.

## Plan

### Phase 1 — drop in the v11 schemas

1. Create [schemas/surveys/lsst_v11.0/](schemas/surveys/lsst_v11.0/) and copy the seven
   v11 avsc files from
   [../alert_packet/python/lsst/alert/packet/schema/11/0/](../alert_packet/python/lsst/alert/packet/schema/11/0/):
   `lsst.v11_0.alert.avsc`, `lsst.v11_0.diaSource.avsc`, `lsst.v11_0.diaForcedSource.avsc`,
   `lsst.v11_0.diaObject.avsc`, `lsst.v11_0.ssSource.avsc`, `lsst.v11_0.ssObject.avsc`,
   `lsst.v11_0.mpc_orbits.avsc`.
2. Update [ingestion_step/scripts/generate_pd_schemas.py](ingestion_step/scripts/generate_pd_schemas.py)
   lines 91-96 to point at `../schemas/surveys/lsst_v11.0/lsst.v11_0.*.avsc`.
3. Re-run the generator to regenerate
   [ingestion_step/ingestion_step/lsst/schemas.py](ingestion_step/ingestion_step/lsst/schemas.py).
   - **Verify:** the diff in `schemas.py` only changes dtypes for the 22 affected fields
     (nullable Int → non-nullable Int / object → int) and nothing else. Anything else
     is a regression in the generator and must be investigated.
4. Keep `lsst_v10.0/` in tree for now — replay tooling and `s3_multisurvey_step` may
   reference it. Schedule its removal for a follow-up after a full LSST cutover cycle.

### Phase 2 — verify nothing downstream breaks on the type tightening

5. Grep the LSST-handling code for the 22 affected field names (`bboxSize`,
   `dipoleNdata`, `psfNdata`, `trailNdata`, `*_psfFluxNdata`, `*_nObs`, `*_nObsUsed`).
   Confirm no code path treats `None` as semantically meaningful for any of them. They
   are counters (always populated when present), so `None`-handling is most likely
   defensive and safe to leave in place even after the type change.
6. Re-encode any v10-encoded test fixtures under
   [ingestion_step/tests/unittest/lsst/](ingestion_step/tests/unittest/lsst/) and
   [correction_multisurvey_step/tests/](correction_multisurvey_step/tests/) using v11.
   Sample alerts are at
   [../alert_packet/python/lsst/alert/packet/schema/11/0/sample_data/](../alert_packet/python/lsst/alert/packet/schema/11/0/sample_data/).
7. Add a v11 case to
   [libs/lsst_schema_parser/tests/unit/test_v9_v10_parser.py](libs/lsst_schema_parser/tests/unit/test_v9_v10_parser.py)
   (rename to `test_lsst_parser.py` or add `test_v11_parser.py`). Since the v11 field set
   is a structural superset/equivalent of v10, this is mostly a sanity check.
8. Run the full ingestion + correction + magstats unit-test suite for the LSST strategy
   and confirm green.

### Phase 3 — clean up stale v10 mentions

9. Update or remove the `# Omitting in schema v10.0` and similar comments in:
   - [correction_multisurvey_step/core/parsers/input_message_parsing/lsst_input_parser.py](correction_multisurvey_step/core/parsers/input_message_parsing/lsst_input_parser.py)
     (lines 143, 167, 184, 200-201, 242, 268-269, 287-288, 299-300)
   - [correction_multisurvey_step/core/parsers/survey_data_join/lsstDataJoiner.py](correction_multisurvey_step/core/parsers/survey_data_join/lsstDataJoiner.py)
     (lines 39, 151)
   - [correction_multisurvey_step/core/DB/lsst_database_strategy.py](correction_multisurvey_step/core/DB/lsst_database_strategy.py)
     (lines 49, 316)
   - [correction_multisurvey_step/core/schemas/LSST/LSST_schemas.py](correction_multisurvey_step/core/schemas/LSST/LSST_schemas.py)
     (line 57)

   Either say "v11.0" or just "current LSST schema". Do not change behavior in this phase.

### Phase 4 — `rubin_stamp_classifier_step`

> Pre-condition: confirm whether the v7.4-era hardcoded defaults in `pre_execute` are
> unconditional (my reading of step.py:121-122) or fallback-only (alternate reading).
> Production reader is confirmed v10 → step 13 below is mandatory regardless.

10. If the `scienceFlux=0.0` / `scienceFluxErr=0.0` assignments at
    [step.py:121-122](rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py#L121-L122)
    are unconditional, replace them with reads from the alert:
    ```python
    processed_message["scienceFlux"] = message["diaSource"]["scienceFlux"]
    processed_message["scienceFluxErr"] = message["diaSource"]["scienceFluxErr"]
    ```
    Keep `airmass=1.0`, `magLim=25.0`, `seeing=0.7` — these fields are still absent in v11.
11. Update or delete the v7.4-era log warnings at
    [step.py:74-79](rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py#L74-L79).
    Keep warnings only for fields still missing in v11 (airmass, magLim, seeing).
12. Update test fixtures and the local-dev producer:
    - [tests/integration/data/load_msgs_sample.py:17](rubin_stamp_classifier_step/tests/integration/data/load_msgs_sample.py#L17)
    - [tests/integration/test_kafka_output.py](rubin_stamp_classifier_step/tests/integration/test_kafka_output.py)
      (lines 44, 49, 56, 144, 246, 251)
    - [scripts/produce_lsst_topic.py](rubin_stamp_classifier_step/scripts/produce_lsst_topic.py)
      (lines 40-42)
    - [README.md](rubin_stamp_classifier_step/README.md) (lines 29, 120)

### Phase 5 — deployment

#### Live LSST topology (confirmed from production ConfigMaps)

```
external Rubin broker (usdf-alert-stream-dev.lsst.cloud:9094)
  topic: lsst-alerts-v10.0
      │
      ▼
reflector_step (LSST)        ns: multisurvey-reflector-lsst-db1-v10-step
                             consumer: KafkaSchemalessRegistryConsumer
                             SCHEMA_PATH (in/out): lsst_v10.0/...alert.avsc
                             producer SCHEMA_PATH: same
      │
      ▼
internal broker (192.168.50.106:29092)
  topic: lsst
      │
      ├──→ ingestion_step (LSST)             group.id: lsst-ingestion
      │    ns: multisurvey-ingestion-lsst-db1-step
      │    consumer: KafkaSchemalessConsumer, SCHEMA_PATH: lsst_v10.0/...alert.avsc
      │    producer: KafkaSchemalessProducer → topic lsst-ingestion (internal output.avsc)
      │
      └──→ rubin_stamp_classifier_step       group.id: lsst-ingestion-rubin-stamp-20260421
           ns: multisurvey-rubin-stamp-classifier-db1-20260409-step
           consumer: KafkaSchemalessConsumer, SCHEMA_PATH: lsst_v10.0/...alert.avsc
```

Cutover is a **hard switch**: at the announced time, Rubin simultaneously moves to a
new Kafka broker URL **and** to schema v11. No parallel-version operation is required
or desired.

#### Cutover targets

Three deployments must flip simultaneously. The remaining LSST deployments (correction,
magstats) consume ALeRCE-internal topics and need no edit.

| # | Namespace | Component | Required edits |
|---|---|---|---|
| A | `multisurvey-reflector-lsst-db1-v10-step` | `reflector_step` | `CONSUMER_CONFIG.PARAMS.bootstrap.servers` (new Rubin broker URL), `CONSUMER_CONFIG.SCHEMA_PATH` (v11), `CONSUMER_CONFIG.TOPICS` (`lsst-alerts-v11.0`), `PRODUCER_CONFIG.SCHEMA_PATH` (v11). Also: SASL credentials likely change with the new broker — confirm with Rubin. |
| B | `multisurvey-ingestion-lsst-db1-step` | `ingestion_step` (LSST) | `CONSUMER_CONFIG.SCHEMA_PATH` (v11) |
| C1 | `multisurvey-rubin-stamp-classifier-db1-v10-step` | `rubin_stamp_classifier_step` (model variant 1) | `CONSUMER_CONFIG.SCHEMA_PATH` (v11) |
| C2 | `multisurvey-rubin-stamp-classifier-db1-20260409-step` | `rubin_stamp_classifier_step` (model variant 2) | `CONSUMER_CONFIG.SCHEMA_PATH` (v11) |

> Both stamp-classifier deployments run the same image with different ML model
> configs. They have distinct `group.id`s (the older one's group.id has not been
> captured here; for the dated one it is `lsst-ingestion-rubin-stamp-20260421`),
> so they are independent consumer groups on the `lsst` topic.

Steps that need **no edit** (verified from live ConfigMaps; all consume internal
ALeRCE schemas, not Rubin's):

| Namespace | Component | Reads | Reader schema |
|---|---|---|---|
| `multisurvey-correction-lsst-db1-step` | `correction_multisurvey_step` (LSST) | `lsst-ingestion` | `/schemas/ingestion_step/lsst/output.avsc` |
| `multisurvey-magstats-lsst-db1-step` | `magstats_multisurvey_step` (LSST) | `lsst-correction` | `/schemas/correction_ms_step/lsst/output.avsc` |
| `multisurvey-rubin-features-db1-step` | `rubin_features` (LSST) | `lsst-magstats` | `/schemas/magstats_ms_step/lsst/output.avsc` |

No LSST `s3_multisurvey_step` namespace exists — not deployed for LSST today.


The reflector's `bootstrap.servers` is currently `usdf-alert-stream-dev.lsst.cloud:9094`
producing to topic `lsst-alerts-v10.0`. Both change at cutover.

#### Steps

13. **Pre-cutover (any time):** bake new images for `reflector_step`, `ingestion_step`,
    and `rubin_stamp_classifier_step` containing the v11 schema files (Phase 1) and
    rebuilt pandas dtype map (Phase 1 step 3). Push to the registry. The same images
    must also still contain the v10 schemas (Phase 1 step 4) for trivial rollback.
14. **Pre-cutover (any time):** prepare the three v11 ConfigMaps in dry-run form.
    Confirm the new Rubin broker URL, topic name (`lsst-alerts-v11.0` is the
    expected pattern), and SASL credentials with the Rubin team well in advance.
15. **Cutover instant (Rubin's announced switch time):**
    - Apply the three v11 ConfigMaps.
    - `kubectl rollout restart deployment …` for each of the three so the new
      images + ConfigMaps are picked up.
    - Verify in this order: (a) reflector consumer reaches the new broker and starts
      receiving from `lsst-alerts-v11.0`; (b) reflector produces to internal `lsst`
      topic and lag is healthy; (c) ingestion resumes PSQL writes; (d) stamp
      classifier resumes producing probabilities.
    - Watch correction and magstats lag on `lsst-ingestion` and `lsst-correction`
      respectively — they should be unaffected, but a problem in ingestion or
      correction will manifest as growing lag here.
16. **Rollback path:** keep the v10 schema files committed (Phase 1 step 4) and the
    v10 image tags in the registry. Reverting any of A/B/C is a ConfigMap edit +
    rollout restart back to the v10 image tag. Note that rollback is only useful
    if the issue is on our side — Rubin's old broker may already be offline at
    that point.
17. **Update [PIPELINE.md](PIPELINE.md):**
    - The stamp-classifier section says "Avro v7.4" — correct to v10 today, v11
      post-cutover. Remove the "deployment not yet confirmed" caveat — live ConfigMap
      proves it is deployed.
    - The LSST flow note "lsst-ingestion (commented in chart values)" is stale: the
      ingestion deployment exists in production via an uncommitted overlay. Consider
      committing a sanitised overlay under [charts/multisurvey_step/](charts/multisurvey_step/)
      so future changes are reviewable. Same for the reflector and stamp classifier.
    - PIPELINE.md lists `feature_step` under "legacy steps with no multisurvey
      equivalent." A `rubin_features` LSST step is in fact deployed in
      `multisurvey-rubin-features-db1-step`, consuming `lsst-magstats`. Update the
      doc to reflect that an LSST features step exists (at least in this `db1`
      cluster). Confirm whether this is the canonical multisurvey successor to
      `feature_step` or a Rubin-specific variant.

### Out of scope (flagged, not bundled)

- [ingestion_step/generator/main.py:15](ingestion_step/generator/main.py#L15) still
  references `lsst_v9.0` — pre-existing staleness, separate ticket.
- Removal of `schemas/surveys/lsst_v8.0/`, `lsst_v9.0/`, `lsst_v10.0/`, and `lsst/`
  (the v7.4 directory) from the repo — wait until after Rubin's cutover is fully
  observed in production.

## Open questions to resolve before implementation

1. **Cutover date.** Confirm the calendar date Rubin will switch from v10 to v11.
2. **New Rubin broker details.** Confirm the new bootstrap URL, the v11 topic name
   (assumed `lsst-alerts-v11.0`), and any SASL credential changes with the Rubin team.
3. **Confirmation on `scienceFlux=0.0` semantics.** Is the assignment at step.py:121-122
   unconditional (my reading), or is there logic I'm missing that only triggers it on
   alerts missing the field? Production has been reading v10 (which carries
   `diaSource.scienceFlux`), so if my reading is correct this has been silently zeroing
   real data. If the latter, Phase 4 step 10 is unnecessary.
