# Legacy ZTF xmatch step: xmatches coming out empty since 2025-11-04

**Date of report:** 2026-07-16
**Author:** Francisco Andrades (investigation assisted by Claude)
**Affected component:** `xmatch_step` (legacy ZTF pipeline), image `ghcr.io/alercebroker/xmatch_step:27.5.7a32.dev1`

## Summary

The legacy ZTF xmatch step appears to have produced no crossmatch results since the 2025-11-04
deployment: every message we sampled from the `xmatch` Kafka topic carries `xmatches: null`, and
we found no rows in the legacy `xmatch` table for objects first detected after that date. The
step itself runs fine and shows no errors. Our leading hypothesis (not yet confirmed against the
running deployment — see "Likely root cause") is a response-format mismatch between the
`XwaveClient` in that image and the xwave conesearch service, which would end up looking exactly
like "no counterpart found" for every object. Whatever the cause turns out to be, it was easy to
miss: nothing crashes, nothing is logged, and the pipeline keeps flowing normally.

## Evidence

### 1. The Kafka topic: 100% null xmatches in a ~100k-message sample

We consumed **100,035 messages** from the start of the `xmatch` topic on the Quimal Kafka cluster
(the broker used by the legacy ZTF pipeline; read with a throwaway consumer group, no offsets
committed, decoded with `schemas/xmatch_step/output.avsc`). The messages sampled have producer
timestamps of 2026-07-08 and 2026-07-09. This is a sample of the topic, not the whole thing — but
the result is uniform:

| producer date | messages | with `xmatches` | missing rate |
|---|---|---|---|
| 2026-07-08 | 54,838 | 0 | 100% |
| 2026-07-09 | 45,197 | 0 | 100% |

Not a single message in the sample carried a crossmatch.

### 2. The database: no xmatch rows for objects first seen after Nov 4

Context: these checks ran against the legacy ZTF database — PostgreSQL on
`quimal-db1.alerce.online`, database `ztf`, schema `alerce` — i.e. tables `alerce.xmatch` and
`alerce.object`. All queries were read-only.

`alerce.xmatch` has no timestamp columns, so we tested recency indirectly with primary-key
probes: sample 1,000 oids per discovery cohort (selected via the indexed `alerce.object.firstmjd`
column), and count how many have any row in `alerce.xmatch`.

| cohort (first detection) | sampled | with xmatch rows |
|---|---|---|
| ZTF18/ZTF19 (old objects, control) | 1,000 | 887 |
| 2025-10-04 → 2025-11-03 (pre-deploy) | 1,000 | 102 |
| **2025-11-04 → 2025-12-16 (post-deploy)** | 1,000 | **0** |
| ZTF26 (2026 objects) | 1,000 | **0** |

Objects discovered up to Nov 3 have crossmatch rows; in our samples, objects discovered from
Nov 4 onward have none. The cutoff coincides with the deployment date.

Note: the newer `multisurvey.xmatch` table (same host, database `ztf`, schema `multisurvey`) does
show recent `created_date` activity, but that table is written by the multisurvey pipeline
(`scribe_multisurvey`), not by the legacy step, so it says nothing about this outage.

### 3. The deployed image

- `ghcr.io/alercebroker/xmatch_step:27.5.7a32.dev1` — built **2025-11-04 18:10 UTC** (per the
  image config in the registry).
- `ghcr.io/alercebroker/feature_step:27.5.7a32.dev1` — built **2025-11-05 14:57 UTC** (same
  release train).
- The `xmatch_step` code inside the image (inspected by pulling its layers from ghcr) is
  byte-identical to the current `pipeline` repo working tree (`xwave_client/client.py`,
  `step.py`), so the behavior analyzed below is what runs in production.

## Likely root cause

The step runs with `USE_XWAVE: true` against the xwave conesearch service
(`http://quimal-db1.alerce.online:8081`). The service itself works and finds counterparts —
queried directly for a sample object (`ZTF19abawuwv`, ra 266.5415851, dec 1.0096207), it returns
its AllWISE match at 0.044″. But it responds in a catalog-wrapper format:

```json
[{"catalog": "allwise", "data": [{"id": "2662p015_ac51-003129", "ra": ..., "dec": ..., "distance": 0.0443}]}]
```

while the `XwaveClient` in the image (written Feb 2025) expects a flat list of sources with an
`ID` key. The chain (paths relative to `xmatch_step/xmatch_step/` in the `pipeline` repo):

1. `process_single_coordinate` iterates the response, so each `entry` is the *wrapper* object,
   not a source (`core/xwave_client/client.py:126`).
2. `process_metadata` reads `entry["ID"]` → `KeyError` on every hit — the id actually lives at
   `entry["data"][0]["id"]` (`client.py:138`).
3. The `except` blocks absorb the exception (`client.py:114`, `client.py:191`), each match is
   dropped, and the client returns a schema-correct empty DataFrame (`client.py:97-98`).
4. To the step this is indistinguishable from "zero counterparts": no retry, no error message.
   Every output message gets `xmatches: null` and `produce_scribe` has nothing to write
   (`step.py:74-75`).

There's a second format difference behind the first: the service's `/v1/metadata` endpoint
returns nullable-float wrappers (`"w1mpro": {"Float64": 13.78, "Valid": true}`) where the client
expects flat scalars — worth handling in the same fix.

What we checked so far: running the CDS-client path for the same object attaches the full xmatch
correctly, and replaying the current service response through the XwaveClient parsing logic
reproduces the silent `KeyError` → empty result. What remains unconfirmed is the behavior of the
actual running deployment: we haven't observed the pod itself, and we don't know the service
side of the timeline — whether the response format changed at some point (and when) or was
always this shape for this endpoint. Ways to confirm: run the deployed image against the service
and watch it return empty for objects with known counterparts, check the xwave service's access
logs / deployment history around Nov 2025, or deploy the client fix and verify the hit-rate
recovers.

## Impact

- Based on our samples, no AllWISE crossmatches for ZTF objects processed since 2025-11-04 —
  on the `xmatch` topic, in `alerce.xmatch`, and downstream.
- The feature step has been receiving `xmatches: null`, so AllWISE color features (W1−W2, W2−W3,
  …) have been missing for the period, which also affects classifiers that use them.
- Hard to spot in monitoring: pods healthy, lag draining, no errors — the failure mode looks
  identical to "no object has a counterpart".

## Related (pre-existing) observation

The orphan `cat_oid`s that started this investigation (rows in the xmatch table whose
`oid_catalog` has no row in the corresponding `allwise` detail table) are a separate, older gap:
since commit `ade93db46` (Jan 2024, "only write xmatch result") the scribe command carries only
`{catoid, dist}`, and nothing currently writes the `allwise` detail table. The full catalog
record is still assembled in `produce_scribe` but no longer sent (the `allwise` variable at
`step.py:78-79` is now unused) — so the data is within easy reach if we want to restore that
write.

## Suggested next steps

1. Update `XwaveClient` to the current service format: iterate `wrapper["data"]`, read lowercase
   `id`/`ra`/`dec`/`distance`, unwrap `{Float64, Valid}` metadata values.
2. Consider making service failures visible — e.g. raising on non-200 responses like the CDS
   client does, or at least logging and counting them — so a format drift surfaces quickly
   instead of reading as "no matches".
3. Build and deploy a new image (the behavior is baked into `27.5.7a32.dev1`; no config change
   can work around it).
4. Evaluate a backfill for the Nov 2025 → present gap (re-run xmatch over objects with
   `firstmjd > 60983`, or replay input topics where retained).
5. A cheap guardrail for the future: a metric on crossmatch hit-rate (matches / messages) — a
   sustained 0% is a clear signal something upstream changed.
