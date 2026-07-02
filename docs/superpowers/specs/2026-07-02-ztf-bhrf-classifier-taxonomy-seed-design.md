# Seed the BHRF hierarchical LC classifier + taxonomy into `multisurvey_ztf`

**Date:** 2026-07-02
**Status:** Approved design, pending implementation plan
**Related:** `docs/superpowers/specs/2026-06-21-offline-ztf-classification-design.md`,
`feature_step/features/offline/FLOW.md` (§3d, §7)

## Problem

The offline ZTF pipeline can compute features and run the BHRF (Squidward)
lightcurve classifier end-to-end, but the DB metadata that gives BHRF output a
home is not seeded. The ZTF feature LUTs (`feature_name_lut`,
`feature_version_lut`) are done; the **`classifier`** and **`taxonomy`** LUTs
still have **no ZTF/BHRF rows** — live `multisurvey_ztf` holds only the four flat
**stamp** classifiers (ids 1–4). Seeding these two LUTs is the prerequisite for
ever persisting BHRF probabilities to `multisurvey_ztf.probability`.

## How the hierarchy was modeled (legacy) vs. now

BHRF is a **hierarchical** classifier: a top level (Transient / Stochastic /
Periodic) and one classifier per branch. The two DB generations represent this
differently:

| | Legacy `alerce.taxonomy` | New `multisurvey_ztf` |
|---|---|---|
| `classifier` shape | `classifier_name` (str), `classifier_version` (str), `classes` (text[]) | `classifier_id` (int), `classifier_name` (str), `classifier_version` (str), `tid` (smallint), `created_date` |
| `taxonomy` shape | (embedded as the `classes` array above) | `class_id` (int), `class_name` (str), `order` (int), `classifier_id` (smallint), `created_date` |
| Hierarchy encoding | **multiple named classifier rows** — one per branch + top + a flat combined | same idea, keyed by integer `classifier_id`; **no hierarchy column** |
| `probability.classifier_version` | — | **smallint** (`2.1.0` → `210`) — separate from `classifier.classifier_version` (string) |

Legacy `alerce.taxonomy` BHRF 2.1.0 rows (the model we deploy):

```
lc_classifier_BHRF_forced_phot_top         v2.1.0 → [Transient, Stochastic, Periodic]                                  (3)
lc_classifier_BHRF_forced_phot_transient   v2.1.0 → [SNIa, SESN, SNII, SNIIn, SLSN, TDE]                               (6)
lc_classifier_BHRF_forced_phot_stochastic  v2.1.0 → [Microlensing, QSO, AGN, Blazar, YSO, CV/Nova]                     (6)
lc_classifier_BHRF_forced_phot_periodic    v2.1.0 → [LPV, EA, EB/EW, Periodic-Other, RSCVn, CEP, RRLab, RRLc, DSCT]    (9)
lc_classifier_BHRF_forced_phot             v2.1.0 → the flat 21-leaf list (6+6+9)                                     (21)
```

Note BHRF 2.1.0 leaf names differ from the old `hierarchical_random_forest_1.0.0`
taxonomy (`SESN` not `SNIbc`; `EA`/`EB/EW` not `E`) — this is why the model, not
legacy, is the source of truth.

## Strategy (the multisurvey mapping)

Mirror legacy: seed **five `classifier` rows** — top + 3 branches + the flat
combined — each with its own `taxonomy` rows. Probabilities will later be stored
**one row per class per classifier** (the existing `probability` shape).

### `classifier` — 5 rows

`classifier_version = "2.1.0"` (string, from `model.model_version`), `tid = 0`
(ZTF). IDs are next-free: live max is **4**, so **5–9**.

| classifier_id | classifier_name | # classes |
|---|---|---|
| 5 | `lc_classifier_BHRF_forced_phot` (flat; the `CLASSIFIER_NAME` the step deploys) | 21 |
| 6 | `lc_classifier_BHRF_forced_phot_top` | 3 |
| 7 | `lc_classifier_BHRF_forced_phot_transient` | 6 |
| 8 | `lc_classifier_BHRF_forced_phot_stochastic` | 6 |
| 9 | `lc_classifier_BHRF_forced_phot_periodic` | 9 |

### `taxonomy` — 45 rows (21+3+6+6+9)

For each classifier: `class_id` is **per-classifier, 0-indexed**, in the model's
output-column order for that branch (matches how existing ids 1–4 each restart at
0). `order = class_id`. `class_name` comes from the model.

## Source of truth: model, cross-checked against legacy

Load BHRF 2.1.0 from the deployed URL
(`https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl`;
the `MODEL_PATH` the step uses) in the `atat2` conda env (which has both
`alerce_classifiers` and `features`). Read the class lists and their order from
the `OutputDTO`:

- **top** classes ← the `hierarchical["top"]` frame columns,
- each **branch**'s leaves ← `hierarchical["children"][branch]` frame columns,
- **flat** 21 leaves ← the main probabilities frame columns.

**Diff every list (names + order) against the legacy `alerce.taxonomy` arrays and
surface any mismatch before writing the SQL.** The model wins on conflict; the
legacy diff is a guardrail against typos/renames.

## Deliverable

1. **`feature_step/features/offline/ztf_classifier_taxonomy_seed.sql`** — new
   file, sibling to `ztf_feature_luts_seed.sql`. Idempotent
   `INSERT … ON CONFLICT DO NOTHING`. Contains the 5 `classifier` rows + 45
   `taxonomy` rows, hand-written from the model-verified class lists.
2. **Applied direct-to-live** with write credentials (manual step — offline
   default credentials are read-only; same procedure used for the feature LUTs).
3. **Docs**:
   - `FLOW.md` §3d table: `classifier` + `taxonomy` rows → **Done** (was Pending);
     §7 Done/Pending lists updated accordingly.
   - `README.md` file map / status updated to mention the new seed file.
   - The **back-port to the db-plugins authority file**
     (`libs/db-plugins-multisurvey/db_plugins/db/sql/_initial_data_pipeline.py`,
     `INITIAL_DATA`) is recorded as **pending**, with the caveat that it must
     first **reconcile the missing live ids 3–4** (this checkout's seed stops at
     id 2) before adding BHRF 5–9, or it will renumber over real ids.

## Out of scope

- **`probability_writer.py`** and wiring `offline_classify.py --save` — a separate,
  larger lift (see FLOW.md §7 item 4). This design only seeds the LUTs that
  unblock it.
- The name→smallint **`classifier_version`** convention for the `probability`
  table (`2.1.0` → `210`) — decided when the writer is built.
- Actually persisting any BHRF probabilities.

## Risks / open points

- **ID drift.** Next-free = 5 is verified against *live* today. Re-verify at
  apply time; another deploy could claim 5+ first.
- **Model vs. legacy mismatch.** If the model emits class names/order that differ
  from legacy beyond the known `SESN`/`EA`/`EB/EW` renames, stop and reconcile
  with the user before seeding — do not silently pick one.
- **Model load cost.** Loading the pickle is a one-time network fetch + heavy
  import; done once during implementation to read the taxonomy, not at seed-apply
  time.
