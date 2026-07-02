# Seed the BHRF hierarchical LC classifier + taxonomy into `multisurvey_ztf`

**Date:** 2026-07-02
**Status:** Approved design, pending implementation plan. Class names locked to
**`SNIbc`** (what the deployed model emits); a possible relabel to `SESN` is
tracked as a pending item (see "Pending: `SNIbc` → `SESN` relabel").
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
| `taxonomy` shape | just `classifier_name`, `classifier_version`, `classes` (text[]) — **no `class_id`, no `order` column** (class order is the array position) | `class_id` (int), `class_name` (str), `order` (int), `classifier_id` (smallint), `created_date`; **PK = `(class_id, classifier_id)`** |
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

**Important:** the legacy `alerce.taxonomy`/`alerce.probability` rows above store
the transient class as **`SESN`**, but the **deployed 2.1.0 model emits `SNIbc`**
(see "Class-name provenance" below). We seed **`SNIbc`** — what the current
pipeline actually produces — not the legacy label. The `SESN`/`EA`/`EB/EW` names
in legacy are catalog artifacts, not what our model outputs.

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
output-column order for that branch. `order = class_id`. `class_name` is the
model's own label.

**PK / conflict target = `(class_id, classifier_id)`** (verified on live:
`pk_taxonomy_classid_classifierid`), so `class_id` restarts at 0 per classifier —
it is *not* globally unique. The seed's `ON CONFLICT` must target
`(class_id, classifier_id)`, **not** `class_id` alone. (Note: the db-plugins
authority file `_initial_data.py` is out of sync — it declares
`index_elements: ["class_id"]` and uses a *global* `class_id` 0–10; the back-port
must reconcile to the live composite key.)

**What `order` means.** It is a static, per-classifier, 0-indexed enumeration of a
classifier's classes. Its only consumer is `get_taxonomy_by_classifier_id`
(`… ORDER BY "order" ASC`), which builds a `{class_name: class_id}` dict — so
`order` only sets iteration order and is **cosmetic** for the actual mapping, not
load-bearing. It is **not** the per-object probability rank (that's
`probability.ranking`, computed at write time). The existing stamp rows all use
`order == class_id`; we follow that. If ALeRCE ever wants the catalog to *display*
classes in the legacy semantic order (e.g. `SNIa, …, SLSN, TDE`), `order` is the
knob — it can diverge from `class_id` without affecting the mapping. Legacy has no
`order` column at all; its ordering is just the `classes[]` array position, and
that semantic order differs from the model's alphabetical `classes_`.

The class lists + order, read directly from the deployed pickle (see below), are:

| classifier_id | classifier | classes (in `class_id`/`order` = 0…n order) |
|---|---|---|
| 5 | flat | `AGN, Blazar, CEP, CV/Nova, DSCT, EA, EB/EW, LPV, Microlensing, Periodic-Other, QSO, RRLab, RRLc, RSCVn, SLSN, SNII, SNIIn, SNIa, SNIbc, TDE, YSO` |
| 6 | top | `Periodic, Stochastic, Transient` |
| 7 | transient | `SLSN, SNII, SNIIn, SNIa, SNIbc, TDE` |
| 8 | stochastic | `AGN, Blazar, CV/Nova, Microlensing, QSO, YSO` |
| 9 | periodic | `CEP, DSCT, EA, EB/EW, LPV, Periodic-Other, RRLab, RRLc, RSCVn` |

Ordering is the RF's `classes_` (alphabetical for branches; the flat list is
`list_of_classes`). Since the multisurvey write path maps `class_name → class_id`
by **exact name match** against this table (see "Why the name must match the
model"), the ordering itself is effectively cosmetic — but the **names must be
exactly these**.

## Source of truth: the deployed pickle

The class names/order are read **directly from the deployed model pickle**, which
is the production artifact (verified against the production `MODEL_CONFIG`:
`model_path = https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl`,
`SquidwardFeaturesClassifier` + `SquidwardMapper`). Load it via **this checkout's
submodule** (`alerce_classifiers/`, importable in the `training_py310` conda env
with `PYTHONPATH` prepended) and read:

- **flat** 21 leaves ← `HierarchicalRandomForestClassifier.list_of_classes`,
- each **branch** ← `dict_of_rf[branch].classes_` (`top`, `Transient`,
  `Stochastic`, `Periodic`).

Nothing between the model and the DB renames these — `SquidwardMapper.postprocess`,
the `lc_classification_step` scribe parsers (`TopBottomScribeParser`), and both
scribes all pass the `OutputDTO` **columns through verbatim**. So the model's
`classes_` *are* what production writes.

### Class-name provenance (`SNIbc` vs `SESN`) — resolved

Investigated 2026-07-02. The legacy `alerce` DB stores the transient class as
`SESN`; the deployed model emits `SNIbc`. Findings:

- The deployed 2.1.0 pickle contains **`SNIbc`** in `list_of_classes`,
  `class_hierarchy`, and every branch's `classes_` — **no `SESN` anywhere**.
- `SESN` appears **only** in the training config
  (`training/.../HBRF/training.py`, `class_hierarchy`), never in the inference
  path (`git log -S SESN` empty in `alerce_classifiers`; not in any step/scribe).
- `SESN` and this model's `SNIbc` are the **same astrophysical class**
  (stripped-envelope SNe): the pickle's transient branch has 6 classes with
  `SNIbc` and **no** `SNIIb` — i.e. `SNIIb` was already merged into `SNIbc`, the
  same merge `SESN` represents (`SESN = SNIbc ∪ SNIIb`).
- The `SESN` rows in `alerce.probability` (also tagged `2.1.0`) therefore came
  from a **different artifact / the retired old-alerce pipeline**, not this
  deployed pickle. The `2.1.0` version tag was reused across the relabel.

Conclusion: **seed `SNIbc`** to match what the deployed model and production
pipeline actually emit.

### Why the name must match the model

The multisurvey write path (pattern in
`stamp_classifier_2025_multisurvey_step/.../db/db.py`) melts the model output to
`(oid, class_name, probability)` and maps `class_name → class_id` via
`get_taxonomy_by_classifier_id` = `{class_name: class_id}` fetched from the
`taxonomy` table — an **exact string match** (`class_name_to_id`, `-1` on miss).
The scribe (`scribe_multisurvey`) then writes the integer `class_id`; it does no
name mapping. So if the taxonomy holds `SESN` while the model emits `SNIbc`, the
lookup misses and `class_id` is garbage. This is the hard reason the taxonomy
names must equal the model's labels.

## Deployed model artifact — full contents (verified 2026-07-02)

The pickle is a `dict` with 4 keys (loaded via this checkout's submodule,
`training_py310` env):

- **`feature_list`** — **199 features** (ndarray, band-suffixed `_1`/`_2` = g/r):
  `Amplitude_1/2`, `AndersonDarling_1/2`, … `Coordinate_x/y/z`, `sgscore1`,
  `ulens_{chi,fs,tE,u0}_1/2`, etc. This is the exact input vector the model
  expects.
- **`list_of_classes`** — the 21 flat leaves (above).
- **`class_hierarchy`** — stored in the pickle (not the `class_hierarchy_old`
  fallback): Transient(6, `SNIbc`, **no `SNIIb`**), Stochastic(6), Periodic(9).
- **`model`** — dict of 4 `imblearn` `BalancedRandomForestClassifier` (`top`,
  `Stochastic`, `Periodic`, `Transient`), each **500 trees**, `max_depth=100`,
  **199 features**, 500 estimators fitted.

### Feature-count cross-check (199 vs 123) — flagged, separate task

The model consumes **199 band-suffixed features**, whereas the offline
`feature_name_lut` we already seeded has **123 band-less names** (band lives in
`feature.band`, so the axes aren't directly comparable). Before trusting offline
BHRF *probabilities* (not part of this taxonomy seed), confirm the offline feature
computation actually produces all 199 features the model expects, in the names the
model's `feature_list` uses. Tracked as a pending verification, out of scope here.

## Deliverable

1. **`feature_step/features/offline/ztf_classifier_taxonomy_seed.sql`** — new
   file, sibling to `ztf_feature_luts_seed.sql`. Idempotent
   `INSERT … ON CONFLICT DO NOTHING`, with conflict targets matching the live
   PKs: **`classifier` → `(classifier_id)`**, **`taxonomy` → `(class_id,
   classifier_id)`**. Contains the 5 `classifier` rows + 45 `taxonomy` rows,
   hand-written from the model-verified class lists.
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

## Pending: `SNIbc` → `SESN` relabel (deferred, not blocking)

We seed `SNIbc` now because it's what the deployed model emits and what the
multisurvey write path requires (exact-match lookup). But the legacy catalog
displays `SESN` for continuity, and the training config uses `SESN`. Open
question for ALeRCE (to resolve in person): **should the new `multisurvey_ztf`
catalog show `SESN`?** If yes, the fix is a **product/pipeline change, not a seed
hack** — either retrain/redeploy a model that emits `SESN`, or add an explicit
rename step *before* the `class_name → class_id` lookup — and then re-seed the two
affected `taxonomy` rows (flat `class_id` for `SNIbc`, and transient). Until that
decision lands, `SNIbc` stands. Also worth confirming: why `alerce.probability`
reused the `2.1.0` tag across the `SESN`↔`SNIbc` relabel.

## Pending: confirm offline features cover the model's 199

See "Feature-count cross-check" above. Needed before offline BHRF probabilities
can be trusted; independent of this taxonomy seed.

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
- **`SNIbc`/`SESN` divergence from the legacy catalog.** Resolved for now
  (seed `SNIbc`); see the pending relabel item. The residual risk is *display*
  inconsistency with the old catalog, not a technical break.
- **Model load cost.** Loading the pickle is a one-time network fetch (1.38 GB) +
  heavy import; done once during implementation to read the taxonomy, not at
  seed-apply time. Requires an env where `imblearn` imports cleanly
  (`training_py310` works; `atat2` has an sklearn/imblearn clash).
