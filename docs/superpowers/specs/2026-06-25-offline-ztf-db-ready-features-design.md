# Offline ZTF features → DB-ready shape (production-LSST rules)

**Date:** 2026-06-25
**Status:** design, awaiting review
**Related:** `feature_step/features/offline/FLOW.md` (§3d, §5, §7),
`docs/superpowers/specs/2026-06-21-offline-ztf-classification-design.md`

---

## 1. Problem

The offline ZTF feature path (`feature_step/features/offline/`) currently returns
`AstroObject.features` **verbatim** — the long named frame `[name, value, fid, sid,
version]`, **including NaN-valued features and with no name→id mapping**. That is
not how features are stored: the `feature` table holds `(oid, sid, feature_id,
band, version, value)` and production drops NaN rows before writing.

We want the offline feature output to behave **as if it were going to be saved
into the same `feature` table as production LSST, with the same rules** — so an
offline feature set is directly comparable to what the pipeline would persist.

Two facts make this more than a filter tweak:

1. **Production ZTF does *not* follow the LSST rules today.** `prepare_ao_features_for_db`
   (`feature_step/features/utils/parsers.py:405`) drops NaN (correct) but maps
   feature ids with a broken per-object `enumerate` (lines 420–426) instead of the
   `feature_name_lut`, and the LUT it *is* handed is silently dropped. So the same
   feature gets a different `feature_id` across objects. Aligning offline to the
   LSST LUT-based rule means **fixing this production path**.
2. **The ZTF `feature_name_lut` and `feature_version_lut` are empty** (FLOW §3d) —
   there is nothing in the DB to map against yet.

## 2. Target: the `feature` table contract

`Feature` (`libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py:1007`):

| column | type | source |
|---|---|---|
| `oid` | BigInteger | object metadata |
| `sid` | SmallInteger | `0` (ZTF) |
| `feature_id` | SmallInteger | `feature_name_lut` (band-less name → id), namespaced by `sid` |
| `band` | SmallInteger | `fid_mapper_for_db(band_str)` → `{g:1, r:2, "g,r":12, else:0}` |
| `version` | SmallInteger | `feature_version_lut` (version string → id) |
| `value` | DOUBLE_PRECISION | feature value (NaN rows dropped) |

PK `(oid, sid, feature_id, band)`.

**Production LSST save rules** (`prepare_ao_features_for_db_lsst`, parsers.py:441) —
the rules we mirror:
1. drop NaN (`value.notna()`)
2. `fid → band` integer code (survey-specific mapper; ZTF uses `fid_mapper_for_db`)
3. inf/nan → None
4. back-compat name fixes (`Power_rate_1_4 → Power_rate_1/4`, …)
5. `name → feature_id` via `feature_name_lut` (`{feature_id: feature_name}` dict)

## 3. Decisions (locked with user)

- **Scope:** Full DB-ready shape. Offline emits `(oid, sid, feature_id, band,
  version, value)` rows. We **do not** INSERT into the DB.
- **Architecture:** *Fix & reuse production.* Fix `prepare_ao_features_for_db` to
  use the `feature_name_lut` (like the LSST variant); both production and offline
  call the same function. The function takes the LUT **as a parameter** — caller
  decides the source.
- **LUT source:** *Local fixture.* Offline supplies the name→id and version→id maps
  from a checked-in fixture (no DB write; offline stays read-only). Production keeps
  loading the LUT from the DB via `get_feature_name_lut`.
- **Output split:** Add `compute_db_features` (DB-ready). **Keep `compute_features`
  / `compute_astro_object`** (named, NaN-inclusive) unchanged — `classify.py` needs
  the wide named vector and `compare_vs_alerce` needs name-keyed features.

### Known trade-off
Fixture `feature_id`s are offline's own. They will **not** equal the eventual DB
`feature_name_lut` ids until that LUT is seeded for real (deferred work, FLOW §3d).
The **shape and rules** match production now; the **id values** reconcile later. The
fixture is the single place that has to change when the DB LUT is finally seeded.

## 4. Design

### Piece 1 — Fix `prepare_ao_features_for_db(astro_object, feature_name_lut)`
File: `feature_step/features/utils/parsers.py`

- Add the `feature_name_lut` parameter (match the LSST signature).
- Replace the `enumerate` block (lines 420–426) with LUT-based mapping:
  ```python
  name_to_id = {name: feature_id for feature_id, name in feature_name_lut.items()}
  ao_features["feature_id"] = ao_features["name"].map(name_to_id)
  ```
- Keep unchanged: the `value.notna()` drop, `band = fid.apply(fid_mapper_for_db)`,
  the inf/nan→None replace, the `Power_rate` back-compat fixes, and the
  unmapped-feature warning.
- Keep `name` in the returned frame (the ZTF caller `parse_scribe_payload` reads it
  for `get_color_from_features`, then drops it). Output columns unchanged:
  `[name, value, band, feature_id]`.
- **Caller wiring (the whole prod fix):** `parse_scribe_payload` (parsers.py:483)
  already *receives* `feature_name_lut` but never forwards it. Change line 500 to
  `prepare_ao_features_for_db(astro_object, feature_name_lut)`. `step.py` already
  loads and threads the LUT for ZTF (step.py:82, :213/:217), so no step.py change
  is needed — the fix lights up the path that's already wired.

### Piece 2 — Local fixture
File: `feature_step/features/offline/feature_lut.py` (new)

- `FEATURE_NAME_LUT: dict[int, str]` — `{feature_id: band-less feature_name}`,
  `sid = 0`. Ids `0..N-1` assigned by **sorted feature name** (deterministic, stable
  across regeneration).
- `FEATURE_VERSION_LUT: dict[int, str]` — `{version_id: version_name}`, one row;
  `version_name` is the extractor version that already rides in `ao.features["version"]`.
- `load_feature_name_lut() -> dict[int, str]` — returns `FEATURE_NAME_LUT` in the
  exact shape `get_feature_name_lut` returns, so it is a drop-in for Piece 1.
- `version_name_to_id(name: str) -> int` — reverse-maps the version string to its
  smallint id (raises/warns on unknown).

**Generation (one-off, checked in as static data):**
A small generator (`scripts/offline_generate_feature_lut.py`) runs
`compute_astro_object` on a representative oid, takes `ao.features` **without** the
NaN filter (so it captures the *full* feature-name universe, not one object's
non-NaN subset), applies the `Power_rate` back-compat name fixes, collects distinct
band-less names, sorts, assigns ids `0..N-1`, and prints the fixture literal. The
output is pasted into `feature_lut.py` and committed. The fixture is **not**
regenerated at runtime.

> Open detail to settle during implementation: a single representative oid should
> expose the extractor's full feature schema (the extractor computes the same
> feature set for every object; only values differ / go NaN). If in doubt, union
> the names across a few oids in the generator. The generator must log the final
> count and name list so the fixture is reviewable.

### Piece 3 — Offline DB-ready output
File: `feature_step/features/offline/lc_features.py`

- New `compute_db_features(message, references_db, allwise, min_detections=1,
  preprocessor=None, extractor=None, feature_name_lut=None) -> pd.DataFrame | None`:
  1. `ao = compute_astro_object(...)`; return `None` if `ao is None`.
  2. `lut = feature_name_lut or load_feature_name_lut()`.
  3. `rows = prepare_ao_features_for_db(ao, lut)` — reuses the fixed production fn
     (drop NaN + band + feature_id, name dropped next).
  4. Attach `oid` (from `ao.metadata`), `sid = 0`, and `version` (smallint via
     `version_name_to_id` over `ao.features["version"]`); drop `name`.
  5. Return rows with exactly the `feature` columns:
     `[oid, sid, feature_id, band, version, value]`.
- `compute_features` and `compute_astro_object` are **unchanged**.
- `scripts/offline_compute_features.py` switches its output to `compute_db_features`
  (this is now "the features" the offline tool reports). `--named`/legacy flag
  optional (decide in plan) if the raw named frame is still wanted for debugging.

## 5. What is explicitly out of scope

- **No DB writes / INSERT** into `multisurvey_ztf.feature`.
- **No seeding of the real DB `feature_name_lut` / `feature_version_lut`** (deferred,
  FLOW §3d) — the fixture stands in.
- **No change to `compute_features`, `compute_astro_object`, `classify.py`, or
  `compare_vs_alerce`** semantics (they keep the named, NaN-inclusive frame).
- **No LSST-path changes.**

## 6. Testing

- **Unit (parsers):** `prepare_ao_features_for_db` with a small fake `AstroObject` +
  a 2–3 entry LUT → asserts NaN rows dropped, `band` codes correct (incl. `None→0`
  and `g,r→12`), `feature_id` from the LUT (not enumerate), unmapped name warns and
  yields NaN id. Regression-guards the prod fix.
- **Unit (fixture):** `load_feature_name_lut` shape; ids contiguous `0..N-1`; sorted;
  `version_name_to_id` round-trips a known version and warns on unknown.
- **Unit (offline):** `compute_db_features` on a stub message → output has exactly
  the `feature` columns, dtypes (`feature_id/band/version` int, `value` float), `sid=0`,
  no NaN values, and oid matches.
- **Integration (manual / opt-in):** `offline_compute_features.py --oid <bigint>`
  against `multisurvey_ztf` produces DB-ready rows; spot-check a known feature's
  id/band/version.

## 7. File map (changes)

| File | Change |
|---|---|
| `feature_step/features/utils/parsers.py` | Fix `prepare_ao_features_for_db` (LUT param + mapping); thread LUT in `parse_scribe_payload`. |
| `feature_step/features/offline/feature_lut.py` | **new** — fixture + loaders. |
| `feature_step/features/offline/lc_features.py` | **new** `compute_db_features`. |
| `feature_step/scripts/offline_generate_feature_lut.py` | **new** — one-off fixture generator. |
| `feature_step/scripts/offline_compute_features.py` | emit `compute_db_features`. |
| `feature_step/features/offline/FLOW.md` / `README.md` | document the DB-ready output + fixture. |
