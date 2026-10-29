# feature_step: ZTF multisurvey support — design

**Date:** 2026-08-04
**Repo:** `pipeline_features/pipeline`, branch `features_step`
**Status:** approved, ready for implementation planning

## Problem

`feature_step` currently only works for LSST. The ZTF branch exists in
`features/step.py` and `features/utils/parsers.py` but has never run against a
real `magstats_ms_ztf` message, and does not correctly consume one.

A validated offline ZTF feature pipeline already exists in the sibling repo
(`desktop/pipeline`, branch `fix/ztf-feature-parser-extra-fields`, package
`feature_step/features/offline/`). It reproduces the step's pure computation
from the database and has been diffed against the legacy `alerce.feature`
table. This design ports its *semantics* into the streaming step.

## Input contract

`feature_step` consumes the magstats output topic for its survey
(`PIPELINE-MULTISURVEY.md`). For ZTF that is
`schemas/magstats_ms_step/ztf/output.avsc`, record `magstats_ms_ztf`:

```
oid                  long
sid                  int
measurement_id       array<long>
meanra, meandec      union{null, float}
detections           array<candidate>
previous_detections  array<prv_candidate>
forced_photometries  array<forced_photometry>
non_detections       array<non_detection>
```

`magstats_multisurvey_step` passes the message through untouched apart from
refreshed mean coordinates (`magstats_multisurvey_step/step.py::execute` →
`refresh_mean_coordinates`), so this is exactly the correction step's output
plus `sid`/`meanra`/`meandec`.

The records are **flat** — there is no `extra_fields` map anywhere in the ZTF
multisurvey path (verified: `correction_multisurvey_step/core/` contains no
occurrence of the string). The three arrays use three overlapping but distinct
field vocabularies:

| field | `candidate` | `prv_candidate` | `forced_photometry` |
|---|:--:|:--:|:--:|
| corrected mag | `magpsf_corr` | `magpsf_corr` | `mag_corr` |
| corrected mag err | `sigmapsf_corr_ext` | `sigmapsf_corr_ext` | `e_mag_corr_ext` |
| `rb` | yes | yes | — |
| `procstatus` | — | — | yes |
| `distnr` | yes | yes | yes |
| `rfid` | yes | — | yes |
| `sharpnr`, `chinr` | yes | yes | yes |
| PS1 (`sgscore1`, `distpsnr1`, …) | yes | — | — |
| `forced` (boolean) | yes | yes | yes |

Every record carries `forced`, which is what makes the flatten below safe.

### Not the offline message shape

`features/offline/message.py::build_message` emits a *different* shape: forced
epochs inline in `detections`, and per-epoch aux fields nested under an
`extra_fields` map. The offline parser was adapted to match (commit `d4dcb1c04`
in `desktop/pipeline`). **That adaptation is not ported.** It is correct for the
offline harness, which controls its own message construction, and wrong for the
stream. We take the schema as authoritative and port only the offline's
behavioral decisions.

## Design

### 1. `pre_execute` is the single message-shaping point

Mirrors both the LSST branch (`step.py:325`, which merges
`sources + previous_sources` into `message['detections']`) and the offline
harness (`features/offline/lc_features.py::_prepare_detections`, which runs
`discard_bogus_detections` over one combined list).

Replace the ZTF arm of `step.py::pre_execute`:

```python
if self.survey == "ztf":
    epochs = (message.get("detections", [])
              + message.get("previous_detections", [])
              + (message.get("forced_photometries") or []))
    filtered_message["detections"] = discard_bogus_detections(epochs)
    filtered_messages.append(filtered_message)
```

Consequences:

- `previous_detections` reaches the extractor for the first time (**#6**).
- `discard_bogus_detections` now sees forced rows, so the `procstatus not in
  {"0","57"}` filter actually runs (**#5**). It reads `rb`/`procstatus` from the
  top level when no `extra_fields` key is present
  (`lc_classifier/features/core/base.py:116-118`), which is the flat case — no
  change needed there.
- `forced_photometries` absent or `null` no longer yields `None` (**#7**).
- `has_enough_detections` is unchanged and still correct: it counts
  `not det.get("forced", False)`, identical to the offline's `n_real`.

### 2. `execute` passes `forced=[]` and stamps `aid`

```python
m = map(lambda x: {**x,
                   "aid": x["oid"],
                   "index_column": f'{x["measurement_id"]}_{x["oid"]}'},
        message.get("detections", []))
ao = self.detections_to_astro_object_fn(list(m), [], xmatch_data, references_db)
```

The `aid` stamp is new (**#13**). `add_mag_and_flux_columns` does
`a.set_index("aid")` (`parsers.py:161`) and the parser then reads
`aid = a.index.values[0]` (`parsers.py:288`), but no ZTF multisurvey record has
an `aid` field — today the entire index is `NaN` and `metadata["aid"]` is `NaN`.
The offline sets `aid = oid` (`lc_features.py:30`); we match it.

### 3. `detections_to_astro_object` drops its second loop

One loop over one list. `forced` is read as an ordinary column from the row and
`aid_forced = a[a["forced"]]` / `aid_detections = a[~a["forced"]]` split at the
end, exactly as today.

This makes **#4** and **#5** structurally impossible rather than patched:
`get_reference_for_each_detection` and `get_bogus_flags_for_each_detection` are
computed over the same list that built `a`, so the `concat(axis=1)` at
`parsers.py:277` and `parsers.py:282` aligns by construction. Under the current
two-loop code `a` has `len(detections) + len(forced)` rows while the aux frames
have `len(detections)`, so every forced row silently receives
`NaN` for `distnr`, `rfid`, `rb` and `procstatus`.

Keep the offline's guard as a tripwire against regressing to the two-loop form:

```python
forced = forced or []
if forced:
    raise NotImplementedError(
        "detections_to_astro_object: `forced` must be empty for ZTF; forced "
        "epochs flow inline via the per-row `forced` flag in `detections`."
    )
```

`detection_keys` gains `"forced"`.

### 4. Corrected-magnitude coalesce (**#1**)

The one place we cannot copy the offline, because it normalized these names in
its own `build_message`. Detections carry `magpsf_corr`/`sigmapsf_corr_ext`;
forced carry `mag_corr`/`e_mag_corr_ext`. Each row populates exactly one pair,
so the coalesce is unambiguous.

Read both, then merge immediately after the DataFrame is built and **before**
the `DETECTION_KEYS_MAP` rename:

```python
detection_keys = [..., "magpsf_corr", "mag_corr",
                       "sigmapsf_corr_ext", "e_mag_corr_ext", ..., "forced"]

a["magpsf_corr"] = a["magpsf_corr"].fillna(a["mag_corr"])
a["sigmapsf_corr_ext"] = a["sigmapsf_corr_ext"].fillna(a["e_mag_corr_ext"])
a.drop(columns=["mag_corr", "e_mag_corr_ext"], inplace=True)
```

Remove the `"mag_corr": "magpsf_corr"` and
`"e_mag_corr_ext": "sigmapsf_corr_ext"` entries from `DETECTION_KEYS_MAP`
(`parsers.py:26-27`) — with both spellings now selected into the frame, those
renames would produce duplicate column names. The LSST path does not select
either key, so it is unaffected.

Without this fix every forced epoch loses its corrected magnitude, which
propagates into the `magnitude`-unit features and the `_corr` colours.

### 5. Two alignments with the LSST path

**#2 — xmatch gate.** `parsers.py:300` reads
`if xmatches is not None and "allwise" in xmatches.keys():`, but
`step.py:302-311` attaches the `conesearch_with_metadata` result, whose keys are
`{oid, catalog, distance, match_id, metadata}`. The condition never fires, so
**W1–W4 are always NaN in the current ZTF path**. Change to match the LSST
parser (`parsers.py:108`):

```python
if xmatches is not None and xmatches.get("catalog") == "allwise":
```

This is hard-blocking, not cosmetic: `multisurvey_ztf.xmatch` and
`multisurvey_ztf.allwise` are both empty (verified 2026-08-04), so the live
Xwave call is the *only* source of WISE magnitudes. Eleven of the 127 seeded
features depend on it: `W1-W2`, `W2-W3`, `W3-W4`, `g-W1`, `r-W1`, `g-W2`,
`r-W2`, `g-W3`, `r-W3`, `g-W4`, `r-W4`.

**#3 — feature id mapping.** `prepare_ao_features_for_db` (`parsers.py:425-430`)
builds ids with `enumerate()` over whatever names happen to be present, so the
same feature gets a different id depending on the batch. Take the LUT as an
argument and invert it, exactly as `prepare_ao_features_for_db_lsst` already
does (`parsers.py:468`):

```python
def prepare_ao_features_for_db(astro_object, feature_name_lut):
    ...
    name_to_id = {name: feature_id for feature_id, name in feature_name_lut.items()}
    ao_features["feature_id"] = ao_features["name"].map(name_to_id)
```

Update the call site at `parsers.py:504` to pass `feature_name_lut`, which
`parse_scribe_payload` already receives.

### 6. `lc_classifier` parity

Port from `desktop/pipeline` (commit `8743448fa`), four files:

| File | Change |
|---|---|
| `features/extractors/spm_extractor.py` | emit `SPM_mjd_ref`; version `1.0.1` → `1.0.2` |
| `features/extractors/tde_extractor.py` | emit `TDE_mjd_ref` and `fleet_mjd_ref`; `TDETailExtractor` `1.0.1` → `1.0.2`, `FleetExtractor` `1.0.2` → `1.0.3` |
| `features/extractors/ulens_extractor.py` | emit `ulens_mjd_ref`; `get_observations` returns the reference epoch; version `1.0.2` → `1.0.3` |
| `features/core/base.py` | `str(procstatus)` coercion in `discard_bogus_detections` |

Required because the deployed LUT was seeded from that extractor set — see
"Database state" below. Without it the step emits 123 of the 127 seeded
features and the four `*_mjd_ref` ids are never written.

## Database state

Verified against `multisurvey_ztf` on `quimal-db2.alerce.online` (2026-08-04,
`readonly_user`):

| Object | State |
|---|---|
| `feature_name_lut` (sid=0, tid=0) | **127 rows, byte-identical to `features/offline/feature_lut.py::FEATURE_NAME_LUT`** — same ids, same names |
| `feature_version_lut` | one row: `(version_id=0, version_name='27.5.7a31', sid=0, tid=0)` |
| `ztf_reference` | ~148M rows; columns match the `ZtfReference` model and the `["oid","rfid","sharpnr","chinr"]` read |
| `feature` | exists, columns `oid, sid, feature_id, band, version, value`; **empty** |
| `xmatch`, `allwise` | **empty** |
| `sid_lut` | ZTF = 0, confirming `sid=0, tid=0` at `step.py:81-82` |

No seeding work is required. Two operational consequences:

**Feature version.** `pyproject.toml` is at `27.7.1`; the LUT only knows
`27.5.7a31`. On first startup `get_or_create_version_id`
(`features/database.py:93`) will not find `27.7.1`, will compute
`MAX(version_id) + 1 = 1`, and `INSERT`. This is the intended behavior: the
step's DB user **needs INSERT privilege on `multisurvey_ztf.feature_version_lut`**,
and streaming features land under `version_id=1`, distinct from the offline
backfill's `version_id=0`.

**Schema config.** `self.schema` comes from `config["DB_CONFIG"]["SCHEMA"]` and
defaults to `"multisurvey"` (`step.py:70`). The ZTF deployment must set it to
`multisurvey_ztf`, or the LUT read, the version lookup and the reference read
all hit the wrong schema.

## Behavior changes

**Forward message contents.** `parse_output` emits
`"detections": message["detections"]` (`parsers.py:638`), which after the
flatten is the merged, bogus-filtered light curve rather than the three original
arrays. The forward message therefore no longer has `magstats_ms_ztf` shape.
The LSST branch already behaves this way, so this is consistency rather than
novelty — but it lands on whatever consumes the ZTF feature topic, and the
output Avro schema is out of scope here (see below).

## ZTF i-band

ZTF has three bands — `multisurvey_ztf.band` maps `sid=0` to `1=g, 2=r, 3=i` —
and i-band data is real but vanishingly rare (3M-row samples: 8 of 3,000,000
`detection` rows and 174 of 3,000,000 `forced_photometry` rows). The
`{1: "g", 2: "r", 3: "i"}` map at `parsers.py:291` must therefore keep its `i`
entry, so those rows are labelled rather than becoming `NaN`.

`fid_mapper_for_db`'s `{"g": 1, "r": 2, "g,r": 12}` with a `0` fallback is
nonetheless **complete**. `ZTFFeatureExtractor._instantiate_extractors` opens
with `bands = list("gr")` (`composites/ztf.py:28`) and passes that list to every
band-aware extractor; no extractor derives bands from the data (the only
`.unique()` calls under `features/extractors/` are on `sid`). So
`astro_object.features["fid"]` is always in `{"g", "r", "g,r", None}` and `"i"`
never reaches the mapper. `band=0` in `multisurvey_ztf.feature` unambiguously
means "object-level feature".

The consequence to be aware of: i-band photometry enters the `AstroObject` and
is silently ignored by every extractor, but still counts toward band-agnostic
values — notably `last_mjd`, computed as `max` over all epochs
(`parsers.py:324-333`) and emitted as `mjd` in the scribe `update_object`
command. An object whose most recent epoch is i-band gets a `last_mjd` that no
feature reflects. This matches legacy ALeRCE behavior (features have always been
g/r only) and is not changed here.

## Testing

`tests/unittest/test_step_ztf.py` and `tests/message_factory.py` build the
legacy/elasticc `extra_fields` shape and exercise none of this. Build a new ZTF
fixture from `schemas/magstats_ms_step/ztf/output.avsc` and cover:

1. Forced epochs retain their corrected magnitudes (`mag_corr` → `brightness`).
2. `previous_detections` rows reach the extractor.
3. Forced rows with `procstatus` outside `{"0","57"}` are dropped; `"0"` and
   `"57"` survive.
4. Forced rows retain `distnr`, `rfid`, `procstatus` — the alignment regression.
5. W1–W4 populate from a `{"catalog": "allwise", "metadata": {...}}` match, and
   are `NaN` when no match is attached.
6. `feature_id`s come from the injected LUT; an unknown name maps to `NaN` and
   logs a warning.
7. `aid` is set to `oid` and `metadata["aid"]` is not `NaN`.
8. A message whose non-forced count is below `MIN_DETECTIONS_FEATURES` is
   filtered out, counting only non-forced rows in the merged list.

## Out of scope

Deliberately excluded; each blocks deployment independently and needs its own
change:

- **Output schema.** `schemas/feature_step/output.avsc` is still legacy ZTF
  (`oid: string`, `candid: array<string>`, `fid: string`) and does not match
  what `parse_output` emits for multisurvey (`oid: long`, `measurement_id`,
  `band: int`). A multisurvey ZTF output schema is needed.
- **Helm chart.** `charts/feature_step/values.yaml` is stale: consumes the
  `xmatch` topic, and defines no `SURVEY`, `DB_CONFIG`, `USE_XMATCH`,
  `XMATCH_CONFIG` or `XMATCH_CATALOGS`. A ZTF values file consuming the magstats
  ZTF topic is needed.
- **`settings.py` parity.** It never defines `SURVEY`, `DB_CONFIG`,
  `USE_XMATCH`, `XMATCH_CONFIG` or `XMATCH_CATALOGS`, yet `run_step.py:48` reads
  `STEP_CONFIG["DB_CONFIG"]`. Only the `CONFIG_FROM_YAML` path works today.
- **Integration tests.** `tests/integration/` still targets the legacy shape.
- **Scribe persistence.** `scribe_multisurvey` is owned separately and is not
  touched here. For the record, its `decode.py` currently routes
  `survey == "lsst" and step == "features"` (line 85) but has no ZTF equivalent,
  so the `{"step": "features", "survey": "ztf"}` command this step emits
  (`parsers.py:541-550`) falls through to the `raise` at line 100. ZTF feature
  rows will not reach `multisurvey_ztf.feature` until that is handled on the
  scribe side. The ZTF colour command (`update-ztf-object-features`, line 97)
  and the ZTF xmatch command (line 88) already have handlers.

## Notes

**`sid` is emitted as a string, and that is fine.** `parsers.py:260` casts `sid`
to `str` when building rows, so `parse_scribe_payload` emits `"sid": "0"`
(`parsers.py:539`, `parsers.py:546`) against a `smallint` column. Nothing
coerces it on the way in — `LSSTFeatureCommand._format_data` takes
`sid = data["sid"]` verbatim
(`scribe_multisurvey/sql_scribe/sql/command/commands.py:462`) — but psycopg2
sends a Python `str` as an untyped literal that Postgres implicitly casts in
`INSERT` context. The LSST path does exactly the same (`parsers.py:80`,
`parsers.py:576`) and runs in production, so ZTF inherits working behavior. No
change needed.
