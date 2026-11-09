# feature_step: one package per survey, a step that does not branch on survey

Date: 2026-11-04. Package: `feature_step`. Base branch: `multisurvey`.

Today `features/step.py` has 10 `self.survey == ...` branches and
`features/utils/parsers.py` (746 lines) holds five ztf/lsst function pairs that
differ in a handful of lines each. Every survey repeats its band table three times.
Adding a survey means touching both files in a dozen places. This refactor moves
every survey-specific decision behind one class per survey, so that a new survey is
a new directory plus one registry line.

## Goal

A `Survey` contract in `features/surveys/base.py` that `FeatureStep` calls instead of
branching, with `ztf` and `lsst` implemented under `features/surveys/<name>/` and the
code they share in `features/surveys/common.py`. Behaviour on the wire (scribe
commands, output messages, DB reads) is identical to `multisurvey` today.

## Non-goals

- No change to any Avro schema, scribe command envelope, output message key, LUT
  query, `sid`/`tid` value, band code, or feature name.
- No change to `lc_classifier` (preprocessors, extractors, `AstroObject`).
- No change to `features/database.py`, `settings.py`, `scripts/run_step.py`,
  `features/offline/`, the Dockerfile or the chart.
- No fix of behaviour found odd on the way (for example the `sid == 2` skip in
  `get_xmatch_info`, or xmatch results attached only for `allwise`). Each is kept
  as is and listed in the PR body under "noticed, not changed".
- No deletion of code in this PR's main commits. Dead-code removal, if any, is one
  separate final commit that the human can drop.
- No new survey. The contract is shaped by ztf and lsst only.

## Design

```
features/
  step.py                 # no survey branches; holds self.survey: Survey
  surveys/
    __init__.py
    base.py               # class Survey (contract below) + SurveyBands
    common.py             # shared helpers (listed below)
    registry.py           # SURVEYS = {"ztf": ZTFSurvey, "lsst": LSSTSurvey}; get_survey(name)
    ztf/
      __init__.py         # class ZTFSurvey(Survey)
      bands.py            # BANDS = SurveyBands({"g": 1, "r": 2, "i": 3}, pairs={"g,r": 12})
      astro_object.py     # detections_to_astro_object (moved, unchanged body)
      scribe.py           # parse_scribe_payload, prepare_ao_features_for_db
      output.py           # parse_output
    lsst/
      (same four files, lsst bodies)
  utils/parsers.py        # shim: re-exports the names below from features.surveys
```

`Survey` contract (one attribute or method each, all overridden per survey):

| member | ztf today | lsst today |
|---|---|---|
| `name` | `"ztf"` | `"lsst"` |
| `sid`, `tid` | 0, 0 | 1, 1 |
| `bands: SurveyBands` | g r i; pair g,r=12 | u g r i z y = 6 1 2 3 4 5; pairs 61 12 23 34 45 |
| `id_column` | `"measurement_id"` | `"measurement_id"` |
| `needs_references` | True | False |
| `make_preprocessor()` | `ZTFLightcurvePreprocessor(drop_bogus=True)` | `LSSTLightcurvePreprocessor()` |
| `make_extractor()` | `ZTFFeatureExtractor()` | `LSSTFeatureExtractor()` |
| `epochs_from_message(msg) -> list` | detections + previous_detections + forced_photometries, then `discard_bogus_detections` | sources + previous_sources, drop rows without `band` |
| `forced_from_message(msg) -> list` | `[]` | `msg.get("forced_sources")` |
| `build_astro_object(dets, forced, xmatch, references_db)` | stamps `aid = oid`, calls the ztf parser | calls the lsst parser |
| `scribe_commands(aos, version, group, lut) -> dict` | update_object + upserting_features | upserting_features only |
| `output_messages(aos, msgs, candids) -> list` | ztf keys | lsst keys |

`SurveyBands` derives from one table: `fid_for_db(band_str) -> int` (today
`fid_mapper_for_db` / `_lsst`), `suffix(band_str) -> str` (today the `fid_map` /
`suffix_map` literals in `parse_output*`), `letter(fid_int) -> str` (today
`band_map_inverse`). Unknown band gives `0` / `""`. For lsst this is today's
result; for ztf today's `parse_output` raises `AttributeError` on an unknown band
and produces no output message, unreachable with the current extractor, and the
refactor emits the feature under its bare name instead.

`common.py` holds the bodies that are copied today: W1-W4 from an allwise match,
`last_mjd` over detections and forced, `prepare_features_for_db(ao, lut, bands,
keep_name)` (the two `prepare_ao_features_for_db*` collapse into one with the band
mapper injected and `keep_name` deciding whether `name` survives), and
`flatten_features_for_output(ao, bands)`.

`FeatureStep` after the change: `__init__` does `self.survey = get_survey(config["SURVEY"])`,
then one block (version id, LUT, xmatch client) using `survey.sid`/`survey.tid`.
`pre_execute`, `execute`, `produce_to_scribe` call the contract and contain no
survey name. `_get_sql_references` stays a method on `FeatureStep` with its name
(tests patch it) and is called only when `survey.needs_references`.

Dispatch is `get_survey(name)`; an unknown name raises `ValueError` naming the
known surveys at construction time, instead of today's silent step with no
extractor.

## Acceptance criteria

1. `grep -n 'survey ==\|survey==\|== "ztf"\|== "lsst"' feature_step/features/step.py`
   prints nothing.
2. `features/utils/parsers.py` has no function bodies: it only re-exports
   `detections_to_astro_object`, `detections_to_astro_object_lsst`,
   `get_bogus_flags_for_each_detection`, `prepare_ao_features_for_db`,
   `prepare_ao_features_for_db_lsst`, `parse_scribe_payload`,
   `parse_scribe_payload_lsst`, `parse_output`, `parse_output_lsst`,
   `fid_mapper_for_db`, `fid_mapper_for_db_lsst`, `DETECTION_KEYS_MAP`, and the
   existing tests import them unchanged.
3. `features/surveys/registry.py` maps `"ztf"` and `"lsst"`; `get_survey("x")`
   raises `ValueError` whose message lists both known names; `FeatureStep` with
   `SURVEY: "x"` raises at construction.
4. For every band string that appears in either survey today, `SurveyBands` returns
   the same DB code and the same output suffix as the current literals (table in the
   test: ztf `g r g,r None other`; lsst `u g r i z y u,g g,r r,i i,z z,y None other`).
5. Contract test, ztf: for the batch from `tests/message_factory_ztf_ms.generate_input_batch`
   (or the existing ztf factory the current tests use), with the feature extractor
   replaced by a stub that writes a fixed feature frame, the new step produces
   exactly the same scribe commands (list of dicts, order included) and the same
   output messages as the `multisurvey` code does, captured once into a fixture
   file committed with the test. Same for lsst with `generate_input_batch_lsst`.
   (Features are stochastic, so the extractor is stubbed; everything around it is
   deterministic and compared byte for byte.)
6. `build_astro_object` for one ztf and one lsst message returns an `AstroObject`
   whose `detections`, `forced_photometry`, `metadata` and `reference` frames are
   equal (`assert_frame_equal`, index included) to the ones the current parsers
   return for the same message.
7. `prepare_features_for_db` with `keep_name=True` equals today's
   `prepare_ao_features_for_db` output and with `keep_name=False` equals today's
   `prepare_ao_features_for_db_lsst`, on a features frame that includes a name the
   LUT does not know (the warning-and-drop path) and an `inf` value.
8. Every new or changed line carries a `#` comment; every public function and class
   has a docstring (reviewer checks).

## Done looks like

```
~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```
exits 0 (no failure outside the recorded baseline: 7 in `test_step_ztf`, 1 in
`test_step_lsst`), with new test files
`feature_step/tests/unittest/test_surveys_<task>.py`, one per task (`golden` for
criterion 5, `bands_registry` for 3-4, `common` for 7, `ztf` and `lsst` for 6,
`step` for 1-3), each seen red before the implementation exists. One file per task
because a committed test file is read-only for the agent (`edit_guard`). The ztf and lsst golden fixtures are generated from the
`multisurvey` code before any refactor commit, in the first task.
