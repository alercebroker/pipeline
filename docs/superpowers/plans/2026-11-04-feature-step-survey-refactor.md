# feature_step survey refactor: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every survey-specific decision in `feature_step` behind a `Survey` class per survey, so `FeatureStep` has no survey branches and the wire behaviour stays byte-identical.

**Architecture:** A golden test captures today's scribe commands and output messages for ztf and lsst (extractor stubbed) before anything moves. Then bottom-up: `SurveyBands` + `Survey` contract + registry, shared helpers in `common.py`, the ztf and lsst parser bodies moved into `features/surveys/<name>/`, and last `step.py` rewired to call the contract while `utils/parsers.py` becomes a re-export shim. The golden test and the existing tests must stay green at every commit.

**Tech Stack:** Python 3.10, pandas, `lc_classifier` (AstroObject, preprocessors, extractors), unittest/pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-feature-step-survey-refactor-design.md` (PR #646)

## Global Constraints

- No change to any Avro schema, scribe envelope, output key, LUT query, `sid`/`tid` (ztf 0/0, lsst 1/1), band code or feature name.
- No change to `features/database.py`, `settings.py`, `scripts/run_step.py`, `features/offline/`, Dockerfile, chart, or `lc_classifier`.
- Odd behaviour found on the way is kept and listed in the PR body under "noticed, not changed".
- Every new or changed line carries a `#` comment; every public function and class has a docstring.
- Committed test files are read-only for the agent: each task writes its tests in its own new file `feature_step/tests/unittest/test_surveys_<task>.py`.
- Test command, run from `feature_step/`:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/Users/panchoandrades/Desktop/repos/pipeline/alerce_classifiers ~/.pyenv/versions/feature-step/bin/python -m pytest <file> -q`
- Green for the package: `~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step` from the repo root exits 0 (baseline: 7 failures in `test_step_ztf`, 1 in `test_step_lsst`).
- Commit message trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A ztf object whose epochs are all bogus (rb below 0.55) is dropped by `pre_execute` today; the refactor must drop it too. Pinned by the third ztf golden message (Task 1).
2. A message with no xmatch, or an xmatch from a catalog other than allwise, yields W1-W4 = NaN and `xmatches` passed through unchanged. Pinned by the second ztf golden message and the lsst golden message (Task 1).
3. A feature name the LUT does not know is dropped with a warning, never sent; an empty LUT sends an upsert with `features: []`. Pinned by `Unknown_feat` in the golden stub (Task 1) and the empty-LUT case in Task 3.
4. ztf oids exceed 2**53; they must stay exact ints in metadata, scribe keys and output. Pinned by the golden oid `36028941624528297` compared as JSON ints (Task 1).
5. `SURVEY` missing or unknown in config silently builds a step with no extractor today; after the refactor it raises `ValueError` at construction. Pinned in Task 2 (`get_survey`) and Task 6 (`FeatureStep`).

---

### Task 1: Golden capture of today's behaviour (criterion 5)

**Files:**
- Create: `feature_step/tests/message_factory_lsst_ms.py`
- Create: `feature_step/tests/unittest/test_surveys_golden.py`
- Create: `feature_step/tests/unittest/fixtures/golden_ztf.json`, `feature_step/tests/unittest/fixtures/golden_lsst.json` (generated, committed)

**Interfaces:**
- Consumes: `tests.message_factory_ztf_ms.generate_message(oid, bands, n_detections, n_previous_detections, n_forced, seed, with_xmatch)`; `FeatureStep(config, db_sql)`; `step.pre_execute`, `step.execute`, `step.post_execute`; `step.scribe_producer.producer.produce.call_args_list`.
- Produces: `generate_message_lsst(oid=..., seed=..., with_xmatch=False) -> dict` in `message_factory_lsst_ms.py`; `capture(survey: str) -> dict` and `StubExtractor` in the test module, reused by no later task (the golden test is the contract).

- [ ] **Step 1: Write `generate_message_lsst` in `feature_step/tests/message_factory_lsst_ms.py`**

Signature: `generate_message_lsst(oid: int = 1152921504606847001, bands=("u","g","r","i"), n_sources=5, n_previous=3, n_forced=4, seed=7, with_xmatch=False) -> dict`. Message keys: `oid`, `sid` (1), `measurement_id` (list of the `sources` ids), `meanra`, `meandec`, `sources`, `previous_sources`, `forced_sources`. Each epoch dict carries exactly the keys the lsst parser reads: `oid, sid, mjd, ra, dec, psfFlux, psfFluxErr, scienceFlux, scienceFluxErr, tid, band, measurement_id`. Values from `random.Random(seed)`. Pin three edge rows: one source with `band: None` (dropped in pre_execute), one source with `scienceFlux: -5.0` (dropped from the magnitude half), and `forced_sources` non-empty. `with_xmatch=True` attaches `tests.message_factory_ztf_ms.allwise_match(oid)` under `xmatches`.

- [ ] **Step 2: Write the golden test `feature_step/tests/unittest/test_surveys_golden.py`**

```python
STUB_FEATURES = {
    "ztf": [("Amplitude","g",1.5), ("Amplitude","r",2.5), ("g-r_max","g,r",0.3),
            ("g-r_mean","g,r",0.2), ("g-r_max_corr","g,r",0.25), ("g-r_mean_corr","g,r",0.15),
            ("Power_rate_1_4",None,0.7), ("Mean","g",float("nan")), ("Unknown_feat","g",1.0),
            ("Skew","r",float("inf"))],
    "lsst": [("Amplitude","u",1.1), ("Amplitude","g",1.2), ("u-g_mean","u,g",0.4),
             ("Power_rate_1_2",None,0.6), ("Mean","i",float("nan")), ("Unknown_feat","r",1.0)],
}
LUT = {"ztf": {1:"Amplitude", 2:"g-r_max", 3:"g-r_mean", 4:"g-r_max_corr", 5:"g-r_mean_corr",
               6:"Power_rate_1/4", 7:"Mean", 8:"Skew"},
       "lsst": {1:"Amplitude", 2:"u-g_mean", 3:"Power_rate_1/2", 4:"Mean"}}

class StubExtractor:
    """Writes STUB_FEATURES[survey] into every AstroObject instead of computing, and records
    each AstroObject's frames (detections, forced_photometry, metadata, reference) as
    `to_dict("split")` under self.frames, keyed by oid, so criterion 6 has an oracle."""
    def compute_features_batch(self, astro_objects, progress_bar=False): ...

def messages_for(survey):
    # ztf: generate_message(seed=1, with_xmatch=True);
    #      generate_message(oid=36028941624528298, seed=2, n_forced=0);
    #      generate_message(oid=36028941624528299, seed=3) with every epoch's rb set to 0.1
    #      (detections, previous_detections and forced_photometries) -> dropped by pre_execute.
    # lsst: generate_message_lsst(seed=7, with_xmatch=True);
    #       generate_message_lsst(oid=1152921504606847002, seed=8).

def references_df():
    # one row: {"oid": 36028941624528297, "rfid": 1, "sharpnr": 0.1, "chinr": 0.5}; columns oid,rfid,sharpnr,chinr

def capture(survey) -> dict:
    # build the step as tests/unittest/test_step_ztf_multisurvey.build_step does (same config dicts,
    # SURVEY=survey, db_sql=MagicMock, scribe_producer=MagicMock, extractor_version=1,
    # feature_name_lut=LUT[survey]); step.feature_extractor = StubExtractor(survey);
    # mock.patch.object(FeatureStep, "_get_sql_references", return_value=references_df());
    # prepared = step.pre_execute(messages_for(survey)); out = step.post_execute(step.execute(prepared))
    # return normalize({"prepared_oids": [m["oid"] for m in prepared],
    #                   "astro_objects": stub.frames,   # oid -> {detections, forced_photometry, metadata, reference} split dicts
    #                   "scribe": [{"topic": kw["topic"], "key": kw["key"].decode(), "value": json.loads(kw["value"])}
    #                              for _, kw in step.scribe_producer.producer.produce.call_args_list],
    #                   "output": out})

def normalize(obj):
    # json round trip with default= converting numpy scalars via .item(), pandas NA/NaT and float NaN to the
    # string "NaN", inf to "Inf"; the result is plain dicts/lists/str/int/float/None.

class GoldenTestCase(unittest.TestCase):
    def test_ztf_matches_golden(self): self._check("ztf")
    def test_lsst_matches_golden(self): self._check("lsst")
    def _check(self, survey):
        path = FIXTURES / f"golden_{survey}.json"
        got = capture(survey)
        if os.environ.get("FEATURE_STEP_WRITE_GOLDEN"):
            path.write_text(json.dumps(got, indent=1, sort_keys=True)); self.fail(f"wrote {path}")
        self.assertEqual(json.loads(path.read_text()), got)
```

- [ ] **Step 3: Run it, expect FAIL (fixture file missing)**

Run: `... -m pytest tests/unittest/test_surveys_golden.py -q`
Expected: 2 failed, `FileNotFoundError` on `golden_ztf.json` / `golden_lsst.json`.

- [ ] **Step 4: Generate the fixtures from today's code**

Run: `FEATURE_STEP_WRITE_GOLDEN=1 ... -m pytest tests/unittest/test_surveys_golden.py -q`
Expected: 2 failed with "wrote ...". Then `git diff --stat feature_step/features` prints nothing (no source touched). Open each fixture and check by eye: ztf has 2 `prepared_oids` (the all-bogus one is gone), scribe holds `update-ztf-object-features` and `features` commands, `Unknown_feat` is absent, `Skew` absent (inf → None → dropped by `notna`? record whatever is there), `Power_rate_1/4` present with its LUT id; lsst has 2 `prepared_oids`, only `features` commands, output keys `oid, measurement_id, detections, features`.

- [ ] **Step 5: Run it again, expect PASS**

Run: `... -m pytest tests/unittest/test_surveys_golden.py -q`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add feature_step/tests/message_factory_lsst_ms.py feature_step/tests/unittest/test_surveys_golden.py feature_step/tests/unittest/fixtures/
git commit -m "test(feature_step): golden capture of ztf and lsst scribe commands and output before the survey refactor"
```

---

### Task 2: `SurveyBands`, `Survey` contract, registry (criteria 3 [get_survey], 4)

**Files:**
- Create: `feature_step/features/surveys/__init__.py` (empty), `base.py`, `registry.py`, `ztf/__init__.py`, `lsst/__init__.py`
- Test: `feature_step/tests/unittest/test_surveys_bands_registry.py`

**Interfaces:**
- Produces, in `base.py`:
  - `class SurveyBands`: `__init__(self, singles: dict[str, int], pairs: dict[str, int], letters: dict[str, int] | None = None)`; `fid_for_db(band: str | None) -> int` over `singles | pairs` (unknown or None → 0); `suffix(band: str | None) -> str` (`"_<code>"`, None → `""`, unknown → `""`); `letter(fid: int) -> str | None` (inverse of `letters`, default `singles`); `singles_inverse` property (inverse of `letters`, what today's `band_map_inverse` is). ztf passes `letters={"g":1,"r":2,"i":3}` because today's `band_map` knows `i` but `fid_mapper_for_db` does not.
  - `class Survey` (plain class, `NotImplementedError` in every method): attributes `name: str`, `sid: int`, `tid: int`, `id_column: str = "measurement_id"`, `needs_references: bool`, `bands: SurveyBands`; methods `make_preprocessor(self)`, `make_extractor(self)`, `epochs_from_message(self, message: dict) -> list[dict]`, `forced_from_message(self, message: dict) -> list[dict]`, `build_astro_object(self, detections: list[dict], forced: list[dict], xmatches: dict | None, references_db) -> AstroObject`, `scribe_commands(self, astro_objects: list, features_version, features_group, feature_name_lut) -> dict`, `output_messages(self, astro_objects: list, messages: list[dict], candids: dict) -> list[dict]`.
- Produces, in `registry.py`: `SURVEYS: dict[str, type[Survey]]`; `get_survey(name: str) -> Survey` raising `ValueError(f"unknown SURVEY {name!r}; known: {sorted(SURVEYS)}")`.
- Produces, in `ztf/__init__.py`: `class ZTFSurvey(Survey)` with `name="ztf", sid=0, tid=0, needs_references=True, bands=SurveyBands({"g":1,"r":2}, {"g,r":12}, letters={"g":1,"r":2,"i":3})`, methods still `NotImplementedError` except `make_preprocessor` (`ZTFLightcurvePreprocessor(drop_bogus=True)`) and `make_extractor` (`ZTFFeatureExtractor()`). In `lsst/__init__.py`: `class LSSTSurvey(Survey)` with `name="lsst", sid=1, tid=1, needs_references=False, bands=SurveyBands({"u":6,"g":1,"r":2,"i":3,"z":4,"y":5}, {"u,g":61,"g,r":12,"r,i":23,"i,z":34,"z,y":45})`, `LSSTLightcurvePreprocessor()`, `LSSTFeatureExtractor()`.

- [ ] **Step 1: Write the failing tests**

```python
ZTF_DB = {"g":1, "r":2, "g,r":12, None:0, "x":0}           # today's fid_mapper_for_db
ZTF_SUFFIX = {"g":"_1", "r":"_2", "g,r":"_12", None:"", "x":""}   # today's fid_map in parse_output
LSST_DB = {"u":6,"g":1,"r":2,"i":3,"z":4,"y":5,"u,g":61,"g,r":12,"r,i":23,"i,z":34,"z,y":45,None:0,"x":0}
LSST_SUFFIX = {"u":"_6","g":"_1","r":"_2","i":"_3","z":"_4","y":"_5","u,g":"_61","g,r":"_12","r,i":"_23","i,z":"_34","z,y":"_45",None:"","x":""}

def test_ztf_bands_match_today(self):   # one assertEqual per entry of ZTF_DB and ZTF_SUFFIX via ZTFSurvey.bands
def test_lsst_bands_match_today(self):  # same for LSST_*
def test_letter_inverts_singles(self):  # LSSTSurvey.bands.letter(6) == "u"; ZTFSurvey.bands.letter(3) == "i"
def test_registry_has_both(self):       # isinstance(get_survey("ztf"), ZTFSurvey); isinstance(get_survey("lsst"), LSSTSurvey)
def test_registry_unknown_raises(self): # with assertRaises(ValueError) as cm: get_survey("x"); "ztf" in str(cm.exception) and "lsst" in str(cm.exception)
def test_sid_tid(self):                 # (ztf.sid, ztf.tid) == (0, 0); (lsst.sid, lsst.tid) == (1, 1)
```

Add `"i": 0` to `ZTF_DB` and `"i": ""` to `ZTF_SUFFIX`: today's `fid_mapper_for_db` and `fid_map` do not know `i`, only `band_map` does (see `letters` in Interfaces).

- [ ] **Step 2: Run, expect FAIL with `ModuleNotFoundError: features.surveys`**

- [ ] **Step 3: Implement `base.py`, `registry.py`, `ztf/__init__.py`, `lsst/__init__.py` as in Interfaces**

- [ ] **Step 4: Run, expect 6 passed. Run the golden test, expect 2 passed (nothing else changed).**

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/surveys feature_step/tests/unittest/test_surveys_bands_registry.py
git commit -m "feat(feature_step): Survey contract, SurveyBands and registry with ztf and lsst stubs"
```

---

### Task 3: Shared helpers in `common.py` (criterion 7)

**Files:**
- Create: `feature_step/features/surveys/common.py`
- Modify: `feature_step/features/utils/parsers.py` (`prepare_ao_features_for_db`, `prepare_ao_features_for_db_lsst` become one-line delegations; the W1-W4 and last_mjd blocks in both `detections_to_astro_object*` call the helpers)
- Test: `feature_step/tests/unittest/test_surveys_common.py`

**Interfaces:**
- Consumes: `SurveyBands` (Task 2).
- Produces, in `common.py`:
  - `wise_magnitudes(xmatches: dict | None) -> tuple[float, float, float, float]`: W1-W4 from `xmatches["metadata"]["w<n>mpro"]["Float64"]` when `xmatches` is not None and `xmatches.get("catalog") == "allwise"`, else four `np.nan`.
  - `last_mjd_of(detections: list[dict], forced: list[dict]) -> float`: max `mjd` over both lists, `np.nan` when neither has one.
  - `prepare_features_for_db(astro_object, feature_name_lut: dict, bands: SurveyBands, keep_name: bool) -> pd.DataFrame`: body of today's `prepare_ao_features_for_db` with `bands.fid_for_db` in place of the mapper; drops `fid`, and also `name` when `keep_name` is False. Warning logger name stays `alerce.FeatureStep`.
  - `flatten_features_for_output(astro_object, bands: SurveyBands) -> dict[str, float | None]`: name + `bands.suffix(fid)`, sorted by name, `-` → `_`, NaN/None → None, values as float. (Used by Tasks 4 and 5; today's two bodies differ only in `astype(np.double)` vs `astype(float)` and the None check, which give the same result.)

- [ ] **Step 1: Write the failing tests**

```python
def features_frame():  # name,fid,value,sid,version rows: ("Amplitude","g",1.5), ("Power_rate_1_4",None,0.7),
                       # ("Unknown_feat","g",1.0), ("Mean","g",nan), ("Skew","r",inf), ("g-r_max","g,r",0.3)
LUT = {1:"Amplitude", 2:"Power_rate_1/4", 3:"Mean", 4:"Skew", 5:"g-r_max"}

def test_prepare_keep_name_equals_today_ztf(self):
    # assert_frame_equal(prepare_features_for_db(ao, LUT, ZTFSurvey.bands, keep_name=True),
    #                    parsers.prepare_ao_features_for_db(ao, LUT))   <- call the ORIGINAL body: copy it
    #                    verbatim into the test module as _today_prepare_ztf before Step 3 replaces it
def test_prepare_drop_name_equals_today_lsst(self):   # same against a verbatim copy _today_prepare_lsst with LSSTSurvey.bands
def test_prepare_drops_unknown_with_warning(self):    # assertLogs("alerce.FeatureStep", "WARNING"); "Unknown_feat" not in result["name"]
def test_prepare_empty_lut_gives_empty_frame(self):   # len(result) == 0 with lut={}
def test_wise_none_and_other_catalog(self):           # wise_magnitudes(None) and ({"catalog":"gaia",...}) are 4 NaN
def test_wise_allwise(self):                          # allwise_match(1) from message_factory_ztf_ms -> (15.1, 14.9, 12.5, 9.1)
def test_last_mjd(self):                              # ([{"mjd":1.0}], [{"mjd":3.0}]) -> 3.0; ([], []) -> NaN; ([{"x":1}], []) -> NaN
def test_flatten_ztf_and_lsst(self):                  # ztf frame -> {"Amplitude_1":1.5, "Mean_1":None, "Power_rate_1_4":0.7, "Skew_2":None?...}
                                                      # (inf: today replace() turns inf into None, then astype(float) -> NaN -> None; assert None)
                                                      # "g_r_max_12": 0.3 ; lsst: ("Amplitude","u") -> "Amplitude_6"
```

The two `_today_*` copies are the only places this plan copies a body: they are the oracle and are deleted in Task 6, Step 5.

- [ ] **Step 2: Run, expect FAIL with `ImportError` on `features.surveys.common`**

- [ ] **Step 3: Implement `common.py`; make `prepare_ao_features_for_db(ao, lut)` return `prepare_features_for_db(ao, lut, ZTFSurvey.bands, keep_name=True)` and the lsst one `keep_name=False` with `LSSTSurvey.bands`; replace the W1-W4 and last_mjd blocks in both `detections_to_astro_object*` with the helpers**

- [ ] **Step 4: Run this file, the golden test and `test_step_ztf_multisurvey.py`; expect all green (that file imports `prepare_ao_features_for_db*` directly)**

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/surveys/common.py feature_step/features/utils/parsers.py feature_step/tests/unittest/test_surveys_common.py
git commit -m "refactor(feature_step): shared feature-row, WISE and last_mjd helpers in surveys/common"
```

---

### Task 4: ztf package (criterion 6, ztf half)

**Files:**
- Create: `feature_step/features/surveys/ztf/astro_object.py`, `scribe.py`, `output.py`
- Modify: `feature_step/features/surveys/ztf/__init__.py` (fill the `ZTFSurvey` methods)
- Modify: `feature_step/features/utils/parsers.py` (ztf bodies removed, names re-exported from `features.surveys.ztf.*`)
- Test: `feature_step/tests/unittest/test_surveys_ztf.py`

**Interfaces:**
- Consumes: `common.py` helpers (Task 3), `ZTFSurvey.bands` (Task 2), `discard_bogus_detections` from `lc_classifier.features.core.base`.
- Produces:
  - `astro_object.py`: `detections_to_astro_object(detections, forced, xmatches, references_db) -> AstroObject` (body moved verbatim; `band_map_inverse` replaced by `bands.singles_inverse`), plus the moved helpers `get_bogus_flags_for_each_detection`, `get_reference_for_each_detection`, `get_new_references_from_message`, `add_mag_and_flux_columns`, and `DETECTION_KEYS_MAP` (moved to `common.py`, re-exported).
  - `scribe.py`: `parse_scribe_payload(astro_objects, features_version, features_group, feature_name_lut) -> dict` (uses `prepare_features_for_db(..., keep_name=True)`); `fid_mapper_for_db(band) -> int` kept as `ZTFSurvey.bands.fid_for_db` alias.
  - `output.py`: `parse_output(astro_objects, messages, candids) -> list[dict]` (uses `flatten_features_for_output`; keeps `reference` records and the `oid_ao == oid` assert).
  - `ZTFSurvey.epochs_from_message(msg)`: `discard_bogus_detections(detections + previous_detections + forced_photometries)` with `or []` on each. `forced_from_message(msg)`: `[]`. `build_astro_object(dets, forced, xmatches, references_db)`: stamps `aid = oid` on every det (today's `execute` lambda) then calls `detections_to_astro_object`. `scribe_commands` → `parse_scribe_payload`; `output_messages` → `parse_output`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_astro_object_equals_golden(self):
    # message = generate_message(seed=1, with_xmatch=True); epochs = ZTFSurvey().epochs_from_message(message)
    # with index_column stamped as execute does; ao = ZTFSurvey().build_astro_object(epochs, [], message["xmatches"], references_df())
    # expected = fixtures/golden_ztf.json["astro_objects"]["36028941624528297"] (Task 1 oracle);
    # assert_frame_equal(ao.<frame>, pd.DataFrame(**expected[<frame>]), check_dtype=False) for the four frames,
    # after the same normalize() round trip on ao's frames (NaN -> "NaN")
def test_epochs_from_message_drops_bogus(self):   # message with every rb=0.1 -> epochs == []
def test_forced_is_empty(self):                   # ZTFSurvey().forced_from_message(message) == []
def test_scribe_and_output_delegate(self):        # ZTFSurvey().scribe_commands is parse_scribe_payload-equivalent:
                                                  # same dict for one stubbed ao (StubExtractor pattern from Task 1, LUT from Task 1)
```

- [ ] **Step 2: Run, expect FAIL (`NotImplementedError` / `ImportError`)**

- [ ] **Step 3: Move the bodies into the three modules, fill `ZTFSurvey`, make `utils/parsers.py` import the ztf names from `features.surveys.ztf.*`**

- [ ] **Step 4: Run this file, the golden test, `test_step_ztf_multisurvey.py` and `test_step_ztf.py` (baseline: 7 failures there, no new ones); expect green**

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/surveys/ztf feature_step/features/surveys/common.py feature_step/features/utils/parsers.py feature_step/tests/unittest/test_surveys_ztf.py
git commit -m "refactor(feature_step): ztf parsers move into features/surveys/ztf behind ZTFSurvey"
```

---

### Task 5: lsst package (criterion 6, lsst half)

**Files:**
- Create: `feature_step/features/surveys/lsst/astro_object.py`, `scribe.py`, `output.py`
- Modify: `feature_step/features/surveys/lsst/__init__.py`
- Modify: `feature_step/features/utils/parsers.py` (lsst bodies removed, names re-exported)
- Test: `feature_step/tests/unittest/test_surveys_lsst.py`

**Interfaces:**
- Consumes: Task 3 helpers, `LSSTSurvey.bands`.
- Produces: `detections_to_astro_object_lsst`, `add_mag_and_flux_lsst`, `flux_err_2_mag_err`, `fluxnjy2mag` in `astro_object.py`; `parse_scribe_payload_lsst` and `fid_mapper_for_db_lsst` in `scribe.py`; `parse_output_lsst` in `output.py`. `LSSTSurvey.epochs_from_message(msg)`: `sources + previous_sources` keeping rows whose `band` is not None (today's `pre_execute` arm). `forced_from_message(msg)`: `msg.get("forced_sources")` (today passes it through unchanged, possibly None; keep). `build_astro_object` calls `detections_to_astro_object_lsst(dets, forced, xmatches)`.

- [ ] **Step 1: Write the failing tests** (mirror of Task 4 with `generate_message_lsst`; `test_epochs_drops_no_band` asserts the `band: None` source is gone and the count equals `n_sources - 1 + n_previous`)

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Move the bodies, fill `LSSTSurvey`, re-export from `utils/parsers.py`**

- [ ] **Step 4: Run this file, the golden test, `test_step_lsst.py` (baseline: 1 failure); expect green**

- [ ] **Step 5: Commit**

```bash
git add feature_step/features/surveys/lsst feature_step/features/utils/parsers.py feature_step/tests/unittest/test_surveys_lsst.py
git commit -m "refactor(feature_step): lsst parsers move into features/surveys/lsst behind LSSTSurvey"
```

---

### Task 6: `FeatureStep` calls the contract; `parsers.py` is a shim (criteria 1, 2, 3 [step])

**Files:**
- Modify: `feature_step/features/step.py` (whole file)
- Modify: `feature_step/features/utils/parsers.py` (re-exports only; module docstring says so)
- Modify: `feature_step/tests/unittest/test_surveys_common.py` only to delete the two `_today_*` oracle copies (agent-created file, so editable) and point those two tests at the golden fixtures instead (or drop them: criterion 7 is still pinned by the other four tests).
- Test: `feature_step/tests/unittest/test_surveys_step.py`

**Interfaces:**
- Consumes: `get_survey` (Task 2), the full `Survey` contract (Tasks 4, 5).
- Produces: `FeatureStep.survey: Survey`; `FeatureStep.survey_name: str` (today's `self.survey` string, kept under a new name because `produce_xmatch_to_scribe` sends it); `_get_sql_references(self, oids)` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_unknown_survey_raises_at_construction(self):   # FeatureStep(config={..., "SURVEY": "x"}, db_sql=MagicMock()) raises ValueError
def test_missing_survey_raises_at_construction(self):   # config without SURVEY raises ValueError
def test_step_has_no_survey_branches(self):             # read features/step.py; assert no line matches r'survey\s*==|==\s*"(ztf|lsst)"'
def test_parsers_is_a_shim(self):                       # ast.parse(features/utils/parsers.py): no FunctionDef/ClassDef nodes; the 12 names
                                                        # from spec criterion 2 are importable from features.utils.parsers
def test_step_uses_survey_objects(self):                # build_step("ztf").survey is a ZTFSurvey; .survey_name == "ztf";
                                                        # build_step("lsst").lightcurve_preprocessor is an LSSTLightcurvePreprocessor
```

- [ ] **Step 2: Run, expect FAIL (`test_step_has_no_survey_branches`, `test_parsers_is_a_shim`, the raise tests)**

- [ ] **Step 3: Rewrite `step.py`**

`__init__`: `self.survey = get_survey(self.config.get("SURVEY"))` (a `None` name raises too); `self.survey_name = self.survey.name`; `self.id_column = self.survey.id_column`; preprocessor/extractor from `make_*`; `self.extractor_group = type(self.feature_extractor).__name__`; one block for `get_or_create_version_id` / `get_feature_name_lut` with `self.survey.sid`, `self.survey.tid`; xmatch client once. `pre_execute`: per message `filtered["detections"] = survey.epochs_from_message(message)`, then the single `has_enough_detections` filter; drop the `"TIENE LENGTH MAYOR A CERO"` log line (debug noise; list it in the PR body). `execute`: `references_db = self._get_sql_references(list(oids)) if self.survey.needs_references else None`; per message `ao = survey.build_astro_object(list(m), survey.forced_from_message(message), message.get("xmatches"), references_db)`. `produce_to_scribe`: one loop over `update_object` then one over `upserting_features`, no survey check. `produce_xmatch_to_scribe(..., survey)` keeps its signature; the call passes `self.survey_name`. Drop the unused imports (`get_sid`, `data_utils`, the four parser names, `discard_bogus_detections`, the four lc_classifier classes, `version`).

- [ ] **Step 4: Run this file, the golden test, and the full package check**

Run from the repo root: `~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step`
Expected: exit 0.

- [ ] **Step 5: Delete the `_today_*` oracle copies from `test_surveys_common.py`, rerun that file, expect green**

- [ ] **Step 6: Commit**

```bash
git add feature_step/features/step.py feature_step/features/utils/parsers.py feature_step/tests/unittest/test_surveys_step.py feature_step/tests/unittest/test_surveys_common.py
git commit -m "refactor(feature_step): FeatureStep dispatches through Survey; utils/parsers is a re-export shim"
```

---

### Task 7: Dead code, one droppable commit (spec non-goal: separate final commit)

**Files:**
- Delete: `feature_step/features/utils/data_utils.py`, `feature_step/features/core/utils/functions.py`, `feature_step/features/utils/metrics.py`, `feature_step/features/core/utils/__init__.py`, `feature_step/features/core/__init__.py` (the `core/` package becomes empty)
- Keep: `feature_step/utils/feature_io.py` (imported by `scripts/plot_features_distribution.py`)

- [ ] **Step 1: Confirm zero importers**

Run from `feature_step/`: `grep -rn "data_utils\|collapse_fid_columns\|get_sid\|features.core" --include='*.py' . | grep -v scripts/local_sim/out`
Expected: only the definitions inside the files being deleted.

- [ ] **Step 2: `git rm` the five files; run the package check; expect exit 0**

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(feature_step): drop unused data_utils, core/utils/functions and metrics.get_sid"
```

---

## PR body items to carry ("noticed, not changed")

- `get_xmatch_info` skips messages with `sid == 2` and only attaches `allwise` matches to messages.
- `produce_xmatch_to_scribe` defaults `sid` to 1 for an oid absent from the batch.
- lsst `forced_from_message` may return `None`; the parser iterates it, so a message without `forced_sources` would fail today as well.
- The `"TIENE LENGTH MAYOR A CERO"` info log was removed in Task 6 (log noise only).
- ztf `parse_output` gave a `None` feature name for a band outside `g`, `r`, `g,r`, `None`; through `SurveyBands.suffix` it now gives the bare name. Unreachable with today's extractor (spec amendment 8aa8715cc).
