# Hunter crop-then-stack and skipped alerts: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The hunter mapper crops each cutout before stacking and skips alerts with a cutout smaller than S; the step drops any alert the model returned no probability for.

**Architecture:** Three tasks in order. Task 1 in the `alerce_classifiers` submodule: `center_crop` on a 2-D plane, `HunterMapper.usable`, `predict` filters with it. Task 2 in `rubin_stamp_classifier_step`: `execute` keeps only messages whose oid the model returned. Task 3 bumps the submodule pointer on the pipeline branch.

**Tech Stack:** Python 3.11, numpy, pandas, torch (submodule), pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-hunter-crop-then-stack-design.md`

## Global Constraints

- Pipeline worktree `/Users/panchoandrades/Desktop/repos/pipeline-hunter`, branch `feat/stamp-hunter-classifier` (PR #633). Submodule checkout `/Users/panchoandrades/Desktop/repos/pipeline-hunter/alerce_classifiers`, branch `feat/hunter-crop-then-stack` (based on c7ada71, pushed, no PR yet; the draft PR opens on the first push of Task 1, base `feat/rubin-model-owns-asteroid-rule`).
- Interpreter for both repos: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python` (torch 2.14.0). Submodule tests run from the submodule dir; step tests from `rubin_stamp_classifier_step/`.
- Every new or changed line of code carries a `#` comment; every public function and class a docstring. Holds in tests too.
- S = `crop_size` = 30 in every test. Class columns `not_candidate`, `candidate`.
- Deletion of `test_center_crop_zero_pads_a_smaller_stamp` and `test_center_crop_keeps_the_middle_of_a_larger_stamp` in `tests/unit/test_hunter_mapper.py` is authorized by the user (chat, 2026-10-29). No other existing test changes.
- Warning logs: one `logging.warning` per skipped alert in the mapper (oid and the three shapes); one `logging.warning` per batch in the step listing the skipped oids.
- Green: submodule `pytest tests/unit/test_hunter_mapper.py tests/unit/test_hunter_model.py tests/unit/test_hunter_skip.py -q` all pass; step `test_delta.py check rubin_stamp_classifier_step` exit 0 (baseline: 0 failures).

## Review Focus

- A cutout that is exactly S x S must be usable and come out unchanged (pinned in Task 1 test (b), the 30x30 oid).
- A plane with NaN or inf pixels is still usable; `normalize` fills them (unchanged behaviour; Task 1 test (a) uses a NaN pixel in one plane).
- `usable` on an empty stamps frame returns an empty Index and `predict` returns the empty frame without calling the network (Task 1 test (d) covers the all-unusable case, which is the same path).
- The step's skip must not reorder kept messages (Task 2 test asserts order).
- The rubin stub returns every oid, so Task 2 must leave all existing step tests green (full suite in Task 2 Step 7).

---

### Task 1: mapper crops per plane, `usable`, `predict` filters

**Files:**
- Modify: `alerce_classifiers/alerce_classifiers/hunter/mapper.py` (`center_crop`, `HunterMapper.preprocess`, new `HunterMapper.usable`)
- Modify: `alerce_classifiers/alerce_classifiers/hunter/model.py:73-83` (`predict`)
- Delete (user-authorized): the two `center_crop` tests in `alerce_classifiers/tests/unit/test_hunter_mapper.py:19-39`
- Create: `alerce_classifiers/tests/unit/test_hunter_skip.py`

**Interfaces:**
- Produces: `center_crop(plane: np.ndarray, size: int) -> np.ndarray` on `(H, W)`, no padding, precondition `H >= size and W >= size`. `HunterMapper.usable(stamps: pd.DataFrame) -> pd.Index`. `HunterMapper.preprocess(stamps) -> np.ndarray (B, 3, S, S)` unchanged signature. `HunterClassifierModel.predict` returns probabilities indexed by `usable(stamps)`.

- [ ] **Step 1: Delete the two authorized tests**

Remove `test_center_crop_keeps_the_middle_of_a_larger_stamp` and `test_center_crop_zero_pads_a_smaller_stamp` from `tests/unit/test_hunter_mapper.py` (both pass a 3-D array to `center_crop`). Leave every other test and helper in that file as is.

- [ ] **Step 2: Write the failing tests `tests/unit/test_hunter_skip.py`**

`tests/` and `tests/unit/` are packages in the submodule: import `CONFIG`, `HPARAMS`, `CROP`, `STAMP_COLS`, `dto` and `stamp` from `tests.unit.test_hunter_model`, and define a module-scoped `model` fixture that builds a checkpoint the same way that file's `checkpoint_path` fixture does (same `DeepHitsD4` construction, `torch.save` of `state_dict`, `hparams`, `config` under `<tmp>/0.1.0/best_model.pt`). Use `caplog` for the warnings.

```python
def test_preprocess_crops_each_plane_then_stacks_planes_of_different_shapes():
    science, template, difference = ramp(37, 37), ramp(33, 41), ramp(30, 30)
    science[0, 0] = np.nan                                   # an invalid pixel is still usable
    stamps = frame({"visit_image": [science], "reference_image": [template], "difference_image": [difference]}, [11])
    out = HunterMapper(crop_size=30, nan_fill=0.0, norm_clip=10.0).preprocess(stamps)
    assert out.shape == (1, 3, 30, 30) and out.dtype == np.float32
    expected = normalize(np.stack([center_crop(science, 30), center_crop(template, 30), center_crop(difference, 30)]), 0.0, 10.0)
    np.testing.assert_allclose(out[0], expected)

def test_usable_keeps_only_alerts_whose_planes_are_2d_and_at_least_crop_size(caplog):
    stamps = frame({col: [ramp(30, 30), ramp(29, 30), np.ones(30, dtype=np.float32)] for col in STAMP_COLS}, [11, 12, 13])
    with caplog.at_level(logging.WARNING):
        kept = HunterMapper(crop_size=30, nan_fill=0.0, norm_clip=10.0).usable(stamps)
    assert kept.tolist() == [11]
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "12" in warnings[0] and "(29, 30)" in warnings[0]
    assert "13" in warnings[1] and "(30,)" in warnings[1]

def test_predict_returns_rows_only_for_usable_alerts(model):
    probs = model.predict(dto([11, 12], sizes=[30, 29])).probabilities
    assert probs.index.tolist() == [11]
    assert list(probs.columns) == ["not_candidate", "candidate"]

def test_predict_returns_an_empty_frame_when_nothing_is_usable(model):
    probs = model.predict(dto([12], sizes=[29])).probabilities
    assert probs.empty
    assert list(probs.columns) == ["not_candidate", "candidate"]
```

`ramp(h, w)` and `frame(columns, oids)` are small helpers in the new file (`frame` builds a DataFrame with `pd.Index(oids, name="oid")`). `dto(oids, sizes=...)` from `test_hunter_model` already makes square stamps of the given size per oid.

- [ ] **Step 3: Run to see them fail**

Run: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit/test_hunter_skip.py -q` from the submodule dir.
Expected: test 1 fails (`center_crop` on a 2-D array hits `_, h, w = arr.shape`), test 2 fails (`AttributeError: usable`), tests 3 and 4 fail (`np.stack` ValueError or 2 rows returned).

- [ ] **Step 4: Implement in `mapper.py`**

`center_crop(plane, size)`: `h, w = plane.shape`; return `plane[top:top+size, left:left+size]` with `top = (h - size) // 2`, `left = (w - size) // 2`. No padding branch.
`HunterMapper.usable(stamps) -> pd.Index`: for each row, `shapes = [np.shape(row[col]) for col in STAMP_COLUMNS]`; usable when every shape has `len == 2` and both dims `>= self.crop_size`; otherwise `logging.warning(f"hunter: skipping oid {oid}: cutout shapes {shapes} (need 2-D, >= {S}x{S})")`. Return `pd.Index([...], name=stamps.index.name)`.
`preprocess`: per row, `planes = [center_crop(np.asarray(row[col], dtype=np.float32), self.crop_size) for col in STAMP_COLUMNS]`, then `np.stack(planes)`, then `normalize`. Return `np.stack(batch)` when batch is non-empty, else `np.empty((0, 3, S, S), dtype=np.float32)`. Update the module docstring (crop per plane, skip rule).

- [ ] **Step 5: Implement in `model.py` `predict`**

`kept = self.mapper.usable(stamps)`; `usable_stamps = stamps.loc[kept]`; if `kept` is empty, `probs = np.empty((0, len(self.CLASSES)), dtype=np.float32)` without touching the network; else the existing batched loop over `self.mapper.preprocess(usable_stamps)`. Return `self.mapper.postprocess(probs, index=kept, class_names=self.CLASSES)`.

- [ ] **Step 6: Run the submodule suite**

Run: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit/test_hunter_mapper.py tests/unit/test_hunter_model.py tests/unit/test_hunter_skip.py -q`
Expected: all pass (`test_real_artifact_loads_and_scores_an_alert` skips without `TEST_HUNTER_MODEL_PATH`; that is fine).

- [ ] **Step 7: Commit on the submodule branch**

```bash
git add alerce_classifiers/hunter/mapper.py alerce_classifiers/hunter/model.py tests/unit/test_hunter_mapper.py tests/unit/test_hunter_skip.py
git commit -m "feat(hunter): crop each cutout before stacking, skip alerts with a cutout smaller than the crop"
```

### Task 2: the step drops alerts the model returned no probability for

**Files:**
- Modify: `rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py:228-256` (`execute`)
- Create: `rubin_stamp_classifier_step/tests/unit/test_skipped_alerts.py`

**Interfaces:**
- Consumes: nothing from Task 1 at code level (the stub model stands in); the contract is "`output_dto.probabilities.index` may be a subset of the input oids".
- Produces: `execute` returns output messages only for oids in `probabilities.index`, input order preserved.

- [ ] **Step 1: Write the failing test `tests/unit/test_skipped_alerts.py`**

A `SkippingStubModel` subclass of `tests.unit.stub_model.StubModel` whose `predict` drops the row of `SKIPPED_OID` from the frame `StubModel.predict` returns. Build the step like `tests/unit/test_ss_object_handling.py` does (same config shape, `RecordingProducer` with `OUTPUT_SCHEMA_PATH`, `PSQLConnection` and `get_taxonomy_by_classifier_id` patched) but with `MODEL_CONFIG.CLASS` pointing at the new stub, and reuse its `processed(oid, sid)` message builder (import it).

```python
def test_execute_emits_no_message_for_an_oid_the_model_skipped(step, caplog):
    with caplog.at_level(logging.WARNING):
        out = step.execute([processed(1, 1), processed(SKIPPED_OID, 1), processed(3, 1)])
    assert [m["oid"] for m in out] == [1, 3]                     # order kept, skipped oid gone
    assert any(str(SKIPPED_OID) in r.message for r in caplog.records if r.levelno == logging.WARNING)

def test_post_execute_only_sees_the_kept_messages(step):
    out = step.execute([processed(1, 1), processed(SKIPPED_OID, 1)])
    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:
        step.post_execute(out)
    assert [p["oid"] for p in store.call_args.kwargs["predictions"]] == [1]
```

- [ ] **Step 2: Run to see them fail**

Run: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit/test_skipped_alerts.py -q` from `rubin_stamp_classifier_step/`.
Expected: both fail with `KeyError: <SKIPPED_OID>` from `predicted_probabilities.loc[oid]`.

- [ ] **Step 3: Implement in `step.py` `execute`**

Before the loop: `skipped = [m["oid"] for m in messages if m["oid"] not in predicted_probabilities.index]`; if `skipped`, `logging.warning(f"model returned no probability for {len(skipped)} alert(s), dropping oids {skipped}")`. In the loop, `continue` on a skipped oid. Everything else unchanged.

- [ ] **Step 4: Run the new test**

Run: same command as Step 2. Expected: 2 passed.

- [ ] **Step 5: Run the suite against the baseline**

Run from the worktree root: `~/.pyenv/versions/feature-step/bin/python /Users/panchoandrades/Desktop/repos/pipeline/.claude/scripts/test_delta.py check rubin_stamp_classifier_step`
Expected: exit 0, `new failures: 0`.

- [ ] **Step 6: Commit on the pipeline branch**

```bash
git add rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py rubin_stamp_classifier_step/tests/unit/test_skipped_alerts.py
git commit -m "feat(rubin_stamp_classifier_step): drop alerts the model returned no probability for"
```

### Task 3: submodule bump

**Files:**
- Modify: submodule pointer `alerce_classifiers` on the pipeline branch

- [ ] **Step 1:** After the submodule PR has its review, from the worktree root: `git add alerce_classifiers && git commit -m "chore(alerce_classifiers): hunter mapper crops each cutout before stacking, skips small cutouts"`.
- [ ] **Step 2:** `git diff HEAD~1 --submodule=log` shows exactly the Task 1 commit. Run the step suite once more (Task 2 Step 5 command): exit 0.
