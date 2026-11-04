# Hunter mapper: crop each cutout, then stack; skip alerts with a small cutout

Date: 2026-10-29. Closes the "hunter mapper stacks before cropping" item of the
PR #633 checklist (2026-09-21). Two repos: `alerce_classifiers` (mapper, model)
and `pipeline` (`rubin_stamp_classifier_step`, submodule bump).

## Goal

The hunter mapper center-crops each of the three cutouts (science, template,
difference) to S x S on its own and stacks them afterwards, so cutouts of
different shapes never break `np.stack`. An alert with any cutout smaller than
S in either dimension, or not 2-D, is skipped: the model returns no
probability for it, the step emits no output message, writes no DB row and
forwards nothing. Skips are logged once per alert with oid and shapes. Rubin
is untouched.

## Non-goals

- Padding small cutouts (training's `center_crop` zero-padded them; the user
  chose to skip instead, see the divergence note below).
- Changing normalization, the class list, batching or model loading.
- Telling the scribe or any downstream about skipped alerts.
- The other PR #633 checklist items.

## Acceptance criteria

`S` is the mapper's `crop_size` (30 in the shipped checkpoint).

1. `alerce_classifiers/hunter/mapper.py`: `center_crop(plane, size)` takes a
   2-D `(H, W)` array and returns its central `size x size` window; it no
   longer pads. `HunterMapper.preprocess(stamps)` converts each of the three
   columns to float32, crops each plane with `center_crop`, then stacks them
   into `(3, S, S)` and normalizes as today. Three planes with different
   shapes, all at least S x S, preprocess without error. Output shape and
   dtype unchanged: `(B, 3, S, S)` float32.
2. `HunterMapper.usable(stamps) -> pd.Index`: the oids (in input order) whose
   three cutouts are each 2-D with `H >= S` and `W >= S`. Each oid left out
   gets one `logging.warning` naming the oid and the three shapes.
3. `HunterClassifierModel.predict` scores only `stamps.loc[usable]` and
   returns probabilities indexed by those oids, in input order. When nothing
   is usable it returns an empty DataFrame with the two class columns and
   calls the network with nothing.
4. `rubin_stamp_classifier_step/step.py` `execute`: a message whose oid is
   not in `output_dto.probabilities.index` produces no output message; one
   `logging.warning` lists the skipped oids. `post_execute` is unchanged,
   so skipped alerts reach neither `store_probability` nor the SN forwarder.
   A model that returns every oid (rubin, the stub) behaves as before.
5. Submodule tests, new file `tests/unit/test_hunter_skip.py`: (a) three
   planes of shapes 37x37, 33x41, 30x30 preprocess to `(1, 3, 30, 30)`;
   (b) `usable` drops an oid with a 29x30 plane and one with a 1-D plane,
   keeps a 30x30 one, and logs a warning per dropped oid; (c) `predict` on a
   batch of one usable and one 29x30 alert returns one row, indexed by the
   usable oid; (d) `predict` on only unusable alerts returns an empty frame
   with columns `not_candidate`, `candidate`.
6. Step test, new file `rubin_stamp_classifier_step/tests/unit/test_skipped_alerts.py`,
   no DB, no broker: a stub model that returns probabilities for all but one
   oid makes `execute` return one message less, with the missing oid absent,
   and `post_execute` hands only the kept messages to `store_probability`
   and the forwarder.
7. Existing tests that encode the old behaviour are removed by the user:
   `tests/unit/test_hunter_mapper.py::test_center_crop_zero_pads_a_smaller_stamp`
   (padding is gone) and, if it calls `center_crop` with a 3-D array,
   `test_center_crop_keeps_the_middle_of_a_larger_stamp`. No other existing
   test changes.
8. Pipeline commit bumps the `alerce_classifiers` submodule pointer to the
   reviewed submodule commit.

## Done looks like

Submodule, from `alerce_classifiers/`, with the interpreter that has torch
(`~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python`
if it does; the plan verifies):
```
python -m pytest tests/unit/test_hunter_mapper.py tests/unit/test_hunter_model.py tests/unit/test_hunter_skip.py -q
```
green. Step, from the worktree root:
```
~/.pyenv/versions/feature-step/bin/python /Users/panchoandrades/Desktop/repos/pipeline/.claude/scripts/test_delta.py check rubin_stamp_classifier_step
```
exit 0 (baseline recorded 2026-10-29: 0 failures).

## Divergence from training (read this)

`alerce-hunter-classifier/src/hunter_classifier/download.py:80-83` dropped a
detection whose three planes differed in shape; `dataset.py:20-40` zero-padded
a plane smaller than S. Production after this spec: different shapes are fine
(each plane is cropped), small planes are skipped. So production is more
permissive than training on mismatched shapes and stricter on small ones. The
user chose both on 2026-10-29. A small cutout that training would have padded
and scored now gets no probability.

## Calls made here (check these)

- `usable` is a separate method rather than a tuple return from `preprocess`,
  so the existing `preprocess` tests keep their shape assertions and only the
  padding test goes.
- `predict` filters with `usable` before `preprocess`; `preprocess` itself
  assumes usable input and does not re-check.
- Step: the filter is "oid in `probabilities.index`", nothing hunter-specific,
  so rubin pays one index lookup per message and nothing else.
- Branches: submodule work on `feat/hunter-crop-then-stack` off
  `feat/rubin-model-owns-asteroid-rule`, draft PR into that branch in
  `alercebroker/alerce_classifiers`; pipeline work as commits on
  `feat/stamp-hunter-classifier` under PR #633 (step change, then the
  submodule bump once the submodule PR is reviewed).
- Deleting or editing the two existing mapper tests (criterion 7) is the
  user's; the implementer removes them only on the user's yes in chat.
