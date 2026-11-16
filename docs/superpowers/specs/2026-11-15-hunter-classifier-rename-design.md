# Hunter classifier rename: `stamp_sn_offnuclear_rubin_beta`

## Goal

Rename classifier 10 and its two classes to the names agreed on Slack (option 2 of the
#machine-learning-private thread, accepted 2026-11-11), without changing any stored
`class_id`, probability or ranking.

| | today | after |
|---|---|---|
| classifier 10 name | `nuclear_classifier` | `stamp_sn_offnuclear_rubin_beta` |
| class_id 0 (model softmax index 1, the positive class) | `off-nuclear` | `SNoffnuclear` |
| class_id 1 (model softmax index 0) | `nuclear` | `notSNoffnuclear` |

The step maps model class name -> `class_id` through the `taxonomy` table
(`rubin_stamp_classifier_step/step.py:362`, `.get(class_name, -1)`), so the names in
`HunterClassifierModel.CLASSES` and in `taxonomy` must change together.

## Non-goals

- No change to `class_id` values, to the softmax order, to the model artifact or to its version.
- No rewrite of existing `probability` rows (they store ids, not names).
- No change to the rubin stamp classifier (classifier and taxonomy rows of the 5-class model).
- No change to step code in `rubin_stamp_classifier_step`: it reads names from the model and the table.
- The DB updates, the image build and the redeploy are operate steps listed under "Rollout", not part of the merged diff.

## Acceptance criteria

1. `HunterClassifierModel.CLASSES == ("notSNoffnuclear", "SNoffnuclear")`: index 0 keeps the
   class that was `nuclear`, index 1 keeps the class that was `off-nuclear`.
2. `HunterClassifierModel.dict_mapping_classes == {0: "notSNoffnuclear", 1: "SNoffnuclear"}`.
3. `predict` returns a probabilities frame whose columns are `["notSNoffnuclear", "SNoffnuclear"]`,
   in that order, and whose values for a fixed input are equal to the values before the rename
   (same column position, new label).
4. The strings `"nuclear"` and `"off-nuclear"` no longer appear as class names anywhere under
   `alerce_classifiers/alerce_classifiers/hunter/` or `alerce_classifiers/tests/unit/test_hunter_*.py`.
5. With a taxonomy `{SNoffnuclear: 0, notSNoffnuclear: 1}`, the step's name -> id mapping gives
   `class_id` 0 for softmax index 1 and `class_id` 1 for softmax index 0, and never -1.

## Done looks like

```
cd alerce_classifiers && <py> -m pytest tests/unit/test_hunter_model.py tests/unit/test_hunter_mapper.py tests/unit/test_hunter_skip.py tests/unit/test_hunter_rename.py -q
```

green, with `<py>` = `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python`.

## Rollout (operate, after the PR is approved)

Order matters: a pod that restarts with the old image after the table changed writes `class_id = -1`.

1. Build `hunter_stamp_classifier_step:<new tag>` from `deploy/stamp-hunter-lsst-v11-2` with the submodule bumped.
2. Scale the hunter release to 0.
3. `UPDATE` the `classifier` row 10 and the two `taxonomy` rows (by hand, superuser, as on 2026-10-30).
4. `helm upgrade` the hunter release to the new tag; the start log must show
   `Class taxonomy: {'SNoffnuclear': 0, 'notSNoffnuclear': 1}`.

## Open decisions

- **Branch base in `alerce_classifiers`.** Proposed: `feat/hunter-rename-classes` off
  `feat/hunter-crop-then-stack` (draft PR #94, the commit the deployed image uses), draft PR against that branch.
- **Committed tests assert the old names** (8 lines in `test_hunter_model.py`, `test_hunter_mapper.py`,
  `test_hunter_skip.py`). They are read-only for the agent; the rename cannot be green without editing them.
- **Pipeline side.** `deploy/stamp-hunter-lsst-v11-2` is local-only. Proposed: bump the submodule there for
  the build, and on `feat/stamp-hunter-classifier` (PR #633) so the PR carries the same commit.
- **Consumers that read names.** Not checked: whether hunter-experiment or the TAP tutorial filter by
  `class_name` instead of `class_id`.
