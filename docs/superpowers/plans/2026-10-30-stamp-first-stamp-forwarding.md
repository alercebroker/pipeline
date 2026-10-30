# Plan: first-stamp SN forwarding and hunter class names

Spec: [2026-10-30-stamp-first-stamp-forwarding-design.md](../specs/2026-10-30-stamp-first-stamp-forwarding-design.md).
The code exists uncommitted in the `pipeline-hunter` worktree and the
`alerce_classifiers` checkout (written and run 2026-10-29/30); each task
turns a part of it into one reviewed commit.

## Task 1: hunter classes (criterion 6)

Repo `alerce_classifiers`, branch `feat/hunter-crop-then-stack` (PR #94).
Commit `alerce_classifiers/hunter/model.py` and the three hunter unit tests.
Green: `pytest tests/unit -q` = 17 passed, 1 skipped. Push.

## Task 2: first-stamp upsert and forwarder (criteria 1-5, 7)

Repo `pipeline`, branch `feat/stamp-hunter-classifier` (PR #633).
Remove the `STAMP_SKIP_DB_WRITE` bypass and `import os` from `step.py`
(criterion 5), then commit `db/db.py`, `step.py`, the four edited unit tests
and the submodule pointer to task 1's commit. Green: the "done looks like"
command of the spec against the recorded baseline. Push.

## Task 3: review

Adversarial reviewer on `4b15a16cb..HEAD` against the spec, findings posted
on PR #633, fixes as follow-up commits, final checklist and PR body.
