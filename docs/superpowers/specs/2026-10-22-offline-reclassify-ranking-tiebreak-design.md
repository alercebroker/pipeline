# Offline probability writer: unique ranking on tied probabilities

Date: 2026-10-22. Package: `feature_step` (`features/offline/probability_writer.py`).
Mirror of the live-step fix `7dbb636f3` (branch `fix/lc-ms-ranking-tiebreak`), so the
reclassification run that repairs the rows written with dense ranks produces the
same rankings the step now produces.

## Goal

`build_probability_rows` assigns, per (oid, classifier head), rankings that are a
permutation of `1..N` (N = classes of the head). Tied probabilities get distinct
rankings; the class that comes first in the model's column order gets the lower one.

## Non-goals

- No change to probability values, `lastmjd`, `classifier_version` or the upsert.
- No change to the live step (already fixed and deployed) or to the scribe.
- No SQL-only repair of the stored rows; the repair is a reclassification run with
  `scripts/offline_reclassify_batch.py` over an oid list the operator provides.
- No change to `offline_reclassify_batch.py` or the oid-list loader.

## Acceptance criteria

1. For a head whose frame has two or more classes with the same probability, the
   `ranking` values of its rows are exactly `{1, ..., N}` with no repeat.
2. On a tie, the class that appears first in the OutputDTO frame's columns gets
   the lower ranking (same tie-break as the step: `rank(method="first")` over the
   frame's column order).
3. For a head without ties, every row keeps the ranking the dense method gave it
   (no behaviour change outside ties).
4. The docstring of `build_probability_rows` names the tie-break rule; the old
   sentence "per-classifier dense rank descending" is gone.

## Done looks like

```
~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```
exits 0, with a new test file `feature_step/tests/unittest/test_probability_writer_tiebreak.py`
holding one test per criterion, seen red before the one-line change.
