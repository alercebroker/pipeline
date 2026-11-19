# lc_classification_multisurvey_step: unique ranking on tied probabilities

## Goal

`probability.ranking` for one (oid, classifier) is a permutation of 1..N over
the N classes of that head. Two classes with the same probability no longer
share a rank: the class that comes first in the model's output columns gets the
lower rank.

## Non-goals

- Changing probabilities, class ids, lastmjd, or which heads are emitted.
- The offline writer `feature_step/features/offline/probability_writer.py` and
  the stamp steps, which rank with `method="dense"` too. Separate change if
  wanted.
- Rewriting rows already stored in `multisurvey.probability`.

## Acceptance criteria

1. For every (oid, classifier) group the rankings are exactly `{1, ..., N}`
   with N the number of classes of the head: no repeated rank, no gap.
2. Rankings are descending in probability: a strictly larger probability always
   gets a strictly smaller rank.
3. On a tie the class that appears first in the frame's columns gets the lower
   rank. Test: `{"SNIa": 0.5, "AGN": 0.5, "LPV": 0.0}` -> SNIa 1, AGN 2, LPV 3;
   `{"AGN": 0.5, "SNIa": 0.5, "LPV": 0.0}` -> AGN 1, SNIa 2, LPV 3.
4. The existing test `test_ranking_is_dense_descending_within_oid_and_head`
   is replaced by one that asserts criterion 3 (edit approved by Francisco).
5. Every new or changed line carries a `#` comment.

## Done looks like

```
~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check lc_classification_multisurvey_step
```
exits 0.
