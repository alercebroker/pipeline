# Plan: hunter classifier rename

Spec: `docs/superpowers/specs/2026-11-15-hunter-classifier-rename-design.md`.

## Task 1 (alerce_classifiers, branch `feat/hunter-rename-classes` off `feat/hunter-crop-then-stack`)

Closes criteria 1, 2, 3, 4, 5.

1. New test file `tests/unit/test_hunter_rename.py`, one test per criterion, committed red:
   - (1) `HunterClassifierModel.CLASSES == ("notSNoffnuclear", "SNoffnuclear")`.
   - (2) `dict_mapping_classes == {0: "notSNoffnuclear", 1: "SNoffnuclear"}`.
   - (3) `predict` on a fixed input with a fixed-weight network: columns are the new names in order, and
     column i equals softmax index i (values unchanged by the rename).
   - (4) no `"nuclear"` / `"off-nuclear"` class-name string left under `alerce_classifiers/hunter/` and
     `tests/unit/test_hunter_*.py` (the rename test builds the old strings so it does not match itself).
   - (5) with taxonomy `{SNoffnuclear: 0, notSNoffnuclear: 1}`, mapping `dict_mapping_classes` names through
     the taxonomy gives id 0 for index 1, id 1 for index 0, never -1.
2. `alerce_classifiers/hunter/model.py`: `CLASSES` and its comment.
3. The three committed hunter test files: replace the old names on the lines that assert them
   (user-authorized 2026-11-15), nothing else.
4. Done: the spec's pytest command is green.

## Task 2 (pipeline, `feat/stamp-hunter-classifier`)

Bump the `alerce_classifiers` gitlink to the head of Task 1. No step code changes.
