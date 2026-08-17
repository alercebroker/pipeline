# LC Classification Multisurvey Step

Consumes the multisurvey `feature_step` output topic, runs the ZTF BHRF
(Squidward 2.1.0) classifier, and produces probabilities for its five heads to
`scribe_multisurvey`, which owns the upsert into `multisurvey_ztf.probability`.

The step writes nothing to the database. It reads `classifier` and `taxonomy`
once at startup to resolve classifier names to ids and class names to class ids,
and refuses to start if either is unseeded (see the design doc, §8).

Design: `docs/superpowers/specs/2026-08-16-multisurvey-lc-classification-step-design.md`

## Tests

The unit suite has no model dependency:

    python -m pytest tests/unittest -v

The offline-equivalence test is opt-in. It needs the offline checkout at
`~/desktop/pipeline/feature_step` — note the `feature_step` subdirectory, since
`features.offline` does not resolve from the checkout root. It needs neither the
`alerce_classifiers` submodule nor `MODEL_PATH`: both row builders are pure and
no classifier is run.

    RUN_EQUIVALENCE_TEST=1 python -m pytest tests/integration -v

If it reports `SKIPPED`, the offline checkout was not found — the skip message
names the path it looked for. A skip here means the test did not run at all, so
treat it as no coverage rather than as a pass.

This is the only test that checks the port against the implementation it was
ported from; the unit suite checks the port's internal consistency. It compares
one single-oid `OutputDTO` through both row builders. It cannot cover the
multi-oid melt path, because the offline reference raises on a multi-row frame
by design.
