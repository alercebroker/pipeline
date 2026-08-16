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

The offline-equivalence test is opt-in and needs the `alerce_classifiers`
submodule plus `MODEL_PATH`:

    RUN_EQUIVALENCE_TEST=1 MODEL_PATH=<s3 url> python -m pytest tests/integration -v
