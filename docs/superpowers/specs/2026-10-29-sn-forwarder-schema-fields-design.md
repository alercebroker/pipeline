# SN forwarder emits only the fields its schema declares

Date: 2026-10-29. Branch `feat/stamp-hunter-classifier`, PR #633.

## Goal

The SN forwarder in `rubin_stamp_classifier_step` must serialize every forwarded
alert with the LSST v11.1 schema without failing on keys the consumer added to
the dict after deserialization.

Observed on the first local run against production (2026-10-29): apf's
`KafkaConsumer` adds `timestamp` (the Kafka message timestamp,
`libs/apf/apf/consumers/kafka.py:213`) to each deserialized alert; the step
keeps that same dict by reference as `alert`; `KafkaSchemalessProducer` is
strict and raised
`ValueError: record contains more fields than the schema specifies: timestamp`
on the first ranking-1 SN. The step exited before committing the offset.

## Non-goals

- Changing apf's consumer (the `timestamp` key is used elsewhere).
- Changing the forwarder's schema or the output topic schema.
- Forwarding anything other than the raw alert (no probabilities, no stamps
  beyond what the alert carries).

## Acceptance criteria

1. `StampClassifierStep.forward_sn_candidates` produces, for each ranking-1
   `SN_FORWARD_CLASS` alert, a dict that contains only the top-level keys
   declared in the forwarder producer's schema
   (`self.sn_forward_producer.schema["fields"]`), in the same way
   `pre_produce` filters with `output_fields`. The original `alert` dict is
   not mutated.
2. When the forwarder producer has no `schema` attribute (a producer without
   `SCHEMA_PATH`, as in the unit tests' recording producer), the alert is
   produced unchanged.
3. A new test file `tests/unit/test_sn_forwarder_schema.py` builds a step
   whose forwarder producer loads the real
   `schemas/surveys/lsst_v11.1/lsst.v11_1.alert.avsc`, feeds a processed
   alert whose `alert` dict carries an extra `timestamp` key, and asserts
   that `fastavro.schemaless_writer(..., strict=True)` with that schema
   accepts what was produced. The same test fails before the change with the
   `ValueError` above (seen red).
4. The existing tests in `tests/unit/test_sn_forwarding.py` stay green and
   are not edited.

## Done looks like

From the worktree, with the libs pinned to it:

```
PYTHONPATH=<worktree>/rubin_stamp_classifier_step:<worktree>/alerce_classifiers:<worktree>/libs/apf:<worktree>/libs/db-plugins-multisurvey \
  ~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check rubin_stamp_classifier_step
```

exits 0, and the local run that failed (`local_config_forwarder.yaml`, group
`lsst-ingestion-rubin-stamp-forwarder-local-20261029`) logs
`Forwarded 1 of 10 alerts as SN candidates` on the batch that crashed.
