# Correction step — scribe flush/commit ordering fix

**Branch:** `fix/correction-step-scribe-flush` (based on `multisurvey`)
**Planning task:** `ztf-legacy-pipeline / correction-step-flush-fix` (owner ignacio; serious but not urgent — legacy quimal pipeline in its tail)
**Same defect class as:** the detection-loss / scribe `parent_candid="nan"` fix (PR #623) — silent data loss in the on-prem legacy steps.

## Root cause (confirmed in code)

`CorrectionStep.produce_scribe` (`correction/_step/step.py`) only forces a flush on the
*last element of the detections list*:

```python
if count == len(detections):
    flush = True
self.scribe_producer.produce(scribe_payload, flush=flush, key=oid)
```

But the loop does `if not detection.pop("new"): continue` **before** reaching that
`produce()` call. So when the **last detection in the batch is `new=False`**, the
`flush=True` produce never happens. Every scribe message buffered earlier in the batch
stays in librdkafka's internal queue, undelivered.

The framework then commits the Kafka offset in `GenericStep._post_produce`
→ `consumer.commit()` (`libs/apf/apf/core/step.py`), which runs **after** `post_execute`.
If the pod is deleted (k8s scale-down / rollout / OOM) in that window — offset committed,
scribe messages not yet delivered — those detections are **silently lost**: never written
to the DB, never retried.

Per-batch flush frequency is unchanged by the fix (the code already intends one flush per
batch), so making the flush unconditional is **not** a throughput regression.

## Step 1 — Reproduction test (deterministic; no real pod-kill)

The real failure is a race (step buffering + k8s killing the pod at the same instant), which
is impractical to reproduce live. Instead we reproduce the **loss window** deterministically
at the unit level: one batch whose last detection is `new=False`, then assert that scribe
messages remain un-delivered at the moment the consumer offset is committed.

New unit test in `tests/unittests/test_step.py`. Model the scribe producer as a
librdkafka-like buffer (`produce()` buffers, `flush()` drains to "delivered") and a consumer
whose `commit()` snapshots what is still buffered:

```python
class FakeScribeProducer:
    def __init__(self):
        self.buffered, self.delivered = [], []
        self.producer = self  # the fixed path calls self.scribe_producer.producer.flush()
    def produce(self, message, flush=False, key=None):
        self.buffered.append(message)
        if flush:
            self.flush()
    def flush(self, *a, **k):
        self.delivered.extend(self.buffered)
        self.buffered = []

class FakeConsumer:
    def __init__(self, scribe):
        self.scribe = scribe
        self.commit_called = False
        self.buffered_at_commit = None
    def commit(self):
        self.commit_called = True
        self.buffered_at_commit = list(self.scribe.buffered)  # un-delivered at commit time
```

The batch is crafted so the **last detection is `new=False`**. No Corrector/pandas needed —
`produce_scribe` only reads `new / candid / oid / forced / has_stamp / extra_fields`, so we
call `post_execute` directly with a hand-built result and then the real framework commit path:

```python
result = {"detections": [
    {"new": True,  "candid": "a", "oid": "OID1", "forced": False, "has_stamp": True, "extra_fields": {}},
    {"new": False, "candid": "b", "oid": "OID1", "forced": False, "has_stamp": True, "extra_fields": {}},
]}
step.post_execute(result)   # real produce_scribe -> buffers msg "a", skips "b", never flushes
step._post_produce()        # real framework path -> consumer.commit()

assert step.consumer.commit_called
assert step.consumer.buffered_at_commit == []   # the invariant
```

Run against the **unfixed** code → the invariant assertion **fails**:
`buffered_at_commit == [msg_a]` while `commit_called is True`. That failure *is* the
reproduction — "committed to Kafka, message never sent to the scribe." Capture that output as
the acceptance-criterion evidence.

## Step 2 — The fix

Drop the fragile per-message flush; flush **unconditionally after the loop**, guaranteeing
delivery before the framework commits offsets:

```python
def produce_scribe(self, detections: list[dict]):
    for detection in detections:
        detection = deepcopy(detection)
        if not detection.pop("new"):
            continue
        # ... unchanged payload construction ...
        self.scribe_producer.produce(scribe_payload, key=oid)   # no per-message flush
    # Guarantee delivery before GenericStep._post_produce commits the offsets
    self.scribe_producer.producer.flush()
```

`.producer.flush()` mirrors the pattern the multisurvey correction step already uses, works on
the real apf `KafkaProducer`, and drains the whole buffer.

**Rejected alternative:** adding a public `flush()` to apf's producer. It widens blast radius
to `libs/apf` and forces an apf version bump; the detection-loss work showed apf drift is
painful. Keeping the fix local to `correction_step`.

## Step 3 — Regression proof

The Step 1 test now passes (`buffered_at_commit == []`). That single test doubles as both
acceptance criteria — "reproduce the loss window" (red before the fix) and "regression test
proving no loss across a pod-delete mid-batch" (green after). A one-line docstring ties it to
this bug.

## Step 4 — Package & deploy (deploy is human-executed)

- Bumped `correction_step/pyproject.toml` version `27.5.7a37` → **`27.5.7a41`**. (Deviates from
  the originally-planned `a38`: a survey of the monorepo showed `a38` was already consumed by
  the known-bad scribe build, and the repo high-water mark had moved to `a40` (metadata). `a41`
  keeps the correction image unambiguously newer than everything else and avoids reusing the
  problematic `a38` tag.) Let the `.github/workflows/correction_step.yaml` CI build the image.
- **Verify at deploy time** whether quimal's legacy correction runs the `correction_step`
  chart or `correction_multistream_ztf_step` — both exist under `charts/`.
- Deploying to quimal is the human-executed step (infra safety — planning sessions don't touch
  infra).

## Out of scope (flagged, per the task)

`correction_multisurvey_step` shares the defect **class** — worse, it never flushes the scribe
before commit (produces + `poll(0)` only). But it is a separate protocol and code path
(per-object structured bundles, not per-detection Mongo-style commands), so it is left for a
follow-up task rather than folded in here.

## Acceptance criteria (mirror of the planning task)

- [x] Reproduce the loss window (committed to Kafka but not flushed to scribe) — Step 1, red.
  Confirmed RED on unfixed code: `commit_called is True` while the scribe payload for candid
  `"a"` was still buffered (`buffered_at_commit == [msg_a]`).
- [x] Fix the ordering: flush the scribe before the Kafka offset commit — Step 2
  (`self.scribe_producer.producer.flush()` after the loop; per-message `flush` removed).
- [x] Regression test proving no loss across a pod-delete mid-batch — Step 3, green.
  `test_scribe_is_flushed_before_offset_commit_when_last_detection_is_not_new` passes; full
  unit suite 52/52 green.
- [ ] Deployed on the quimal legacy correction step — Step 4 (human-executed). Version bumped
  to `27.5.7a41`; CI build + quimal deploy pending.
