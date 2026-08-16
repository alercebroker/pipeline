"""feature_step messages -> features-only InputDTO, plus the lastmjd map.

`SquidwardFeaturesClassifier.can_predict` inspects only `input_dto.features`, and
`predict` calls `mapper.preprocess(input_dto)` which reads only features. So
detections / non-detections / xmatch / stamps are passed empty (design doc §4),
which also drops the legacy step's stale candid schema and its pickled
extra_fields round-trip.

`alerce_classifiers` is imported lazily inside `create_input_dto` so the rest of
this module — and the unit suite — needs no model dependency.
"""
import logging
import math

import pandas as pd

log = logging.getLogger(__name__)


def filter_messages(messages: list, min_detections=None) -> list:
    """Drop messages the classifier cannot or should not consume.

    - unparseable oid -> dropped before anything else touches int(oid). `oid`
      is a plain Avro "string": the schema cannot constrain it to digits, so
      its validity rests on a producer convention, not the wire format. A
      message whose oid does not convert would otherwise crash int(oid) inside
      _collapse_by_oid and take the whole batch down, repeating on every Kafka
      redelivery (design §8: log and drop rather than kill the batch). Bad
      oids are aggregated into one warning per batch naming the raw values.
    - no features (`features` is None or empty) -> cannot classify (design §8);
    - fewer than `min_detections` *non-forced* detections -> optional pre-filter,
      counted the way the legacy step counts it (design §13). Unset by default.
    """
    kept = []
    bad_oids = []
    for message in messages:
        try:
            int(message["oid"])
        except ValueError:
            bad_oids.append(message["oid"])
            continue
        if not message.get("features"):
            continue
        if min_detections is not None:
            n_detections = sum(
                1 for d in (message.get("detections") or []) if not d.get("forced", False)
            )
            if n_detections < min_detections:
                continue
        kept.append(message)
    if bad_oids:
        log.warning(
            "%d message(s) had an oid that does not parse as int; dropped: %s",
            len(bad_oids),
            bad_oids,
        )
    return kept


def _collapse_by_oid(messages: list) -> dict:
    """{oid: winning message}, one pass, oid cast to int.

    Two messages for the same oid landing in a single consume batch are
    normally an update carrying a different detection set, not identical
    duplicates. `feature_step` produces keyed by `str(oid)`, so same-oid
    messages land on one Kafka partition in offset order, and the LAST message
    by arrival order really is the newest one — that is why it wins.

    `build_features_frame` and `lastmjd_by_oid` both derive from this single
    collapsed mapping so they cannot disagree about which message won for a
    given oid: computing the winner twice, independently, let a duplicate
    oid's features come from one message while its lastmjd came from another.
    """
    collapsed = {}
    for message in messages:
        collapsed[int(message["oid"])] = message
    return collapsed


def build_features_frame(messages: list) -> pd.DataFrame:
    """One row per distinct oid, indexed by the bigint oid, columns = feature names.

    The multisurvey feature_step already emits the bigint masterid in `oid` (the
    Avro field is typed string), so this casts with `int()` and calls no idmapper
    — unlike the stamp step, which starts from raw ZTF alerts (design doc §4).

    Duplicate oids collapse via `_collapse_by_oid`, keeping the winning message
    for that oid. Left alone, duplicates would yield two probability rows
    colliding on `(oid, sid, classifier_id, class_id)`, which the scribe's
    highest-lastmjd dedup cannot break because both carry the same lastmjd.
    This is what upholds `build_probability_rows`' unique-oid-index contract.
    """
    if not messages:
        frame = pd.DataFrame()
        frame.index.name = "oid"
        return frame

    collapsed = _collapse_by_oid(messages)
    frame = pd.DataFrame(
        [message["features"] for message in collapsed.values()],
        index=list(collapsed.keys()),
    )
    frame.index.name = "oid"
    return frame


def lastmjd_by_oid(messages: list) -> dict:
    """{oid: max detection mjd} for the same winning message as build_features_frame.

    Already MJD — do NOT subtract 2400000.5. The `detections` array carries
    forced photometry too (each entry has a `forced` flag), so this is the max
    over detections and forced together, matching offline `classify._lc_lastmjd`.

    A missing or non-finite (NaN/inf) mjd is skipped rather than compared: `max`
    is order-sensitive with NaN (`max(nan, x)` is `nan`, `max(x, nan)` is `x`),
    so a NaN must be filtered out before `max`, not left for `max` to resolve.
    """
    collapsed = _collapse_by_oid(messages)
    lastmjd = {}
    missing_oids = []
    for oid, message in collapsed.items():
        mjds = []
        for d in message.get("detections") or []:
            mjd = d.get("mjd")
            if mjd is None:
                continue
            mjd = float(mjd)
            if not math.isfinite(mjd):
                continue
            mjds.append(mjd)
        if not mjds:
            missing_oids.append(oid)
            continue
        lastmjd[oid] = max(mjds)
    if missing_oids:
        log.warning(
            "%d oid(s) have no usable detection mjd; they will produce no rows: %s",
            len(missing_oids),
            missing_oids,
        )
    return lastmjd


def create_input_dto(messages: list):
    """Features-only InputDTO for the batch."""
    from alerce_classifiers.base.factories import input_dto_factory

    empty = pd.DataFrame()
    return input_dto_factory(empty, empty, build_features_frame(messages), empty, empty)
