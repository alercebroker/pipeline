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

import pandas as pd

log = logging.getLogger(__name__)


def filter_messages(messages: list, min_detections=None) -> list:
    """Drop messages the classifier cannot or should not consume.

    - no features (`features` is None or empty) -> cannot classify (design §8);
    - fewer than `min_detections` *non-forced* detections -> optional pre-filter,
      counted the way the legacy step counts it (design §13). Unset by default.
    """
    kept = []
    for message in messages:
        if not message.get("features"):
            continue
        if min_detections is not None:
            n_detections = sum(
                1 for d in (message.get("detections") or []) if not d.get("forced", False)
            )
            if n_detections < min_detections:
                continue
        kept.append(message)
    return kept


def build_features_frame(messages: list) -> pd.DataFrame:
    """One row per message, indexed by the bigint oid, columns = feature names.

    The multisurvey feature_step already emits the bigint masterid in `oid` (the
    Avro field is typed string), so this casts with `int()` and calls no idmapper
    — unlike the stamp step, which starts from raw ZTF alerts (design doc §4).

    Duplicate oids within one batch are collapsed, keeping the LAST message for
    that oid. Two messages for the same object can arrive in a single consume
    batch; left alone they would yield two probability rows colliding on
    `(oid, sid, classifier_id, class_id)`, which the scribe's highest-lastmjd
    dedup cannot break because both carry the same lastmjd. This is what upholds
    `build_probability_rows`' unique-oid-index contract.
    """
    if not messages:
        frame = pd.DataFrame()
        frame.index.name = "oid"
        return frame

    frame = pd.DataFrame(
        [message["features"] for message in messages],
        index=[int(message["oid"]) for message in messages],
    )
    frame.index.name = "oid"
    return frame[~frame.index.duplicated(keep="last")]


def lastmjd_by_oid(messages: list) -> dict:
    """{oid: max detection mjd}. Already MJD — do NOT subtract 2400000.5.

    The `detections` array carries forced photometry too (each entry has a
    `forced` flag), so this is the max over detections and forced together,
    matching offline `classify._lc_lastmjd`.
    """
    lastmjd = {}
    for message in messages:
        mjds = [
            float(d["mjd"])
            for d in (message.get("detections") or [])
            if d.get("mjd") is not None
        ]
        if not mjds:
            log.warning("oid=%s has no detection mjd; it will produce no rows", message["oid"])
            continue
        lastmjd[int(message["oid"])] = max(mjds)
    return lastmjd


def create_input_dto(messages: list):
    """Features-only InputDTO for the batch."""
    from alerce_classifiers.base.factories import input_dto_factory

    empty = pd.DataFrame()
    return input_dto_factory(empty, empty, build_features_frame(messages), empty, empty)
