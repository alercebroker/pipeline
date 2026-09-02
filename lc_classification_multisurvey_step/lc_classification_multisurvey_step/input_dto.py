"""feature_step messages -> features-only InputDTO, plus the lastmjd map.

`SquidwardFeaturesClassifier` reads only `input_dto.features`, so detections /
non-detections / xmatch / stamps are passed empty (design doc §4).

`alerce_classifiers` is imported lazily inside `create_input_dto` so the rest of
this module — and the unit suite — needs no model dependency.
"""
import logging
import math

import pandas as pd

log = logging.getLogger(__name__)


def filter_messages(messages: list, min_detections=None) -> list:
    """Drop messages the classifier cannot or should not consume (design §8).

    - unparseable oid: `oid` is a plain Avro string, so its digits rest on a
      producer convention. Dropped here so `int(oid)` downstream cannot take the
      whole batch down on every redelivery. Aggregated into one warning.
    - no features -> cannot classify;
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

    Two messages for one oid in a batch are an update, not a duplicate.
    feature_step produces keyed by `str(oid)`, so they land on one partition in
    offset order and the last by arrival is the newest — that is why it wins.

    `build_features_frame` and `lastmjd_by_oid` both derive from this so they
    cannot disagree about which message won for a given oid.
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

    Collapsing duplicates is what upholds `build_probability_rows`' unique-oid
    contract: two rows for one oid would collide on the probability primary key,
    and the scribe's highest-lastmjd dedup cannot break a tie of equal lastmjd.
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

    Already MJD — do NOT subtract 2400000.5. `detections` carries forced
    photometry too, so this is the max over both, matching offline
    `classify._lc_lastmjd`.

    Non-finite mjds are filtered before `max`, not left for it to resolve: `max`
    is order-sensitive with NaN (`max(nan, x)` is `nan`, `max(x, nan)` is `x`).
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
