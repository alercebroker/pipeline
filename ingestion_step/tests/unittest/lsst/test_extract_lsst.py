from ingestion_step.core.types import Message
from ingestion_step.lsst.extractor import (
    LsstDiaObjectExtractor,
    LsstForcedSourceExtractor,
    LsstNonDetectionsExtractor,
    LsstPrvSourceExtractor,
    LsstSourceExtractor,
    LsstSsObjectExtractor,
)


def test_extract_dia_object(lsst_alerts: list[Message]):
    objects = LsstDiaObjectExtractor.extract(lsst_alerts)
    assert len(objects) == sum(alert["diaObject"] is not None for alert in lsst_alerts)

    fields = {"message_id", "midpointMjdTai"}
    assert fields <= set(objects.keys())


def test_extract_ss_object(lsst_alerts: list[Message]):
    objects = LsstSsObjectExtractor.extract(lsst_alerts)
    assert len(objects) == sum(alert["ssObject"] is not None for alert in lsst_alerts)

    fields = {"message_id", "ra", "dec", "midpointMjdTai"}
    assert fields <= set(objects.keys())


def test_extract_source(lsst_alerts: list[Message]):
    sources = LsstSourceExtractor.extract(lsst_alerts)
    assert len(sources) == len(lsst_alerts)

    fields = {"message_id", "has_stamp"}
    assert fields <= set(sources.keys())


def test_extract_prv_source(lsst_alerts: list[Message]):
    prv_sources = LsstPrvSourceExtractor.extract(lsst_alerts)
    assert len(prv_sources) == sum(
        len(alert["prvDiaSources"]) for alert in lsst_alerts if alert["prvDiaSources"]
    )

    fields = {"message_id", "has_stamp"}
    assert fields <= set(prv_sources.keys())


def test_extract_non_detection(lsst_alerts: list[Message]):
    non_detections = LsstNonDetectionsExtractor.extract(lsst_alerts)
    assert len(non_detections) == sum(
        len(alert["prvDiaNondetectionLimits"])
        for alert in lsst_alerts
        if alert["prvDiaNondetectionLimits"]
    )

    fields = {"message_id", "diaObjectId", "ssObjectId"}
    assert fields <= set(non_detections.keys())


def test_extract_forced(lsst_alerts: list[Message]):
    forced_sources = LsstForcedSourceExtractor.extract(lsst_alerts)
    assert len(forced_sources) == sum(
        len(alert["prvDiaForcedSources"])
        for alert in lsst_alerts
        if alert["prvDiaForcedSources"]
    )

    fields = {"message_id"}
    assert fields <= set(forced_sources.keys())
