# pyright: reportPrivateUsage=false
from typing import Any

import ingestion_step.lsst.extractor as extractor


def test_extract(lsst_alerts: list[dict[str, Any]]):
    lsst_data = extractor.extract(lsst_alerts)

    n_sources = len(lsst_alerts)
    n_prv_sources = sum(
        len(alert["prvDiaSources"])
        for alert in lsst_alerts
        if alert["prvDiaSources"]
    )
    n_forced_sources = sum(
        len(alert["prvDiaForcedSources"])
        for alert in lsst_alerts
        if alert["prvDiaForcedSources"]
    )
    n_non_detections = sum(
        len(alert["prvDiaNondetectionLimits"])
        for alert in lsst_alerts
        if alert["prvDiaNondetectionLimits"]
    )
    n_dia = len(list(filter(lambda alert: alert["diaObject"], lsst_alerts)))
    n_ss = len(list(filter(lambda alert: alert["ssObject"], lsst_alerts)))

    assert n_sources == len(lsst_data["sources"])
    assert n_prv_sources == len(lsst_data["previous_sources"])
    assert n_forced_sources == len(lsst_data["forced_sources"])
    assert n_non_detections == len(lsst_data["non_detections"])
    assert n_dia == len(lsst_data["dia_object"])
    assert n_ss == len(lsst_data["ss_object"])

    # for key in lsst_data:
    #     print(f"\n\n--- {key}: ---")
    #     print(lsst_data[key])
