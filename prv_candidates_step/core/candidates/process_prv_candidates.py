from typing import Tuple, List
from prv_candidates_step.core.strategy.atlas_strategy import ATLASPrvCandidatesStrategy
from prv_candidates_step.core.strategy.ztf_strategy import ZTFPrvCandidatesStrategy


def process_prv_candidates(
    alerts: List[dict],
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Separate previous candidates from alerts.

    The input must be a DataFrame created from a list of GenericAlert.

    Parameters
    ----------
    nalerts: list
        a list of alerts as they come from kafka. Same as messages argument for step.execute
    """
    prv_detections = []
    non_detections = []
    for alert in alerts:
        survey = alert["sid"]
        if survey.lower() == "ztf":
            strategy = ZTFPrvCandidatesStrategy()
        elif survey.lower() == "atlas":
            strategy = ATLASPrvCandidatesStrategy()
        else:
            raise ValueError(f"Not recognized survey: {survey}")
        alert_prv_detections, alert_non_detections = strategy.process_prv_candidates(
            alert
        )
        prv_detections.append(alert_prv_detections)
        non_detections.append(alert_non_detections)
    return prv_detections, non_detections
