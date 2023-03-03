from typing import Tuple, List
from prv_candidates_step.core.processor.processor import Processor
from prv_candidates_step.core.strategy.atlas_strategy import ATLASPrvCandidatesStrategy
from prv_candidates_step.core.strategy.ztf_strategy import ZTFPrvCandidatesStrategy


def process_prv_candidates(
    processor: Processor, alerts: List[dict]
) -> Tuple[List[List[dict]], List[List[dict]]]:
    """Separate previous candidates from alerts.

    The input must be a DataFrame created from a list of GenericAlert.

    Parameters
    ----------
    nalerts: list
        a list of alerts as they come from kafka. Same as messages argument for step.execute
    processor: Processor
        context for the strategy pattern, where strategies are for atlas and ztf prv_candidates processes
    """
    prv_detections = []
    non_detections = []
    for alert in alerts:
        if alert["tid"].lower() == "ztf":
            processor.strategy = ZTFPrvCandidatesStrategy()
        else:
            processor.strategy = ATLASPrvCandidatesStrategy()
        alert_prv_detections, alert_non_detections = processor.compute(alert)
        prv_detections.append(alert_prv_detections)
        non_detections.append(alert_non_detections)
    return prv_detections, non_detections
