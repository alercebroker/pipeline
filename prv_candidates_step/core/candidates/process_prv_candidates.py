import pandas as pd
from typing import Tuple
from prv_candidates_step.core.processor.processor import Processor
from prv_candidates_step.core.strategy.atlas_strategy import ATLASPrvCandidatesStrategy
from prv_candidates_step.core.strategy.ztf_strategy import ZTFPrvCandidatesStrategy


def process_prv_candidates(
    processor: Processor, alerts: Tuple[pd.DataFrame, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate previous candidates from alerts.

    The input must be a DataFrame created from a list of GenericAlert.

    Parameters
    ----------
    alerts: A pandas DataFrame created from a list of GenericAlerts.

    Returns A Tuple with detections a non_detections from previous candidates
    -------

    """
    # dicto = {
    #     "ZTF": ZTFPrvCandidatesStrategy()
    # }
    data = alerts[["aid", "oid", "tid", "candid", "ra", "dec", "pid", "extra_fields"]]
    detections = []
    non_detections = []
    for tid, subset_data in data.groupby("tid"):
        if tid == "ZTF":
            processor.strategy = ZTFPrvCandidatesStrategy()
        elif "ATLAS" in tid:
            processor.strategy = ATLASPrvCandidatesStrategy()
        else:
            raise ValueError(f"Unknown Survey {tid}")
        det, non_det = processor.compute(subset_data)
        detections.append(det)
        non_detections.append(non_det)
    detections = pd.concat(detections, ignore_index=True)
    non_detections = pd.concat(non_detections, ignore_index=True)
    return detections, non_detections
