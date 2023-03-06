import pickle
from prv_candidates_step.core.strategy.base_strategy import BasePrvCandidatesStrategy
import logging


class ZTFPrvCandidatesStrategy(BasePrvCandidatesStrategy):
    def process_prv_candidates(self, alert: dict):
        detections = []
        non_detections = []
        prv_candidates = alert["extra_fields"].get("prv_candidates")
        if prv_candidates:
            prv_candidates = pickle.loads(prv_candidates)
            try:
                for prv_cand in prv_candidates:
                    candid = prv_cand["candid"]
                    if not candid:
                        non_detections.append(prv_cand)
                    else:
                        detections.append(prv_cand)
            except TypeError:
                logging.info("No prv_candidates")
                return [], []

        return detections, non_detections
