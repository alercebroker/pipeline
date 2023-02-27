from apf.core.step import GenericStep

from .utils.prv_candidates.processor import Processor
from prv_detection_step.utils.prv_candidates.strategies import (
    ATLASPrvCandidatesStrategy,
    ZTFPrvCandidatesStrategy,
)

from typing import Tuple

import numpy as np
import pandas as pd
import logging


class PrvDetectionStep(GenericStep):
    """PrvDetectionStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
            self,
            consumer=None,
            config=None,
            level=logging.INFO,
            producer=None,
            **step_args,
    ):
        super().__init__(consumer, config=config, level=level)
        self.version = config["STEP_METADATA"]["STEP_VERSION"]
        self.prv_candidates_processor = Processor(
            ZTFPrvCandidatesStrategy()
        )  # initial strategy (can change)

    def process_prv_candidates(
        self, alerts: pd.DataFrame
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
        data = alerts[
            ["aid", "oid", "tid", "candid", "ra", "dec", "pid", "extra_fields"]
        ]
        detections = []
        non_detections = []
        for tid, subset_data in data.groupby("tid"):
            if tid == "ZTF":
                self.prv_candidates_processor.strategy = (
                    ZTFPrvCandidatesStrategy()
                )
            elif "ATLAS" in tid:
                self.prv_candidates_processor.strategy = (
                    ATLASPrvCandidatesStrategy()
                )
            else:
                raise ValueError(f"Unknown Survey {tid}")
            det, non_det = self.prv_candidates_processor.compute(subset_data)
            detections.append(det)
            non_detections.append(non_det)
        detections = pd.concat(detections, ignore_index=True)
        non_detections = pd.concat(non_detections, ignore_index=True)
        return detections, non_detections

    def execute(self, messages):
        self.logger.info(f"Processing {len(messages)} alerts")
        alerts = pd.DataFrame(messages)
        # If is an empiric alert must has stamp
        alerts["has_stamp"] = True
        # Process previous candidates of each alert
        (
            dets_from_prv_candidates,
            non_dets_from_prv_candidates,
        ) = self.process_prv_candidates(alerts)
        # If is an alert from previous candidate hasn't stamps
        # Concat detections from alerts and detections from previous candidates
        if dets_from_prv_candidates.empty:
            detections = alerts.copy()
            detections["parent_candid"] = np.nan
        else:
            dets_from_prv_candidates["has_stamp"] = False
            detections = pd.concat(
                [alerts, dets_from_prv_candidates], ignore_index=True
            )
        # Remove alerts with the same candid duplicated.
        # It may be the case that some candids are repeated or some
        # detections from prv_candidates share the candid.
        # We use keep='first' for maintain the candid of empiric detections.
        detections.drop_duplicates(
            "candid", inplace=True, keep="first", ignore_index=True
        )