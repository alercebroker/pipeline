from apf.core.step import GenericStep
from prv_candidates_step.core.candidates.process_prv_candidates import (
    process_prv_candidates,
)
from prv_candidates_step.core.strategy.ztf_strategy import ZTFPrvCandidatesStrategy
from prv_candidates_step.core.processor.processor import Processor


import numpy as np
import pandas as pd
import logging


class PrvCandidatesStep(GenericStep):
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
        config,
        level=logging.INFO,
        **step_args,
    ):
        super().__init__(config=config, level=level, **step_args)
        self.prv_candidates_processor = Processor(
            ZTFPrvCandidatesStrategy()
        )  # initial strategy (can change)
        self.producers = {"scribe": None, "alerts": None}

    def pre_produce(self, result: pd.DataFrame):
        self.set_producer_key_field("aid")
        return result.to_dict("records")

    def execute(self, messages):
        self.logger.info("Processing %s alerts", str(len(messages)))
        alerts_with_prv_candidates = list(
            filter(
                lambda alert: alert.get("extra_fields", {}).get("prv_candidates")
                is not None,
                messages,
            )
        )
        prv_detections, non_detections = process_prv_candidates(
            self.prv_candidates_processor, alerts_with_prv_candidates
        )

        return messages, prv_detections, non_detections
