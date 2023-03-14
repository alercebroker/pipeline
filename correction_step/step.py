from typing import List, Tuple

import numpy as np
from apf.core.step import GenericStep
from .core.correction.corrector import Corrector
from .core.strategies import (
    ZTFCorrectionStrategy,
    ATLASCorrectionStrategy,
)

import pandas as pd
import logging
import json
from apf.core import get_class


class DetectionStep(GenericStep):
    """DetectionStep Description

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
        self.detections_corrector = Corrector(
            ZTFCorrectionStrategy()
        )  # initial strategy (can change)
        producer_class = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = producer_class(self.config["SCRIBE_PRODUCER_CONFIG"])

    def pre_produce(self, result: Tuple):
        self.set_producer_key_field("aid")
        output = []
        for index, alert in enumerate(result[0]):
            output.append(
                {
                    "aid": alert["aid"],
                    "detections": result[1][index],
                    "non_detections": alert["non_detections"],
                }
            )
        return output

    def correct(self, messages: List[dict]) -> List[List[dict]]:
        """Correct Detections.

        Parameters
        ----------
        messages

        Returns
        -------

        """
        corrections = []
        for message in messages:
            detections = message["prv_detections"] + [message["new_alert"]]
            if "ZTF" == message["new_alert"]["tid"]:
                self.detections_corrector.strategy = ZTFCorrectionStrategy()
            elif "ATLAS" == message["new_alert"]["tid"]:
                self.detections_corrector.strategy = ATLASCorrectionStrategy()
            df = pd.DataFrame(detections).replace({np.nan: None})
            df["rfid"] = df["rfid"].astype("Int64")
            correction_df = self.detections_corrector.compute(df)
            alert_corrections = correction_df.to_dict("records")
            corrections.append(alert_corrections)
        return corrections

    def execute(self, messages):
        self.logger.info("Processing %s alerts", str(len(messages)))
        # If is an empiric alert must has stamp
        # Do correction to detections from stream
        detections = self.correct(messages)
        return messages, detections

    def post_execute(self, result: Tuple):
        for detections in result[1]:
            self.produce_scribe(detections)
        return result

    def produce_scribe(self, detections: List[dict]):
        for detection in detections:
            # candid = detection.pop('candid')
            scribe_data = {
                "collection": "detection",
                "type": "update",
                "criteria": {"_id": detection["candid"]},
                "data": detection,
                "options": {"upsert": True},
            }
            scribe_payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(scribe_payload)
