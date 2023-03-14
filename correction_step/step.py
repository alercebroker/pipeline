import logging
import json

import pandas as pd
import numpy as np
from apf.core import get_class
from apf.core.step import GenericStep

from .core.correction.corrector import Corrector
from .core.strategies import (
    ZTFCorrectionStrategy,
    ATLASCorrectionStrategy,
)


class CorrectionStep(GenericStep):
    """Step that applies magnitude correction to new alert and previous candidates.

    The correction refers to passing from the magnitude measured from the flux in the difference
    stamp to the actual apparent magnitude. This requires a reference magnitude to work.
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
        cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])

    def pre_produce(self, result: tuple):
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

    def correct(self, messages: list[dict]) -> list[dict]:
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
        self.logger.info(f"Processing {len(messages)} alerts")
        detections = self.correct(messages)
        return detections

    def post_execute(self, result: list[dict]):
        for detection in result:
            self.produce_scribe(detection)
        return result

    def produce_scribe(self, detection: dict):
        scribe_data = {
            "collection": "detection",
            "type": "update",
            "criteria": {"_id": detection["candid"]},
            "data": detection,
            "options": {"upsert": True},
        }
        scribe_payload = {"payload": json.dumps(scribe_data)}
        self.scribe_producer.produce(scribe_payload)
