import numpy as np
import pandas as pd
import logging
import traceback
from alerce_classifiers.base.dto import OutputDTO
from .kafka_parser import KafkaOutput, KafkaParser


class StampParser(KafkaParser):
    def __init__(self):
        super().__init__(None)

    def parse(
        self, model_output: OutputDTO, messages: dict, **kwargs
    ) -> KafkaOutput[list]:
        parsed = []
        probabilities = model_output.probabilities
        for oid, msg in messages.items():
            try:
                data = self._get_probabilities_by_oid(oid, probabilities=probabilities)
                parsed.append(
                    {
                        "oid": oid,
                        "measurement_id": msg["candid"],
                        "probabilities": data,
                        "sid": 0,
                    }
                )
            except Exception as e:
                logging.error("Message with no probability")
                logging.error(e)
                logging.error(traceback.print_exc())

        return KafkaOutput(parsed)

    def _get_probabilities_by_oid(self, oid, probabilities):
        if oid not in probabilities.index:
            return {"error": "OID not found"}
        return probabilities.loc[oid].to_dict()
