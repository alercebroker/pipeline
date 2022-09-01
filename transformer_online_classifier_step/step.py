import os

import pandas as pd
import numpy as np
import datetime
import logging
from apf.core.step import GenericStep
from apf.producers import KafkaSchemalessProducer
from alerce_classifiers.transformer_lc_header import TransformerLCHeaderClassifier
from typing import List
import warnings

warnings.filterwarnings("ignore")


class TransformerLCHeaderClassifierStep(GenericStep):
    """TransformerOnlineClassifierStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, producer=None, **step_args):
        super().__init__(consumer, config=config, level=level)
        prod_config = self.config.get("PRODUCER_CONFIG", None)
        if prod_config:
            self.producer = producer or KafkaSchemalessProducer(prod_config)
        else:
            self.producer = None

        self.model = TransformerLCHeaderClassifier(self.config.get("MODEL_PATH"), self.config.get("QUANTILES_PATH"))
        self.model_version = os.getenv("MODEL_VERSION", "0.0.0")
        self._class_mapper = {
            "Periodic/Other": 210,
            "Cepheid": 211,
            "RR Lyrae": 212,
            "Delta Scuti": 213,
            "EB": 214,
            "LPV/Mira": 215,
            "Non-Periodic/Other": 220,
            "AGN": 221,
            "SN-like/Other": 110,
            "Ia": 111,
            "Ib/c": 112,
            "II": 113,
            "Iax": 114,
            "91bg": 115,
            "Fast/Other": 120,
            "KN": 121,
            "M-dwarf Flare": 122,
            "Dwarf Novae": 123,
            "uLens": 124,
            "Long/Other": 130,
            "SLSN": 131,
            "TDE": 132,
            "ILOT": 133,
            "CART": 134,
            "PISN": 135
        }

    def format_output_message(self, predictions: pd.DataFrame, light_curves: pd.DataFrame) -> List[dict]:
        for class_name in self._class_mapper.keys():
            if class_name not in predictions.columns:
                predictions[class_name] = 0.0
        classifications = lambda x: [{
            "classifierName": os.getenv("CLASSIFIER_NAME", "balto"),
            "classifierParams": self.model_version,
            "classId": self._class_mapper[predicted_class],
            "probability": predicted_prob
        }
            for predicted_class, predicted_prob in x.iteritems()]

        response = pd.DataFrame({
            "classifications": predictions.apply(classifications, axis=1),
            "brokerVersion": [self.model_version] * len(predictions)
        })
        response["brokerPublishTimestamp"] = int(datetime.datetime.now().timestamp() * 1000)
        response["brokerName"] = "ALeRCE"
        response = response.join(light_curves)
        response.replace({np.nan: None}, inplace=True)
        response.rename(columns={"candid": "diaSourceId"}, inplace=True)
        response["alertId"] = response["alertId"].astype(int)
        response["diaSourceId"] = response["diaSourceId"].astype(int)
        return response.to_dict("records")

    def produce(self, output_messages):
        for message in output_messages:
            aid = message["alertId"]
            self.producer.produce(message, key=str(aid))

    def execute(self, messages: List[dict]):
        light_curves_dataframe = pd.DataFrame(messages)
        light_curves_dataframe.drop_duplicates(subset="aid", inplace=True, keep="last")
        light_curves_dataframe.set_index("aid", inplace=True)
        self.logger.info(f"Processing {len(messages)} light curves.")
        predictions = self.model.predict_proba(light_curves_dataframe)
        output = self.format_output_message(predictions, light_curves_dataframe)
        self.produce(output)
