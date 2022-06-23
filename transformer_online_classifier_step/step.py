import pandas as pd
import numpy as np
import datetime
import logging
from apf.core.step import GenericStep
from apf.producers import KafkaProducer
from alerce_classifiers.transformer_online_classifier import TransformerOnlineClassifier
from typing import List


class TransformerOnlineClassifierStep(GenericStep):
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
            self.producer = producer or KafkaProducer(prod_config)
        else:
            self.producer = None

        self.model = TransformerOnlineClassifier("../Encoder.pt")

        self._rename_cols = {
            "mag": "FLUXCAL",
            "e_mag": "FLUXCALERR",
            "fid": "BAND",
            "mjd": "MJD"
        }

        self._fid_mapper = {
            0: "u",
            1: "g",
            2: "r",
            3: "i",
            4: "z",
            5: "Y",
        }

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

    def map_detections(self, light_curves: pd.DataFrame) -> pd.DataFrame:
        light_curves.drop(columns=["meanra", "meandec", "ndet", "non_detections", "metadata"], inplace=True)
        exploded = light_curves.explode("detections")
        detections = pd.DataFrame.from_records(exploded["detections"].values, index=exploded.index)
        detections = detections[self._rename_cols.keys()]
        detections = detections.rename(columns=self._rename_cols)
        detections["BAND"] = detections["BAND"].map(lambda x: self._fid_mapper[x])
        return detections

    def format_output_message(self, predictions: pd.DataFrame, light_curves: pd.DataFrame) -> List[dict]:
        for class_name in self._class_mapper.keys():
            if class_name not in predictions.columns:
                predictions[class_name] = 0.0
        classifications = lambda x: [{
            "classifierName": "balto_classifier",
            "classifierParams": "version1.0.0",
            "classId": self._class_mapper[predicted_class],
            "probability": predicted_prob
        }
            for predicted_class, predicted_prob in x.iteritems()]

        response = pd.DataFrame({
            "classifications": predictions.apply(classifications, axis=1),
        })
        response["brokerPublishTimestamp"] = int(datetime.datetime.now().timestamp() * 1000)
        response["brokerName"] = "ALeRCE"
        response["brokerVersion"] = "1.0.0"
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
        detections = self.map_detections(light_curves_dataframe)
        predictions = self.model.predict_proba(detections)
        output = self.format_output_message(predictions, light_curves_dataframe)
        self.produce(output)
