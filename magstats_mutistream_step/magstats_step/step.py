import json
import numpy as np
import pandas as pd
from typing import List
from apf.core.step import GenericStep, get_class
from importlib.metadata import version
import copy
from .core.parsers.parser_scribe import scribe_parser, NumpyEncoder, remove_timestamp
from .core.parsers.add_meancoordinates import refresh_mean_coordinates
from .core.StatisticsSelector.statistics_selector import get_object_statistics_class, get_magnitude_statistics_class



class MagstatsStep_Multistream(GenericStep):
    def __init__(
        self,
        config,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        self.excluded = set(config["EXCLUDED_CALCULATORS"])
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])
        self.survey = config["SURVEY"]

    def execute(self, messages: List[dict]):
        messages = list(map(remove_timestamp, messages))
        messages_original = copy.deepcopy(messages) # Keep the original messages without altering
        detections, non_detections = [], []
        for msg in messages:
            # code to add mjdendref to detections fields from extra fields
            for det in msg["detections"]:
                jdendref = det["extra_fields"].get("jdendref", None)
                if jdendref:
                    det["mjdendref"] = jdendref - 2400000.5
            detections.extend(msg["detections"])
            non_detections.extend(msg["non_detections"])
        message = {"detections": detections, "non_detections": non_detections}

        magstats = self._calculate_magstats(
            message["detections"], message["non_detections"]
        )
        objstats = self._calculate_objstats(
            message["detections"]
        )
        for oid in objstats:
            self.parse_magstats_result(magstats, oid, objstats)
        
        messages_updated = refresh_mean_coordinates(messages_original, objstats)
        result = [messages_updated, objstats] # Send the updated messages and objstats together to produce the results correctly
        return result

    
    def _calculate_objstats(self, detections) -> pd.DataFrame:
        """Calculate survey-specific objectstatistics"""
        obj_stats_class = get_object_statistics_class(self.survey)
        obj_calculator = obj_stats_class(detections)
        stats = obj_calculator.generate_statistics(self.excluded).replace(
            {np.nan: None}
        )
        assert stats.index.name == "oid"
        return stats.to_dict("index")
    
    def _calculate_magstats(self, detections, non_detections) -> pd.DataFrame:
        """Calculate survey-specific magnitude statistics"""
        mag_stats_class = get_magnitude_statistics_class(self.survey)
        magstats_calculator = mag_stats_class(
            detections=detections, non_detections=non_detections
        )
        stats = (
            magstats_calculator.generate_statistics(self.excluded)
            .reset_index()
            .set_index("oid")
            .replace({np.nan: None})
        )
        return stats

    def parse_magstats_result(
        self, magstats: pd.DataFrame, oid: str, stats: dict
    ) -> dict:
        """Adds magstats in the correct format for the oid
        **Note**: Updates stats dictionary in place
        """
        magstats_by_oid = magstats.loc[oid]
        # magstats_by_oid could have multiple fid values,
        # so the loc could potentially return a dataframe with a row
        # for each fid
        if isinstance(magstats_by_oid, pd.Series):
            # when calling to_dict on a series
            # the result is a single dict
            # we then ensure that it is always a list of dict
            stats[oid]["magstats"] = [magstats_by_oid.to_dict()]
        elif isinstance(magstats_by_oid, pd.DataFrame):
            # when calling to_dict on a dataframe with orient=records
            # the result should be already a list of dicts
            stats[oid]["magstats"] = magstats_by_oid.to_dict(orient="records")
        else:
            raise TypeError(f"Unknown magstats type {type(magstats_by_oid)}")

    def post_execute(self, result: List[dict]):
        messages_updated = result[0]
        objstats = result[1]
        self.produce_scribe(scribe_parser(objstats, self.survey))
        return messages_updated

    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            payload = {"payload": json.dumps(scribe_data, cls=NumpyEncoder)}
            self.scribe_producer.produce(payload)



