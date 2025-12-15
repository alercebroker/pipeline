import json
import numpy as np
import pandas as pd
from typing import List
from apf.core.step import GenericStep, get_class
from importlib.metadata import version
import copy
from .core.parsers.object_update_lsst import update_object
from .core.parsers.parser_scribe import scribe_parser, scribe_parser_objects, NumpyEncoder, remove_timestamp
from .core.parsers.add_meancoordinates import refresh_mean_coordinates
from .core.StatisticsSelector.statistics_selector import get_object_statistics_class, get_magnitude_statistics_class
from .core.parsers.survey_preparser import SurveyDataSelector


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
        self.data_selector = SurveyDataSelector(self.survey)
        self.producer.set_key_field("oid")


    def execute(self, messages: List[dict]):
        
        def remove_timestamp(message: dict):
            message.pop("timestamp", None)
            return message

        messages_original = list(map(remove_timestamp, copy.deepcopy(messages)))
        
        # Extract data separating into dets, forced, non-dets, ss_dets (the last is filled only for LSST)
        survey_data = self.data_selector.extract_data(messages)
        
        # Initialize list and dict to collect all magstats and objstats from different detection types (for sid 1 and sid 2)
        all_magstats = []
        all_objstats = {}
        
        # Process SS detections first (LSST only)
        if self.survey.lower() == "lsst" and survey_data.ss_detections:
            ss_magstats = self._calculate_magstats(
                survey_data.ss_detections, survey_data.non_detections
            )
            ss_objstats = self._calculate_objstats(
                survey_data.ss_detections, 
                survey_data.non_detections, 
                survey_data.forced_photometries
            )
            all_magstats.append(ss_magstats)
            all_objstats.update(ss_objstats)
        
        # Process regular detections common to all surveys
        if survey_data.detections:
            reg_magstats = self._calculate_magstats(
                survey_data.detections, survey_data.non_detections
            )
            reg_objstats = self._calculate_objstats(
                survey_data.detections, 
                survey_data.non_detections, 
                survey_data.forced_photometries
            )
            all_magstats.append(reg_magstats)
            all_objstats.update(reg_objstats)
        
        # Concatenate magstats (ss_detections first, then regular detections so later ss detections count of ndets/fphots could be rewritten)
        if len(all_magstats) > 1:
            magstats = pd.concat(all_magstats, axis=0)
        elif len(all_magstats) == 1:
            magstats = all_magstats[0]
        
        # Parse magstats results into objstats
        for oid in all_objstats:
            if not magstats.empty and oid in magstats.index:
                self.parse_magstats_result(magstats, oid, all_objstats)
        
        # Update the messages with refreshed mean coordinates #!TODO: CHECK IF THIS IS OK
        messages_updated = refresh_mean_coordinates(messages_original, all_objstats)
        update_object_list = []
        if self.survey.lower() == "lsst":
            update_object_list = update_object(messages_updated, all_objstats)
        
        result = [messages_updated, all_objstats, update_object_list]
        return result

    
    def _calculate_objstats(self, detections, non_detections, forced_detections) -> dict:
        """Calculate survey-specific objectstatistics"""
        obj_stats_class = get_object_statistics_class(self.survey)
        obj_calculator = obj_stats_class(detections, non_detections, forced_detections, self.survey)
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
        objects_to_update = result[2]
        self.produce_scribe(scribe_parser(objstats, self.survey))
        self.produce_scribe(scribe_parser_objects(objects_to_update, self.survey))
        return messages_updated


    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            oid = scribe_data["oid"]
            self.scribe_producer.producer.produce(
                topic="scribe-multisurvey",
                key=str(oid).encode("utf-8"),               
                value=json.dumps(scribe_data).encode("utf-8"),  
            )
