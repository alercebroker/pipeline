import json
import pandas as pd
from typing import List
from apf.core.step import GenericStep, get_class
import copy
from .core.parsers.parser_scribe import scribe_parser, remove_timestamp
from .core.parsers.add_meancoordinates import refresh_mean_coordinates
from .core.surveys.survey_registry import SurveyRegistry


class MagstatsStep_Multisurvey(GenericStep):
    """Refactored step with survey logic delegated to handlers"""
    
    def __init__(self, config, **step_args):
        super().__init__(config=config, **step_args)
        self.excluded = set(config["EXCLUDED_CALCULATORS"])
        
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])
        
        self.survey = config["SURVEY"]
        self.survey_handler = SurveyRegistry.get_handler(self.survey, self.excluded)
        
        self.producer.set_key_field("oid")

    def execute(self, messages: List[dict]):
        messages_original = [remove_timestamp(copy.deepcopy(msg)) for msg in messages]
        
        survey_data = self.survey_handler.extract_data(messages)
        
        results = self.survey_handler.process(survey_data)
        
        magstats = self._concat_magstats(results.magstats)
        
        self._attach_magstats(magstats, results.objstats)
        
        messages_updated = refresh_mean_coordinates(messages_original, results.objstats)
        return [messages_updated, results.objstats]
    
    def _concat_magstats(self, magstats_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multiple magstats DataFrames"""
        if len(magstats_list) > 1:
            return pd.concat(magstats_list, axis=0)
        elif len(magstats_list) == 1:
            return magstats_list[0]
        return pd.DataFrame()
    
    def _attach_magstats(self, magstats: pd.DataFrame, objstats: dict):
        """Attach magstats to objstats dictionary"""
        if magstats.empty:
            return
        
        for oid in objstats:
            if oid in magstats.index:
                self.parse_magstats_result(magstats, oid, objstats)
    
    def parse_magstats_result(self, magstats: pd.DataFrame, oid: str, stats: dict):
        """Parse magstats results into objstats structure"""
        magstats_by_oid = magstats.loc[oid]

        if isinstance(magstats_by_oid, pd.Series):
            self._add_magstat_row(magstats_by_oid.to_dict(), stats, oid)
        elif isinstance(magstats_by_oid, pd.DataFrame):
            for _, row in magstats_by_oid.iterrows():
                self._add_magstat_row(row.to_dict(), stats, oid)
        else:
            raise TypeError(f"Unknown magstats type {type(magstats_by_oid)}")
    
    @staticmethod
    def _add_magstat_row(row: dict, stats: dict, oid: str):
        """Add a single magstat row to objstats"""
        sid = int(row.pop("sid"))
        stats[oid][sid].setdefault("magstats", {})
        stats[oid][sid]["magstats"].setdefault(sid, [])
        stats[oid][sid]["magstats"][sid].append(row)

    def post_execute(self, result: List[dict]):
        messages_updated = result[0]
        objstats = result[1]
        self.produce_scribe(scribe_parser(objstats, self.survey))
        return messages_updated

    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            oid = scribe_data["oid"]
            self.scribe_producer.producer.produce(
                topic="scribe-multisurvey",
                key=str(oid).encode("utf-8"),
                value=json.dumps(scribe_data).encode("utf-8"),
            )
            self.scribe_producer.producer.poll(0)