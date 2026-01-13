import logging
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List
import json

from core.DB.database_connection import PSQLConnection
from apf.core import get_class
from core.parsers.output_message_parsing.parser_utils import NumpyEncoder, get_measurement_ids, parse_output_correction, get_sids_from_messages
from core.parsers.output_message_parsing.scribe_parser import scribe_parser_survey
from core.Corrector.corrector_multisurvey import Corrector

from core.surveys.registry import get_survey_components


class CorrectionMultisurveyStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultisurveyStep")
        self.scribe_enabled = config.get("SCRIBE_ENABLED", True)
        
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        
        self.survey = self.config.get("SURVEY")
        if not self.survey:
            raise ValueError("SURVEY must be specified in config")
        
        self.survey = self.survey.lower()
        self.producer.set_key_field("oid")
        
        # Survey internal processees for parsing the input, the strategy to get the historical data and the joiner
        # of the historical data with the message data are done via components specific to each survey
        components = get_survey_components(self.survey, self.db_sql)
        self.input_parser = components['input_parser']
        self.db_strategy = components['db_strategy']
        self.joiner = components['joiner']
        
        self.logger.info(f"Initialized components for survey: {self.survey}")

    def execute(self, messages: List[dict]) -> dict:
        """Execute correction pipeline."""
        
        parsed_data, oid_to_sids = self.input_parser.parse_input_messages(messages)
        oids = parsed_data['oids']
        sids = get_sids_from_messages(messages)
    
        historical_data = self.db_strategy.get_all_historical_data_as_dataframes(oid_to_sids)
        processed_data = self.joiner.process(parsed_data, historical_data)

        corrector = Corrector(processed_data)
        corrector.set_survey(self.survey)
        corrected_data = corrector.corrected_as_dataframes()
        
        measurement_ids = get_measurement_ids(corrected_data, self.survey)
        
        parsed_output = parse_output_correction(corrected_data, measurement_ids, oids, sids)
        return parsed_output

    def post_execute(self, result: dict):
        self.produce_scribe(scribe_parser_survey(result, self.survey))
        return result

    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            payload = {"payload": scribe_data}
            oid = scribe_data["payload"]["oid"]
            self.scribe_producer.producer.produce(
                topic="scribe-multisurvey",
                key=str(oid).encode("utf-8"),               
                value=json.dumps(payload, cls=NumpyEncoder).encode("utf-8"), 
            )
            self.scribe_producer.producer.poll(0)

    def post_produce(self):
        self.producer.producer.poll(0)
        return 

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()