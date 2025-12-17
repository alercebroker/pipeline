import logging
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List
import json

from core.DB.database_connection import PSQLConnection
from apf.core import get_class
from core.parsers.output_message_parsing.parser_utils import NumpyEncoder, parse_output_correction
from core.parsers.output_message_parsing.scribe_parser import scribe_parser_survey
from core.Corrector.corrector_multisurvey import Corrector

from core.parsers.input_message_parsing import get_input_message_parser
from core.DB.database_strategy import get_database_strategy
from core.parsers.survey_data_join.surveyDataProcessor import SurveyDataProcessor

class CorrectionMultistreamStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultistreamStep")
        self.scribe_enabled = config.get("SCRIBE_ENABLED", True)
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.survey = self.config.get("SURVEY")
        self.producer.set_key_field("oid")


    def execute(self, messages: List[dict]) -> dict:
        # Select the input message parser (to sepate the different data sources from the message into pandas dataframes without losing precision)
        input_parser = get_input_message_parser(self.survey)

        # Parse the data using the parser corresponding to the survey strategy
        parsed_data = input_parser.parse_input_messages(messages)

        # Get the list of all oids and measurement_ids. This will be used to query the database
        oids = parsed_data['oids']
        measurement_ids = parsed_data['measurement_ids']
    
        # Select the strategy to query to the DDBB 
        db_strategy = get_database_strategy(self.survey, self.db_sql)  

        # Obtain all the data from the database according to the corresponding survey
        # The data is parsed using the schemas corresponding to the survey to avoid precision loss
        historical_data = db_strategy.get_all_historical_data_as_dataframes(oids)

        # Create the processor
        processor = SurveyDataProcessor()

        # Process the data using the appropriate strategy
        processed_data = processor.process_survey_data(self.survey, parsed_data, historical_data)

       # Apply the correction to processed data, which has joined the historical data with the message data
        corrector_multisurvey = Corrector(processed_data)
        corrected_data = corrector_multisurvey.set_survey(self.survey).corrected_as_dataframes()
        
        # Parse the output message
        parsed_output = parse_output_correction(corrected_data, measurement_ids, oids)
 
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
        self.producer.poll(0)
        return

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
