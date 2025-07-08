import logging
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List
import json

from core.DB.database_connection import PSQLConnection
from apf.core import get_class
from core.parsers.output_message_parsing.parser_utils import NumpyEncoder, parse_output_df_lsst, parse_data_for_avro
from core.parsers.output_message_parsing.scribe_parser import scribe_parser
from core.Corrector.corrector import Corrector

from core.parsers.input_message_parsing import get_input_message_parser
from core.DB.database_strategy import get_database_strategy
from core.parsers.survey_data_join.surveyDataProcessor import SurveyDataProcessor

class CorrectionMultistreamLSSTStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultistreamLSSTStep")
        self.scribe_enabled = config.get("SCRIBE_ENABLED", True)
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])
        self.survey = "lsst" # Add it later to the config yaml

    # The data will come parsed differently, as we will have detections separated into sources, forced sources, and previous sources
    # We will not have the extra fields separation either, so no need to add them back in (Everything will be in the main fields)
    def execute(self, messages: List[dict]) -> dict:
        # For now we handle it here, but later it should be in the config yaml
        survey = self.config.get("SURVEY", "lsst")

        # Select the input message parser (to sepate the different data sources from the message into pandas dataframes without losing precision)
        input_parser = get_input_message_parser(survey)

        # Parse the data using the parser corresponding to the survey strategy
        parsed_data = input_parser.parse_input_messages(messages)
        
        #! maybe this should be somwhere else to keep it cleaner? => Create the input parser with logger so we can process the log inside the parser
        for data_type, count in parsed_data['counts'].items():
            self.logger.info(f"Received {count} {data_type}")

        # Get the list of all oids and measurement_ids. This will be used to query the database
        oids = parsed_data['oids']
        measurement_ids = parsed_data['measurement_ids']

        # Select the strategy to query to the DDBB 
        db_strategy = get_database_strategy(survey, self.db_sql)  

        # Obtain all the data from the database according to the corresponding survey
        # The data is parsed using the schemas corresponding to the survey to avoid precision loss
        historical_data = db_strategy.get_all_historical_data_as_dataframes(oids)
        
        #TODO verify this logger the logger info
        #! Put the logger inside the database strategy so we can process the logger inside the database strategy
        self.logger.info(f"Retrieved {historical_data['detections'].shape[0]} detections, {historical_data['forced_detections'].shape[0]} \
                         forced photometries and  {historical_data['non_detections'].shape[0]} non detections from the database")
        
        # Create the processor
        processor = SurveyDataProcessor()

        # Process the data using the appropriate strategy
        processed_data = processor.process_survey_data(survey, parsed_data, historical_data)

        #TODO add the logger info here as well but then we need to add the logger to the database strategy so we can process the logger inside the database strategy
        #self.logger.debug(f"Obtained {len(sources_df[sources_df['new']]) + len(previous_sources_df[previous_sources_df['new']]) + \
        #    len(forced_sources_df[forced_sources_df['new']])} new detections and {len(non_detections_df)} new sources")
        
        print(processed_data)
        print("post processed data arrived :o")
        exit()

        #TODO add corrector with selection per survey here 
        # For lsst the corrector currently does nothing but we must implement the structure for the other surveys
        # Apply correction

        # Parse the output message for each survey
        # Handle with a strategy pattern for each survey
        parsed_output = parse_output_df_lsst(sources_df, previous_sources_df, forced_sources_df, non_detections_df, df_dia_object, df_ss_object, measurement_ids)
        clean_output = parse_data_for_avro(parsed_output)
        return clean_output

    def post_execute(self, result: dict):
        self.produce_scribe(scribe_parser(result))
        return result

    def produce_scribe(self, scribe_payloads):
        for scribe_data in scribe_payloads:
            payload = {"payload": json.dumps(scribe_data, cls=NumpyEncoder)}
            self.scribe_producer.produce(payload)

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
