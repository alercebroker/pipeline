import logging
import json
from apf.core.step import GenericStep
from apf.consumers import KafkaConsumer
from typing import List

### Imports for lightcurve database queries
from core.DB.database_sql import (
    PSQLConnection,
)

from apf.core import get_class

from core.parsers.scribe_parser import scribe_parser

from .step_utils import ( 
    process_messages,
    fetch_database_data,
    merge_and_clean_data,
    apply_corrections,
    build_result
)

class CorrectionMultistreamZTFStep(GenericStep):
    def __init__(
        self,
        config: dict,
        db_sql: PSQLConnection,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.db_sql = db_sql
        self.logger = logging.getLogger("alerce.CorrectionMultistreamZTFStep")
        self.scribe_enabled = config.get("SCRIBE_ENABLED", True)
        if self.scribe_enabled:
            cls = get_class(self.config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
            self.scribe_producer = cls(self.config["SCRIBE_PRODUCER_CONFIG"])

    def execute(self, messages: List[dict]) -> dict:

        processed_data = process_messages(messages)
        db_data = fetch_database_data(processed_data['oids'], self.db_sql)
        merged_data = merge_and_clean_data(processed_data, db_data)
        corrected_data = apply_corrections(merged_data, self.config)
            
        return build_result(corrected_data, processed_data['msg_df'])

    def post_execute(self, result: dict):
        self.produce_scribe(scribe_parser(result))
        return result

    def produce_scribe(self, scribe_payloads):

        for scribe_data in scribe_payloads:
            payload = {"payload": json.dumps(scribe_data)}
            self.scribe_producer.produce(payload)

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
        self.producer.__del__()
