from abc import abstractmethod
from io import BytesIO
from typing import List
from fastavro import schema

class BaseAlertManager:
    survey_id = None
    """
    Base class for alert managers.
    """
    def __init__(self, bucket_config: dict):
        self.bucket_region = bucket_config["REGION"]
        self.bucket_name = bucket_config["BUCKET_NAME"]
        self.parsed_schema = schema.load_schema(bucket_config["SCHEMA_FILE"])
        self.messages = {}
        
        self.check_bucket_name()

    @abstractmethod
    def get_file_name(self, alert: str) -> str:
        """
        Returns the file name for the file based on the alerts from the messages.
        """
        pass

    @abstractmethod
    def get_file_data(self, alert: str) -> BytesIO:
        """
        Returns the file data for the file based on the alerts from the messages.
        """
        pass

    
    def check_bucket_name(self):
        """
        Checks if the bucket name is valid.
        The bucket name must match the Manager survey
        """

        if not self.survey_id in self.bucket_name:
            raise ValueError(
                f"Bucket name {self.bucket_name} does not match the survey {self.survey_id}"
            )
        
    def process_alerts(self, alerts: List[dict]):
        """
        Processes the alerts from the messages.
        """
        for alert in alerts:
            file_name = self.get_file_name(alert)
            file_data = self.get_file_data(alert)
            yield file_name, file_data