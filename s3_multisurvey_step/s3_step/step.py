from typing import List
import io
import logging
import boto3
from apf.core.step import GenericStep
from s3_step.alert_manager import manager_selector


class S3Step(GenericStep):
    """S3Step Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)

        alert_manager_class = manager_selector(self.config["SURVEY_ID"])
        self.alert_manager = alert_manager_class(self.config["BUCKET_CONFIG"])
        self.s3_client = boto3.client(
            "s3",
            region_name=self.alert_manager.bucket_region,
        )


    def upload_file(self, file_name: str, file_data: io.BytesIO):
        """
        Uploads a avro file to s3 storage

        You have to configure STORAGE settings in the step. A dictionary like this is required:

        .. code-block:: python

            STEP_CONFIG = {
                "STORAGE": {
                    "REGION_NAME": "",
                }
            }

        Parameters
        ----------
        f : file-like object
            Readable file like object that will be uploaded
        candid : int
            candid of the avro file. Avro files are stored using avro as object name
        bucket_name : str
            name of the s3 bucket
        """
        print("IO DATA")
        print("\n\n\n\n--------")
        self.s3_client.upload_fileobj(file_data, self.alert_manager.bucket_name, file_name)
        

    def execute(self, messages: List[dict]):
        self.alert_manager.process_alerts(messages)

        for file_name, file_data in self.alert_manager.process_alerts(messages):
            self.upload_file(file_name, file_data)

        return {}