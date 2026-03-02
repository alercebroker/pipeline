from typing import List
import io
import logging
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            config=boto3.session.Config(
                max_pool_connections=50,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
        )
        self.max_workers = self.config.get("THREAD_WORKERS", 10)
    
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
        file_name : str
            name of the file to be uploaded to S3
        file_data : io.BytesIO
            Readable file like object that will be uploaded
            
        Returns
        -------
        tuple
            (file_name, success_status)
        """
        try:
            self.s3_client.upload_fileobj(file_data, self.alert_manager.bucket_name, file_name)
            return file_name, True
        except Exception as e:
            self.logger.error(f"Failed to upload {file_name}: {e}")
            return file_name, False
        
    def execute(self, messages: List[dict]):
        """
        Processes and uploads alert files to S3 in parallel.
        
        Parameters
        ----------
        messages : List[dict]
            List of alert messages to process and upload
            
        Returns
        -------
        dict
            Dictionary containing upload statistics (uploaded count and failed count)
        """
        files_to_upload = list(self.alert_manager.process_alerts(messages))
        
        if not files_to_upload:
            return {}
        
        uploaded_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.upload_file, file_name, file_data): file_name
                for file_name, file_data in files_to_upload
            }
            
            for future in as_completed(future_to_file):
                file_name, success = future.result()
                if success:
                    uploaded_count += 1
                else:
                    failed_count += 1
        
        self.logger.info(f"Uploaded {uploaded_count} files, {failed_count} failed")
        return {}