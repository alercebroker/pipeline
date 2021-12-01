from apf.core.step import GenericStep
from io import BytesIO
from fastavro import writer
import logging
import boto3
from botocore.exceptions import ClientError


class AlertArchivingStep(GenericStep):
    """AlertArchivingStep Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level, **step_args)
        self.formatt = config["FORMAT"]  # = "avro"
        self.bucket_name = config["BUCKET_NAME"]  # = "alerts_archive"

    def upload_file(self, filee, bucket, object_name):
        """Upload a file to an S3 bucket

        :param file: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name.
        :return: True if file was uploaded, else False
        """

        # Upload the file
        s3_client = boto3.client("s3")
        try:
            response = s3_client.upload_fileobj(filee, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def execute(self, messages):
        """Process each consumed alert or batch"""
        parsed_schema = ""
        topic_date = ""  # yyyymmdd
        partition_name = ""  # count

        file_name = topic_date + "_" + partition_name

        fo = BytesIO()
        writer(fo, parsed_schema, messages)

        survey = ""  # ztf
        object_name = "{}/{}_{}".format(survey, self.formatt, topic_date)

        self.upload_file(file_name, self.bucket_name, object_name)
