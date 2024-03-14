import fastavro
import io
import logging
from apf.core.step import GenericStep
from fastavro import writer
import boto3
from botocore.exceptions import ClientError
from uuid import uuid4
from .jd import jd_to_date


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

        self.bucket_name = {
            "ztf": config["ZTF_BUCKET_NAME"],
            "atlas": config["ATLAS_BUCKET_NAME"],
        }

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

    def deserialize_message(self, message):
        bytes_io = io.BytesIO(message.value())
        reader = fastavro.reader(bytes_io)
        data = reader.next()
        schema = reader.writer_schema
        return data, schema

    def get_date(self, message: dict):
        jd = message["candidate"]["jd"]
        year, month, day = jd_to_date(jd)
        return f"{int( year )}{int( month )}{int( day )}"

    def deserialize_messages(self, messages):
        deserialized = {}
        for message in messages:
            if message.error():
                if message.error().name() == "_PARTITION_EOF":
                    self.logger.info("PARTITION_EOF: No more messages")
                    return
                self.logger.exception(f"Error in kafka stream: {message.error()}")
                continue
            else:
                message, schema = self.deserialize_message(message)
                date = self.get_date(message)
                self.schema = schema
                if date in deserialized:
                    deserialized[date].append(message)
                else:
                    deserialized[date] = [message]

        self.messages = messages
        return deserialized

    def execute(self, messages):
        ################################
        #   Here comes the Step Logic  #
        ################################
        clean_messages = self.deserialize_messages(messages)
        for date in clean_messages:
            partition_name = str(uuid4())  # count
            file_name = date + "_" + partition_name + ".avro"
            fo = io.BytesIO()
            writer(fo, self.schema, clean_messages[date], codec="snappy")
            object_name = "{}_{}/{}".format(self.formatt, date, file_name)

            # Reset read pointer. DOT NOT FORGET THIS, else all uploaded files will be empty!
            fo.seek(0)

            self.upload_file(fo, self.bucket_name["ztf"], object_name)
