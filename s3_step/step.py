from apf.core.step import GenericStep
import logging
import io
import math
import datetime
from db_plugins.db.sql import SQLConnection
from db_plugins.db.sql.models import Step
import boto3
from botocore.config import Config


class S3Step(GenericStep):
    """S3Step Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    **step_args : type
        Other args passed to step (DB connections, API requests, etc.)

    """

    def __init__(
        self,
        consumer=None,
        config=None,
        db_connection=None,
        level=logging.INFO,
        **step_args
    ):
        super().__init__(consumer, config=config, level=level)
        self.db = db_connection or SQLConnection()
        self.db.connect(self.config["DB_CONFIG"]["SQL"])
        if not step_args.get("test_mode", False):
            self.insert_step_metadata()

    def insert_step_metadata(self):
        """
        Inserts step version and other metadata to step table.
        """
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_VERSION"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["STEP_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def get_object_url(self, bucket_name, candid):
        """
        Formats a valid s3 url for an avro file given a bucket and candid.
        The format for saving avros on s3 is <candid>.avro and they are
        all stored in the root directory of the bucket.

        Parameters
        ----------
        bucket_name : str
            name of the bucket
        candid : int
            candid of the avro to be stored
        """
        return "https://{}.s3.amazonaws.com/{}.avro".format(bucket_name, candid)

    def upload_file(self, f, candid, bucket_name):
        """
        Uploads a avro file to s3 storage

        You have to configure STORAGE settings in the step. A dictionary like this is required:

        .. code-block:: python

            STEP_CONFIG = {
                "STORAGE": {
                    "AWS_ACCESS_KEY": "",
                    "AWS_SECRET_ACCESS_KEY": "",
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
        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.config["STORAGE"]["AWS_ACCESS_KEY"],
            aws_secret_access_key=self.config["STORAGE"]["AWS_SECRET_ACCESS_KEY"],
            region_name=self.config["STORAGE"]["REGION_NAME"],
        )
        object_name = "{}.avro".format(candid)
        s3.upload_fileobj(f, bucket_name, object_name)
        return self.get_object_url(bucket_name, candid)

    def execute(self, message):
        self.logger.debug(message["objectId"])
        f = io.BytesIO(self.consumer.messages[0].value())
        self.upload_file(
            f, message["candidate"]["candid"], self.config["STORAGE"]["BUCKET_NAME"]
        )
