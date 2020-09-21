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
        self.db.query(Step).get_or_create(
            filter_by={"step_id": self.config["STEP_METADATA"]["STEP_VERSION"]},
            name=self.config["STEP_METADATA"]["STEP_NAME"],
            version=self.config["STEP_METADATA"]["FEATURE_VERSION"],
            comments=self.config["STEP_METADATA"]["STEP_COMMENTS"],
            date=datetime.datetime.now(),
        )

    def get_object_url(self, bucket_name, candid):
        return "https://{}.s3.amazonaws.com/{}.avro".format(bucket_name, candid)

    def upload_file(self, f, candid, bucket_name):
        s3 = boto3.client(
            "s3",
            aws_access_key_id=self.config["STORAGE"]["AWS_ACCESS_KEY"],
            aws_secret_access_key=self.config["STORAGE"]["AWS_SECRET_ACCESS_KEY"],
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
