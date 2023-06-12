import io
import logging
import boto3
from apf.core.step import GenericStep


class S3Step(GenericStep):
    """S3Step Description

    Parameters
    ----------
    consumer : GenericConsumer
        Description of parameter `consumer`.
    """

    def __init__(self, consumer=None, config=None, level=logging.INFO, **step_args):
        super().__init__(consumer, config=config, level=level)
        self.s3_client = step_args["s3_client"]

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
        reverse_candid = self.reverse_candid(candid)
        return "https://{}.s3.amazonaws.com/{}.avro".format(bucket_name, reverse_candid)

    def upload_file(self, f, candid, bucket_name):
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
        reverse_candid = self.reverse_candid(candid)
        object_name = "{}.avro".format(reverse_candid)
        self.s3_client.upload_fileobj(f, bucket_name, object_name)
        return self.get_object_url(bucket_name, candid)

    @staticmethod
    def reverse_candid(candid):
        """
        Returns reverse digits of the candid

        Parameters
        ----------
        candid : int or str
            original candid to be reversed
        """
        return str(candid)[::-1]

    def _upload_message(self, message, serialized):
        self.logger.debug(message["objectId"])
        file = io.BytesIO(serialized.value())
        try:
            bucket = self._find_bucket(serialized.topic())
        except KeyError as err:
            self.logger.error(f"{err}")
            raise
        else:
            self.upload_file(file, message["candidate"]["candid"], bucket)

    def _find_bucket(self, topic):
        for key in self.config["STORAGE"]["BUCKET_NAME"]:
            if topic.startswith(key):
                return self.config["STORAGE"]["BUCKET_NAME"][key]
        raise KeyError(
            f"Topic {topic} does not match with set prefixes: {', '.join(self.config['STORAGE']['BUCKET_NAME'])}"
        )

    def execute(self, message):
        try:
            (serialized,) = self.consumer.messages
        except ValueError:
            for msg, serialized in zip(message, self.consumer.messages):
                self._upload_message(msg, serialized)
        else:
            self._upload_message(message, serialized)
