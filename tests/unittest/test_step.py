import unittest
from unittest import mock
from s3_step.step import S3Step, SQLConnection, boto3, io
from apf.consumers import GenericConsumer


class StepTestCase(unittest.TestCase):
    def setUp(self):
        STORAGE_CONFIG = {
            "BUCKET_NAME": "fake_bucket",
            "AWS_ACCESS_KEY": "fake",
            "AWS_SECRET_ACCESS_KEY": "fake",
            "REGION_NAME": "fake",
        }
        STEP_METADATA = {
            "STEP_VERSION": "dev",
            "STEP_ID": "s3",
            "STEP_NAME": "s3",
            "STEP_COMMENTS": "s3 upload",
        }
        METRICS_CONFIG = {}
        DB_CONFIG = {"SQL": {}}
        self.step_config = {
            "DB_CONFIG": DB_CONFIG,
            "STORAGE": STORAGE_CONFIG,
            "STEP_METADATA": STEP_METADATA,
            "METRICS_CONFIG": METRICS_CONFIG,
        }
        mock_db = mock.create_autospec(SQLConnection)
        mock_consumer = mock.create_autospec(GenericConsumer)
        mock_message = mock.MagicMock()
        mock_message.value.return_value = b"fake"
        mock_consumer.messages = [mock_message]
        self.step = S3Step(config=self.step_config, db_connection=mock_db, consumer=mock_consumer, test_mode=True)

    def test_get_object_url(self):
        bucket_name = self.step_config["STORAGE"]["BUCKET_NAME"]
        candid = 123
        url = self.step.get_object_url(bucket_name, candid)
        self.assertEqual(url, "https://fake_bucket.s3.amazonaws.com/321.avro")

    @mock.patch("boto3.client")
    def test_upload_file(self, mock_client):
        f = io.BytesIO(b"fake")
        candid = 123
        bucket_name = self.step_config["STORAGE"]["BUCKET_NAME"]
        self.step.upload_file(f, candid, bucket_name)
        mock_client.assert_called_with(
            "s3",
            aws_access_key_id=self.step_config["STORAGE"]["AWS_ACCESS_KEY"],
            aws_secret_access_key=self.step_config["STORAGE"]["AWS_SECRET_ACCESS_KEY"],
            region_name=self.step_config["STORAGE"]["REGION_NAME"]
        )
        mock_client().upload_fileobj.assert_called_with(f,bucket_name, f"{321}.avro")

    @mock.patch("s3_step.S3Step.upload_file")
    def test_execute(self, mock_upload):
        message = {"objectId": "obj", "candidate": {"candid": 123 }}
        self.step.execute(message)
        mock_upload.assert_called_once()


    def test_insert_step_metadata(self):
        self.step.insert_step_metadata()
        self.step.db.query().get_or_create.assert_called_once()
