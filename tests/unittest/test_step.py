import unittest
from unittest import mock
from s3_step.step import S3Step, io
from apf.consumers import GenericConsumer


SVY1_TOPIC, SVY1_BUCKET = "svy1_topic", "test_svy1_bucket"
SVY2_TOPIC, SVY2_BUCKET = "svy2_topic", "test_svy2_bucket"


STORAGE_CONFIG = {
    "BUCKET_NAME": f"{SVY1_BUCKET}:svy1,{SVY2_BUCKET}:svy2",
    "AWS_ACCESS_KEY": "fake",
    "AWS_SECRET_ACCESS_KEY": "fake",
    "REGION_NAME": "fake",
}


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "STORAGE": STORAGE_CONFIG,
        }
        mock_consumer = mock.create_autospec(GenericConsumer)
        mock_message = mock.MagicMock()
        mock_message.value.return_value = b"fake"
        mock_message.topic.return_value = SVY1_TOPIC
        mock_consumer.messages = [mock_message]
        self.step = S3Step(config=self.step_config, consumer=mock_consumer)

    def test_get_object_url(self):
        bucket_name = SVY1_BUCKET
        candid = 123
        url = self.step.get_object_url(bucket_name, candid)
        self.assertEqual(url, f"https://{SVY1_BUCKET}.s3.amazonaws.com/321.avro")

    @mock.patch("boto3.client")
    def test_upload_file(self, mock_client):
        f = io.BytesIO(b"fake")
        candid = 123
        self.step.upload_file(f, candid, SVY1_BUCKET)
        mock_client.assert_called_with(
            "s3",
            aws_access_key_id=self.step_config["STORAGE"]["AWS_ACCESS_KEY"],
            aws_secret_access_key=self.step_config["STORAGE"]["AWS_SECRET_ACCESS_KEY"],
            region_name=self.step_config["STORAGE"]["REGION_NAME"]
            )
        mock_client().upload_fileobj.assert_called_with(f, SVY1_BUCKET, "321.avro")

    @mock.patch("s3_step.S3Step.upload_file")
    def test_execute(self, mock_upload):
        message = {"objectId": "obj", "candidate": {"candid": 123}}
        self.step.execute(message)
        mock_upload.assert_called_once()
