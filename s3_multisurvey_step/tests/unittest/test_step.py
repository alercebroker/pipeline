import unittest
from unittest import mock
from s3_step.step import S3Step, io
from apf.consumers import GenericConsumer


SVY1_TOPIC, SVY1_BUCKET = "svy1_topic", "test_svy1_bucket"
SVY2_TOPIC, SVY2_BUCKET = "svy2_topic", "test_svy2_bucket"


STORAGE_CONFIG = {
    "BUCKET_NAME": {"svy1": SVY1_BUCKET, "svy2": SVY2_BUCKET},
    "REGION_NAME": "fake",
}


class StepTestCase(unittest.TestCase):
    def setUp(self):
        self.step_config = {
            "STORAGE": STORAGE_CONFIG,
        }
        self.mock_consumer = mock.create_autospec(GenericConsumer)
        self.mock_message = mock.MagicMock()
        self.mock_message.value.return_value = b"fake"
        self.mock_message.topic.return_value = SVY1_TOPIC
        self.mock_consumer.messages = [self.mock_message]
        self.s3_mock = mock.MagicMock()
        self.step = S3Step(
            config=self.step_config, consumer=self.mock_consumer, s3_client=self.s3_mock
        )

    def test_get_object_url(self):
        candid = 123
        url = self.step.get_object_url(SVY1_BUCKET, candid)
        self.assertEqual(url, f"https://{SVY1_BUCKET}.s3.amazonaws.com/321.avro")

    def test_upload_file(self):
        f = io.BytesIO(b"fake")
        candid = 123
        self.step.upload_file(f, candid, SVY1_BUCKET)
        self.s3_mock.upload_fileobj.assert_called_with(f, SVY1_BUCKET, "321.avro")

    @mock.patch("s3_step.S3Step.upload_file")
    def test_execute(self, mock_upload):
        message = {"objectId": "obj", "candidate": {"candid": 123}}
        self.step.execute(message)
        mock_upload.assert_called_once()

    @mock.patch("s3_step.S3Step.upload_file")
    def test_execute_with_unknown_topic_for_bucket(self, mock_upload):
        message = {"objectId": "obj", "candidate": {"candid": 123}}
        self.mock_message.topic.return_value = "svy3_topic"
        self.mock_consumer.messages = [self.mock_message]
        with self.assertRaisesRegex(KeyError, "svy3_topic"):
            self.step.execute(message)
        mock_upload.assert_not_called()

    @mock.patch("s3_step.S3Step.upload_file")
    def test_execute_with_message_list(self, mock_upload):
        message = {"objectId": "obj", "candidate": {"candid": 123}}
        self.mock_consumer.messages = [self.mock_message, self.mock_message]
        self.step.execute([message, message])
        self.assertEqual(2, mock_upload.call_count)
