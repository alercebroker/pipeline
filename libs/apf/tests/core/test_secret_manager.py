from moto import mock_secretsmanager
import os
from unittest import TestCase
import boto3

os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@mock_secretsmanager
class TestCredentials(TestCase):
    def setUp(self):
        self.client = boto3.client(
            service_name="secretsmanager", region_name="us-east-1"
        )
        self.secret = '{"username":"test","password":"test"}'
        self.client.create_secret(Name="test-secret", SecretString=self.secret)

    def test_db_config(self):
        from apf.core.secret_manager import get_secret

        secret = get_secret("test-secret")
        assert secret == self.secret
