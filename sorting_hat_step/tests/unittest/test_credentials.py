from moto import mock_secretsmanager
import os
from unittest import TestCase
import boto3

os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@mock_secretsmanager
class TestCredentials(TestCase):
    def setUp(self):
        self.client = boto3.client(
            service_name="secretsmanager", region_name="us-east-1"
        )
        self.mongo_secret = '{"username":"test","password":"test", "host": "test", "database": "test", "port": 1}'
        self.client.create_secret(Name="test-secret", SecretString=self.mongo_secret)
        self.psql_secret = '{"username":"postgres","password":"postgres", "host": "localhost", "database": "postgres", "port": 5432}'
        self.client.create_secret(Name="psql-secret", SecretString=self.psql_secret)

    def test_db_config(self):
        from credentials import get_secret

        secret = get_secret("test-secret")
        assert secret == self.mongo_secret

    def test_get_mongodb_credentials(self):
        from credentials import get_credentials

        result = get_credentials("test-secret")
        assert result == {
            "username": "test",
            "password": "test",
            "host": "test",
            "database": "test",
            "port": 1,
        }

    def test_settings_with_psql(self):
        os.environ["USE_PSQL"] = "True"
        os.environ["MONGODB_SECRET_NAME"] = "test-secret"
        os.environ["PSQL_SECRET_NAME"] = "psql-secret"
        os.environ["CONSUMER_GROUP_ID"] = "test-secret"
        os.environ["CONSUMER_SERVER"] = "test-secret"
        os.environ["CONSUMER_TOPICS"] = "test-secret"
        os.environ["PRODUCER_TOPIC"] = "test-secret"
        os.environ["PRODUCER_SERVER"] = "test-secret"
        os.environ["METRICS_HOST"] = "test-secret"
        os.environ["SCRIBE_PRODUCER_SERVER"] = "test-secret"
        os.environ["SCRIBE_PRODUCER_TOPIC"] = "test-secret"
        from settings import PSQL_CONFIG, MONGO_CONFIG

        assert PSQL_CONFIG == {
            "username": "postgres",
            "password": "postgres",
            "host": "localhost",
            "database": "postgres",
            "port": 5432,
        }

        assert MONGO_CONFIG == {
            "username": "test",
            "password": "test",
            "host": "test",
            "database": "test",
            "port": 1,
        }
