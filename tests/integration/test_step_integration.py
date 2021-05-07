import pytest
from watchlist_step.step import WatchlistStep
from unittest import mock
import datetime
from apf.consumers import KafkaConsumer
from db_plugins.db.sql import SQLConnection


@pytest.fixture
def step_creator():
    def create_step(consumer, alerts_db_connection, users_db_connection, config):
        return WatchlistStep(
            consumer=consumer,
            alerts_db_connection=alerts_db_connection,
            users_db_connection=users_db_connection,
            config=config,
        )

    return create_step


class TestStep:
    consumer_config = {
        "TOPICS": ["test"],
        "PARAMS": {
            "bootstrap.servers": "localhost:9094",
            "group.id": "",
            "auto.offset.reset": "beginning",
            "enable.partition.eof": "true",
            "enable.auto.commit": "false",
        },
        "consume.timeout": 5,
        "consume.messages": 2,
    }
    alerts_db_connection = SQLConnection()
    users_db_connection = SQLConnection()
    config = {
        "alert_db_config": {
            "SQL": {
                "ENGINE": "postgresql",
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        },
        "users_db_config": {
            "SQL": {
                "ENGINE": "postgresql",
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "password",
                "PORT": 5433,
                "DB_NAME": "postgres",
            }
        },
    }

    def test_should_insert_matches_if_matches_returned(
        self,
        kafka_service,
        alerts_database,
        users_database,
        step_creator,
    ):
        self.consumer_config["PARAMS"]["group.id"] = "test_should_insert"
        consumer = KafkaConsumer(self.consumer_config)
        step = step_creator(
            consumer,
            self.alerts_db_connection,
            self.users_db_connection,
            self.config,
        )
        step.start()
        matches = self.users_db_connection.session.execute(
            "SELECT * FROM watchlist_match;"
        ).fetchall()
        assert len(matches) == 1
