import pytest
from apf.consumers import KafkaConsumer

from watchlist_step.step import WatchlistStep


@pytest.fixture
def step_creator():
    def create_step(consumer, strategy_name, config):
        return WatchlistStep(
            consumer=consumer,
            config=config,
            strategy_name=strategy_name,
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
    config = {
        "PSQL_CONFIG": {
            "ENGINE": "postgresql",
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "password",
            "PORT": 5433,
            "DB_NAME": "postgres",
        }
    }

    def test_should_insert_matches_if_matches_returned(
        self,
        kafka_service,
        users_db,
        step_creator,
    ):
        self.consumer_config["PARAMS"]["group.id"] = "test_should_insert"
        consumer = KafkaConsumer(self.consumer_config)
        strategy_name = "SortingHat"
        step = step_creator(
            consumer,
            strategy_name,
            self.config,
        )
        step.start()

        matches = []
        with users_db.conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM watchlist_match")
                matches = cursor.fetchall()

                cursor.execute("SELECT * FROM watchlist_target")
        assert len(matches) == 1
