import pathlib

import pytest

from watchlist_step.step import WatchlistStep

SORTING_HAT_SCHEMA_PATH = pathlib.Path(
    pathlib.Path(__file__).parent.parent.parent.parent,
    "schemas/sorting_hat_step",
    "output.avsc",
)


@pytest.fixture
def step_creator():
    def create_step(strategy_name, config):
        return WatchlistStep(
            config=config,
            strategy_name=strategy_name,
        )

    return create_step


class TestStep:
    consumer_config = {
        "CLASS": "apf.consumers.KafkaSchemalessConsumer",
        "SCHEMA_PATH": SORTING_HAT_SCHEMA_PATH,
        "TOPICS": ["test"],
        "PARAMS": {
            "bootstrap.servers": "localhost:9092",
            "group.id": "test_integration",
            "auto.offset.reset": "beginning",
            "enable.partition.eof": True,
        },
        "consume.timeout": 30,
        "consume.messages": 2,
    }

    config = {
        "CONSUMER_CONFIG": consumer_config,
        "PSQL_CONFIG": {
            "ENGINE": "postgresql",
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "postgres",
            "PORT": 5432,
            "DB_NAME": "postgres",
        },
    }

    def test_should_insert_matches_if_matches_returned(
        self,
        kafka_service,
        users_db,
        step_creator,
    ):
        self.consumer_config["PARAMS"]["group.id"] = "test_integration"
        # consumer = KafkaConsumer(self.consumer_config)
        strategy_name = "SortingHat"
        step = step_creator(
            strategy_name,
            self.config,
        )
        step.start()

        matches = []
        with users_db.conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM watchlist_match WHERE ready_to_notify = true"
                )
                matches = cursor.fetchall()

                cursor.execute("SELECT * FROM watchlist_target")
                targets = cursor.fetchall()

        assert len(matches) == 10
