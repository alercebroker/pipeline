import json
import os
import pytest
import unittest

from sqlalchemy import text

from mongo_scribe.step import MongoScribe

from apf.producers.kafka import KafkaProducer
from db_plugins.db.sql._connection import PsqlDatabase

DB_CONFIG = {
    "PSQL": {
        "ENGINE": "postgresql",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "PORT": 5432,
        "DB_NAME": "postgres",
    }
}

CONSUMER_CONFIG = {
    "CLASS": "apf.consumers.KafkaConsumer",
    "TOPICS": ["test_topic"],
    "PARAMS": {
        "bootstrap.servers": "localhost:9092",
        "group.id": "command_consumer_2",
        "enable.partition.eof": True,
        "auto.offset.reset": "beginning",
    },
    "NUM_MESSAGES": 2,
    "TIMEOUT": 10,
}

PRODUCER_CONFIG = {
    "TOPIC": "test_topic",
    "PARAMS": {"bootstrap.servers": "localhost:9092"},
    "SCHEMA": {
        "namespace": "db_operation",
        "type": "record",
        "name": "Command",
        "fields": [
            {"name": "payload", "type": "string"},
        ],
    },
}


@pytest.mark.usefixtures("psql_service")
@pytest.mark.usefixtures("kafka_service")
class MongoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = PsqlDatabase(DB_CONFIG["PSQL"])
        step_config = {
            "DB_CONFIG": DB_CONFIG,
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
        }
        cls.db.create_db()
        with cls.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO step(step_id, name, version, comments, date) 
                    VALUES ('v1', 'version 1', '1', '', current_timestamp)
                    """
                )
            )
            session.commit()
        cls.step = MongoScribe(config=step_config, db="sql")
        cls.producer = KafkaProducer(config=PRODUCER_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.db.drop_db()

    def test_insert_objects_into_database(self):
        command = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF02ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        command2 = json.dumps(
            {
                "collection": "object",
                "type": "insert",
                "data": {
                    "oid": "ZTF03ululeea",
                    "ndet": 1,
                    "firstmjd": 50001,
                    "g_r_max": 1.0,
                    "g_r_mean_corr": 0.9,
                    "meanra": 45,
                    "meandec": 45,
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command2})
        self.producer.producer.flush(1)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(text("SELECT * FROM object"))
            oids = [r[0] for r in result]
            assert "ZTF02ululeea" in oids
            assert "ZTF03ululeea" in oids

    def test_insert_detections_into_database(self):
        command = json.dumps(
            {
                "collection": "detection",
                "type": "update",
                "data": {
                    "candid": 932472823,
                    "oid": "ZTF04ululeea",
                    "mjd": 55000,
                    "fid": 1,
                    "pid": 4.3,
                    "isdiffpos": 1,
                    "ra": 99.0,
                    "dec": 55.0,
                    "magpsf": 220.0,
                    "sigmapsf": 33.0,
                    "corrected": True,
                    "dubious": True,
                    "has_stamp": True,
                    "step_id_corr": "steppu",
                },
            }
        )
        self.producer.produce({"payload": command})
        self.producer.produce({"payload": command})

        with self.db.session() as session:
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        self.step.start()

        with self.db.session() as session:
            result = session.execute(
                text(
                    """ SELECT candid, oid FROM detection WHERE oid = 'ZTF04ululeea' """
                )
            )
            result = list(result)[0]
            assert result[0] == 932472823
            assert result[1] == "ZTF04ululeea"

    def test_upsert_detections(self):
        with self.db.session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO feature_version(version, step_id_feature, step_id_preprocess)
                    VALUES ('1.0.0', 'v1', 'v1')
                    """
                )
            )
            session.execute(
                text(
                    """INSERT INTO object(oid, ndet, firstmjd, g_r_max, g_r_mean_corr, meanra, meandec)
                    VALUES ('ZTF04ululeea', 1, 50001, 1.0, 0.9, 45, 45) ON CONFLICT DO NOTHING"""
                )
            )
            session.commit()

        command_data = [
            {
                "oid": "ZTF04ululeea",
                "name": "feature1",
                "value": 55.0,
                "fid": 1,
            },
            {
                "oid": "ZTF04ululeea",
                "name": "feature2",
                "value": 22.0,
                "fid": 2,
            },
            {
                "oid": "ZTF04ululeea",
                "name": "feature3",
                "value": 130.0,
                "fid": 1,
            },
            {
                "oid": "ZTF04ululeea",
                "name": "feature1",
                "value": 694211.0,
                "fid": 1,
            },
        ]

        commands = [
            {
                "payload": json.dumps(
                    {
                        "collection": "object",
                        "type": "update_features",
                        "data": {
                            "features_version": "1.0.0",
                            "features_group": "ztf",
                            "features": [data],
                        },
                    }
                )
            }
            for data in command_data
        ]

        for command in commands:
            self.producer.produce(command)
        self.producer.producer.flush(4)
        self.step.start()
        with self.db.session() as session:
            result = session.execute(
                text("""
                    SELECT name, value FROM feature WHERE oid = 'ZTF04ululeea' AND name = 'feature1'
                """)
            )
            result = list(result)[0]
            assert result[0] == "feature1"
            assert result[1] == 694211.0
