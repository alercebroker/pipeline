import pathlib
import unittest
from unittest import mock

from apf.producers import GenericProducer
from tests.data.messages import generate_input_batch

from delight_step import DelightStep 


CONSUMER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "SCHEMA_PATH": "CONSUMER_SCHEMA_PATH",
    "PARAMS": {
      "bootstrap.servers": "SERVER",
      "group.id": "delight",
    },
    "TOPICS": ["CONSUMER"],
    "consume.timeout": 10,
    "consume.messages": 200,
}

# Producer Configuration

PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "TOPIC": "PRODUCER",
    "PARAMS": {
        "bootstrap.servers": "SERVER",
        "message.max.bytes": 6291456,
    },
    "SCHEMA_PATH": "PRODUCER_SCHEMA_PATH",
}




SCRIBE_PRODUCER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "PARAMS": {
        "bootstrap.servers": "SCRIBE_SERVER",
    },
    "TOPIC": "SCRIBE_TOPIC",
    "SCHEMA_PATH": "SCRIBE_SCHEMA_PATH",
}

METRICS_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "EXTRA_METRICS": [
        {"key": "candid"},
        {"key": "oid", "alias": "oid"},
        {"key": "aid", "alias": "aid"},
        {"key": "tid", "format": lambda x: str(x)},
    ],
    "PARAMS": {
        "PARAMS": {
            "bootstrap.servers": "METRICS_HOST",
            "auto.offset.reset": "smallest",
        },
        "TOPIC": "METRICS_TOPIC",
        "SCHEMA_PATH": "METRICS_SCHEMA_PATH",
    },
}


class DelightStepTestCase(unittest.TestCase):
    
    def set_up_step(self, step_config, db_client) -> None:
        return DelightStep(
            config=step_config,
            db_client=db_client
        )

    def get_step_config(self, delight_config):
        step_config = {
            "CONSUMER_CONFIG": CONSUMER_CONFIG,
            "PRODUCER_CONFIG": PRODUCER_CONFIG,
            "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
            "DELIGHT_CONFIG": delight_config,
        }
        return step_config

    def get_get_sql_probabily_mock(self, oid_list):
        mock_db_client = mock.Mock()
        mock_db_client.get_sql_probabily.return_value = oid_list
        return mock_db_client


    @mock.patch('delight_step.step.get_sql_probabily')
    def test_step(self, mock_get_sql_probabily):
        delight_config = {
            "filter": {
                "class": "DELIGHT_FILTER_CLASS",
                "clasifier": "DELIGHT_FILTER_CLASSIFIER",
                "prob": "DELIGHT_FILTER_PROBABILITY",
            },
            "calc_dispersion": False,
            "calc_galaxy_size": False,
        }
        input_messages = generate_input_batch(5)
        input_messages_oids = [
            message["oid"]
            for message in input_messages
        ]
        mock_get_sql_probabily.return_value = input_messages_oids
        step_config = self.get_step_config(delight_config)
        mock_db_client = mock.MagicMock()
        step = self.set_up_step(
            step_config,
            db_client = mock_db_client
        )
        result = step.execute(input_messages)
        assert len(result) == 5
        for res in result:
            assert "galaxy_properties" in res
            assert "ra" in res["galaxy_properties"]
            assert "dec" in res["galaxy_properties"]
            assert "dispersion" in res["galaxy_properties"]
            assert "galaxy_size" in res["galaxy_properties"]

    @mock.patch('delight_step.step.get_sql_probabily')
    def test_step_empty(self, mock_get_sql_probabily):
        delight_config = {
            "filter": {
                "class": "DELIGHT_FILTER_CLASS",
                "clasifier": "DELIGHT_FILTER_CLASSIFIER",
                "prob": "DELIGHT_FILTER_PROBABILITY",
            },
            "calc_dispersion": False,
            "calc_galaxy_size": False,
        }
        input_messages = generate_input_batch(5)
        step_config = self.get_step_config(delight_config)
        mock_db_client = mock.MagicMock()
        mock_get_sql_probabily.return_value = []  # Aquí se pasa una lista vacía
        step = self.set_up_step(
            step_config,
            db_client=mock_db_client
        )
        result = step.execute(input_messages)

        assert len(result) == 5
        for res in result:
            assert "galaxy_properties" in res
            assert res["galaxy_properties"] is None

    @mock.patch('delight_step.step.get_sql_probabily')
    def test_step_partial(self, mock_get_sql_probabily):
        delight_config = {
            "filter": {
                "class": "DELIGHT_FILTER_CLASS",
                "clasifier": "DELIGHT_FILTER_CLASSIFIER",
                "prob": "DELIGHT_FILTER_PROBABILITY",
            },
            "calc_dispersion": False,
            "calc_galaxy_size": False,
        }
        input_messages = generate_input_batch(5)
        input_messages_oids = [
            message["oid"]
            for message in input_messages
        ]
        step_config = self.get_step_config(delight_config)
        mock_db_client = mock.MagicMock()
        mock_get_sql_probabily.return_value = (input_messages_oids[:2])  # Respuesta para solo los primeros 2 oids
        step = self.set_up_step(
            step_config,
            db_client=mock_db_client
        )
        result = step.execute(input_messages)
        assert len(result) == 5
        # si el oid en result[oid] esta en los oids filtrado, el properties tiene cosas, si no, galaxy properties es none
        for res in result:
            if res["oid"] in input_messages_oids[:2]:
                assert "galaxy_properties" in res
                assert res["galaxy_properties"] is not None
            else:
                assert "galaxy_properties" in res
                assert res["galaxy_properties"] is None