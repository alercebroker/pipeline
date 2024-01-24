import unittest
from unittest import mock

from mongo_scribe.mongo.db.executor import ScribeCommandExecutor
from mongo_scribe.mongo.command.commands import InsertCommand


class TestExecutor(unittest.TestCase):
    def setUp(self):
        db_config = {
            "MONGO": {
                "database": "test",
                "port": 27017,
                "host": "localhost",
                "username": "user",
                "password": "pass",
            }
        }
        executor = ScribeCommandExecutor(db_config)
        self.executor = executor
        self.executor.connection = mock.MagicMock()

    def test_bulk_execute_skips_empty(self):
        self.executor.bulk_execute([])
        self.executor.connection.database.__getitem__.return_value.bulk_write.assert_not_called()

    def test_bulk_execute_runs_bulk_write(self):
        # Manual mocking needed to make this work
        command = InsertCommand("object", {"data": "data"}, {})
        command.get_operations = lambda: [mock.MagicMock(), mock.MagicMock()]
        self.executor.bulk_execute([command])
        self.executor.connection.database.__getitem__.return_value.bulk_write.assert_called_once()
