import unittest
from unittest import mock

from mongo_scribe.db.executor import ScribeCommandExecutor


class TestExecutor(unittest.TestCase):
    def setUp(self):
        db_config = {
            "MONGO": {
                "DATABASE": "test",
                "PORT": 27017,
                "HOST": "localhost",
                "USERNAME": "user",
                "PASSWORD": "pass",
            }
        }
        executor = ScribeCommandExecutor(db_config)
        self.executor = executor
        self.executor.connection = mock.MagicMock()

    def test_bulk_execute_skips_empty(self):
        self.executor.bulk_execute("object", [])
        self.executor.connection.database.__getitem__.return_value.bulk_write.assert_not_called()

    def test_bulk_execute_runs_bulk_write(self):
        command = mock.MagicMock()
        command.get_operations.return_value = [mock.MagicMock(), mock.MagicMock()]
        self.executor.bulk_execute("object", [command])
        self.executor.connection.database.__getitem__.return_value.bulk_write.assert_called_once()
