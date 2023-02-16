from unittest.mock import patch
import unittest

from mongo_scribe.db.executor import ScribeCommandExecutor


class ExecutorTest(unittest.TestCase):
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

    @patch("mongo_scribe.db.executor.create_operations")
    @patch("mongo_scribe.db.executor.execute_operations")
    def test_bulk_execute(self, mock_execute, mock_create):
        self.executor.bulk_execute("object", [])
        mock_create.assert_called()
        mock_execute.assert_called()
