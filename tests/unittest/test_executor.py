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

    def test_bulk_execute(self):
        self.executor.bulk_execute("object", [])
