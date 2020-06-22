import unittest
from db_plugins.cli import manage
from click.testing import CliRunner
import os
import alembic.config
from unittest.mock import MagicMock



class TestManage(unittest.TestCase):
    runner = CliRunner()
    settings_path = "."

    def test_initdb(self):
        alembic.config.main = MagicMock()
        result = self.runner.invoke(manage.initdb)
        assert result.exit_code == 0
        alembic.config.main.assert_called()
        assert (
            "Database created with credentials from {}".format(self.settings_path)
            in result.output
        )

    def test_initdb_error(self):
        result = self.runner.invoke(manage.initdb, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)

    def test_make_migrations(self):
        alembic.config.main = MagicMock()
        self.runner.invoke(manage.make_migrations)
        alembic.config.main.assert_called()

    def test_make_migrations_error(self):
        result = self.runner.invoke(manage.make_migrations, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)

    def test_migrate(self):
        alembic.config.main = MagicMock()
        self.runner.invoke(manage.migrate)
        alembic.config.main.assert_called()

    def test_migrate_error(self):
        result = self.runner.invoke(manage.migrate, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)