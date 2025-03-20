import unittest
from unittest import mock
from db_plugins.cli import manage
from click.testing import CliRunner


class TestManage(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.settings_path = "/tmp"

    @mock.patch("db_plugins.db.sql._connection.PsqlDatabase", autospec=True)
    @mock.patch("os.makedirs")
    @mock.patch("db_plugins.db.sql.initialization.alembic.config.main")
    def test_init_sql(self, main_mock, mock_makedirs, mock_connection):
        session = mock.MagicMock()
        mock_connection.session = session
        manage.init_sql(
            {
                "USER": "nada",
                "PASSWORD": "nada",
                "HOST": "nada",
                "PORT": 5432,
                "DB_NAME": "nada",
            },
            db=mock_connection,
        )
        main_mock.assert_called()

    @mock.patch("db_plugins.cli.manage.init_sql")
    def test_initdb_sql(self, mock_init_sql):
        with self.runner.isolated_filesystem(temp_dir=self.settings_path) as td:
            with open("settings.py", "w") as f:
                f.write("DB_CONFIG=")
                DB_CONFIG = {
                    "SQL": {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"},
                }
                f.write(str(DB_CONFIG))

            result = self.runner.invoke(
                manage.initdb,
                ["--settings_path", td],
            )
            assert result.exit_code == 0
            mock_init_sql.assert_called()
            assert (
                "Database created with credentials from {}".format(td) in result.output
            )

    def test_initdb_error(self):
        result = self.runner.invoke(manage.initdb, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)

    @mock.patch("db_plugins.db.sql.initialization.alembic.config.main")
    def test_make_migrations(self, main_mock):
        with self.runner.isolated_filesystem(temp_dir=self.settings_path) as td:
            with open("settings.py", "w") as f:
                f.write("DB_CONFIG=")
                DB_CONFIG = {
                    "SQL": {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"},
                }
                f.write(str(DB_CONFIG))
            result = self.runner.invoke(manage.make_migrations, ["--settings_path", td])
            assert result.exit_code == 0
            main_mock.assert_called()

    def test_make_migrations_error(self):
        result = self.runner.invoke(manage.make_migrations, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)

    @mock.patch("db_plugins.db.sql.initialization.alembic.config.main")
    def test_migrate(self, main_mock):
        with self.runner.isolated_filesystem(temp_dir=self.settings_path) as td:
            with open("settings.py", "w") as f:
                f.write("DB_CONFIG=")
                DB_CONFIG = {
                    "SQL": {"SQLALCHEMY_DATABASE_URL": "sqlite:///:memory:"},
                }
                f.write(str(DB_CONFIG))
            result = self.runner.invoke(manage.migrate, ["--settings_path", td])
            assert result.exit_code == 0
            main_mock.assert_called()

    def test_migrate_error(self):
        result = self.runner.invoke(manage.migrate, "--settings_path fail")
        assert result.exit_code != 0
        assert "Settings file not found" == str(result.exception)

    @mock.patch("db_plugins.db.mongo._connection.MongoConnection", autospec=True)
    def test_init_mongo(self, mock_connection):
        # smoke test
        # TODO make actual test
        manage.init_mongo(
            {
                "USER": "",
                "HOST": "",
                "PASSWORD": "",
                "PORT": "",
                "DATABASE": "",
            },
            db=mock_connection,
        )

    @mock.patch("db_plugins.cli.manage.init_mongo")
    def test_initdb_mongo(self, mock_init_mongo):
        with self.runner.isolated_filesystem(temp_dir=self.settings_path) as td:
            with open("settings.py", "w") as f:
                f.write("DB_CONFIG=")
                DB_CONFIG = {
                    "MONGO": {
                        "USER": "",
                        "PASSWORD": "",
                        "HOST": "",
                        "DATABASE": "",
                        "PORT": 123,
                    },
                }
                f.write(str(DB_CONFIG))
            result = self.runner.invoke(manage.initdb, ["--settings_path", td])
            assert result.exit_code == 0
            mock_init_mongo.assert_called()
            assert (
                "Database created with credentials from {}".format(td) in result.output
            )
