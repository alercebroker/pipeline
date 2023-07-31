from db_plugins.db.sql.connection import (
    SQLConnection,
    satisfy_keys,
    settings_map,
)
import alembic.config
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_PATH = os.path.abspath(os.path.join(THIS_PATH, "../sql/"))


def init_sql_database(config, db=None):
    db = db or SQLConnection()
    db.connect(config=config)
    db.create_db()
    db.session.close()
    os.chdir(MIGRATIONS_PATH)
    try:
        os.makedirs(os.getcwd() + "/migrations/versions")
    except FileExistsError:
        pass
    alembic.config.main(["stamp", "head"])


def init(db_config, db=None):
    required_key = satisfy_keys(set(db_config.keys()))
    if len(required_key) == 0:
        db_config["SQLALCHEMY_DATABASE_URL"] = settings_map(db_config)

    if "SQLALCHEMY_DATABASE_URL" in db_config:
        init_sql_database(db_config, db=db)
    else:
        raise Exception(
            f"Missing arguments 'SQLALCHEMY_DATABASE_URL' or {required_key}"
        )


def make_migrations():
    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "revision", "--autogenerate", "-m", "tables"]
    alembic.config.main(alembicArgs)


def migrate():
    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    alembic.config.main(alembicArgs)
