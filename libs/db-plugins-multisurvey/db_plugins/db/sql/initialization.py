from ._connection import PsqlDatabase, get_db_url
import alembic.config
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_PATH = os.path.abspath(os.path.join(THIS_PATH, "../sql/"))


def init_sql_database(config, db=None):
    db = db or PsqlDatabase(config)
    db.create_db()
    os.chdir(MIGRATIONS_PATH)
    try:
        os.makedirs(os.getcwd() + "/migrations/versions")
    except FileExistsError:
        pass
    alembic.config.main(["stamp", "head"])


def init(db_config, db=None):
    db_config["SQLALCHEMY_DATABASE_URL"] = get_db_url(db_config)
    init_sql_database(db_config, db=db)


def make_migrations():
    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "revision", "--autogenerate", "-m", "tables"]
    alembic.config.main(alembicArgs)


def migrate():
    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    alembic.config.main(alembicArgs)
