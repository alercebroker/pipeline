from db_plugins.db.sql.connection import SQLConnection, satisfy_keys, settings_map
import alembic.config
import click
import sys
import os


MANAGE_PATH = os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_PATH = os.path.abspath(os.path.join(MANAGE_PATH, "../db/sql/"))


def init_sql(config, db=None):
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


@click.group()
def cli():
    pass


@cli.command()
@click.option("--settings_path", default=".", help="settings.py path")
def initdb(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.dirname(os.path.expanduser(settings_path)))
    from settings import DB_CONFIG

    db_config = DB_CONFIG["SQL"]

    if "SQL" in DB_CONFIG:
        required_key = satisfy_keys(set(db_config.keys()))
        if len(required_key) == 0:
            db_config["SQLALCHEMY_DATABASE_URL"] = settings_map(db_config)

        if "SQLALCHEMY_DATABASE_URL" in db_config:
            init_sql(db_config)
        else:
            raise Exception(
                f"Missing arguments 'SQLALCHEMY_DATABASE_URL' or {required_key}"
            )

        click.echo("Database created with credentials from {}".format(settings_path))
    elif "MONGO" in DB_CONFIG:
        pass

    else:
        raise Exception("Invalid settings file")


@cli.command()
@click.option("--settings_path", default=".", help="settings.py path")
def make_migrations(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.abspath(settings_path))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "revision", "--autogenerate", "-m", "tables"]
    alembic.config.main(alembicArgs)


@cli.command()
@click.option("--settings_path", default=".", help="settings.py path")
def migrate(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.abspath(settings_path))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    alembic.config.main(alembicArgs)
