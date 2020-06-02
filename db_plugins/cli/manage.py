from db_plugins.db.sql import start_db
import alembic.config
import click
import os, sys


MANAGE_PATH = os.path.dirname(os.path.abspath(__file__))
MIGRATIONS_PATH = os.path.abspath(
    os.path.join(MANAGE_PATH, "../db/sql/"))

@click.group()
def cli():
    pass


@cli.command()
@click.option('--settings_path', default=".", help="settings.py path")
def initdb(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.dirname(os.path.expanduser(settings_path)))
    from settings import DB_CONFIG
    if "PSQL" in DB_CONFIG:
        db_config = DB_CONFIG["PSQL"]
        db_credentials = 'postgresql://{}:{}@{}:{}/{}'.format(
            db_config["USER"], db_config["PASSWORD"], db_config["HOST"], db_config["PORT"], db_config["DB_NAME"])
    else:
        db_credentials = "sqlite:///:memory:"
    start_db(db_credentials)
    os.chdir(MIGRATIONS_PATH)
    alembic.config.main([
        'stamp', 'head'
    ])
    click.echo("Database created with credentials from {}".format(settings_path))


@cli.command()
@click.option('--settings_path', default=".", help="settings.py path")
def make_migrations(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.abspath(settings_path))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = [
        '--raiseerr',
        'revision', '--autogenerate', '-m', 'tables'
    ]
    alembic.config.main(alembicArgs)


@cli.command()
@click.option('--settings_path', default=".", help="settings.py path")
def migrate(settings_path):
    if not os.path.exists(settings_path):
        raise Exception("Settings file not found")
    sys.path.append(os.path.abspath(settings_path))

    os.chdir(MIGRATIONS_PATH)
    alembicArgs = [
        '--raiseerr',
        'upgrade', 'head'
    ]
    alembic.config.main(alembicArgs)