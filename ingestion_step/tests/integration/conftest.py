import os

import psycopg2
import pytest
from db_plugins.db.sql._connection import PsqlDatabase
from sqlalchemy import text

# Direct connection to PostgreSQL — used for DDL (create_db/drop_db) and ALTER ROLE.
psql_config_direct = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}

# PgBouncer connection — used for all test operations (transaction-mode pool, NullPool on client).
psql_config_pgbouncer = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5433,
    "DB_NAME": "postgres",
    "POOLCLASS": "NullPool",
}


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_responsive_psql(host: str, port: int):
    try:
        conn = psycopg2.connect(
            f"dbname='postgres' user='postgres' host={host} port={port} password='postgres'"
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip: str, docker_services):
    """Ensure that the PostgreSQL service is up and responsive."""
    port = docker_services.port_for("postgres", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_psql(docker_ip, port),
    )


@pytest.fixture(scope="session")
def pgbouncer_service(psql_service, docker_ip: str, docker_services):
    """Ensure PgBouncer is up, then set the role's search_path so it survives transaction-mode pooling."""
    port = docker_services.port_for("pgbouncer", 5432)
    docker_services.wait_until_responsive(
        timeout=60.0,
        pause=0.5,
        check=lambda: is_responsive_psql(docker_ip, port),
    )
    # Set search_path at the role level so it persists across pgbouncer connections.
    # This replaces the per-connection SET that pgbouncer transaction mode drops.
    conn = psycopg2.connect(
        "dbname='postgres' user='postgres' host=localhost port=5432 password='postgres'"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("ALTER ROLE postgres SET search_path = public")
    cur.close()
    conn.close()


@pytest.fixture(scope="session")
def psql_db(psql_service, pgbouncer_service):
    # Create schema via direct connection.
    direct_db = PsqlDatabase(psql_config_direct)
    direct_db.create_db()

    # Yield a pgbouncer-connected instance for the tests.
    pgb_db = PsqlDatabase(psql_config_pgbouncer)
    yield pgb_db

    # Teardown via direct connection.
    direct_db.drop_db()

