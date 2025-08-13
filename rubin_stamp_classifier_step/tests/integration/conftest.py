import os

import psycopg2
import pytest
from db_plugins.db.sql._connection import PsqlDatabase

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
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


def is_responsive_psql(host, port):
    try:
        conn = psycopg2.connect(
            f"dbname='postgres' user='postgres' host={host} port={port} password='postgres'"
        )
        conn.close()
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    """Ensure that PSQL service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("postgres", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_psql(docker_ip, port),
    )


@pytest.fixture(scope="session")
def psql_db(docker_ip, docker_services):
    port = docker_services.port_for("postgres", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_psql(docker_ip, port),
    )

    psql_db = PsqlDatabase(psql_config)
    psql_db.create_db()

    yield psql_db

    psql_db.drop_db()
