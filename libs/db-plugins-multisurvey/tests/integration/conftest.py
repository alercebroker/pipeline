import os
import unittest

import psycopg2
import pytest

from db_plugins.db.sql._connection import PsqlDatabase

psql_config = {
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "postgres",
}


@pytest.fixture(scope="session")
def docker_compose_command():
    # compose_version = os.getenv("COMPOSE_VERSION", "v1")
    # if compose_version == "v1":
    #     return "docker-compose"
    return "docker compose"


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    try:
        print(str(pytestconfig.rootdir))
        path = os.path.join(
            str(pytestconfig.rootdir), "tests", "integration", "docker-compose.yml"
        )
        assert os.path.exists(path)
    except Exception:
        path = os.path.join(
            str(pytestconfig.rootdir),
            "libs",
            "db-plugins",
            "tests",
            "integration",
            "docker-compose.yml",
        )

    return path


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
def psql_db(docker_ip: str, docker_services):
    port = docker_services.port_for("postgres", 5432)
    docker_services.wait_until_responsive(
        timeout=30.0,
        pause=0.1,
        check=lambda: is_responsive_psql(docker_ip, port),
    )

    return PsqlDatabase(psql_config)
