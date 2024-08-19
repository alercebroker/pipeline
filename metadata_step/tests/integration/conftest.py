import os
from db_plugins.db.sql._connection import PsqlDatabase
import pytest


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(str(pytestconfig.rootdir), "tests/integration", "docker-compose.yaml")


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_psql_responsive(ip, port):
    config = {
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "localhost",
        "PORT": "5432",
        "DB_NAME": "postgres",
    }
    try:
        conn = PsqlDatabase(config)
        conn.create_db()
        return True
    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    port = docker_services.port_for("postgres", 5432)
    server = f"{docker_ip}:{port}"
    docker_services.wait_until_responsive(timeout=30.0, pause=0.1, check=lambda: is_psql_responsive(server, port))
    return server
