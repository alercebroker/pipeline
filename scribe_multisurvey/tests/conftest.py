import os

import psycopg2
import pytest

PSQL_PORT = 5432


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


@pytest.fixture(scope="session")
def docker_compose_command():
    version = os.getenv("COMPOSE", "v2")
    return "docker compose" if version == "v2" else "docker-compose"


def is_psql_responsive(host, port):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host=host,
            port=port,
        )
        conn.close()
        return True
    except psycopg2.OperationalError:
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    port = docker_services.port_for("postgres", PSQL_PORT)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.5, check=lambda: is_psql_responsive(docker_ip, port)
    )
    return {"host": docker_ip, "port": port}
