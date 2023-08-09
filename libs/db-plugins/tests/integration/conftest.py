import pytest
import os
import psycopg2
import pymongo


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


def is_responsive_psql(url):
    try:
        conn = psycopg2.connect(
            "dbname='postgres' user='postgres' host=localhost password='postgres'"
        )
        conn.close()
        return True
    except Exception:
        return False


def is_responsive_mongo(url):
    (host, port) = url.split(":")
    try:
        client = pymongo.MongoClient(
            host=host,  # <-- IP and port go here
            serverSelectionTimeoutMS=3000,  # 3 second timeout
            username="mongo",
            password="mongo",
            port=int(port),
            authSource="database",
        )
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    """Ensure that PSQL service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("postgres", 5432)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_psql(server)
    )
    return server


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("mongo", 27017)
    server = "{}:{}".format(docker_ip, port)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_mongo(server)
    )
    return server
