import psycopg2
import pytest
import os

@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


def is_responsive_psql(url):
    try:
        conn = psycopg2.connect(
            f"dbname='postgres' user='postgres' host=localhost password='postgres'"
        )
        conn.close()
        return True
    except:
        return False


@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    """Ensure that Kafka service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("postgres", 5432)
    server = "{}:{}".format(docker_ip, port)
    print("psql", server)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_psql(server)
    )
    return server
