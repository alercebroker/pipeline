import pytest
import os


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )



@pytest.fixture(scope="session")
def psql_service(docker_ip, docker_services):
    pass
