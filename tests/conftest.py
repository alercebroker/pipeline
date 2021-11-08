import pytest
import os
from db_plugins.db.generic import new_DBConnection
from db_plugins.db.mongo.connection import MongoDatabaseCreator
from pymongo import MongoClient


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(
        str(pytestconfig.rootdir), "tests/integration", "docker-compose.yml"
    )


settings = {
    "HOST": "localhost",
    "USER": "root",
    "PASSWORD": "rootpassword",
    "PORT": 27017,
    "DATABASE": "admin",
}


def is_responsive_mongo(url):
    try:
        # driver = new_DBConnection(MongoDatabaseCreator)
        # driver.connect(settings)
        client = MongoClient(
            "localhost",
            27017,
            username="root",
            password="rootpassword",
            authSource="admin",
        )
        db = client.test_db
        db.command(
            "createUser",
            "testo",
            pwd="passu",
            roles=["dbOwner"],
        )
        return True
    except Exception as e:
        print(e)
        return False


@pytest.fixture(scope="session")
def mongo_service(docker_ip, docker_services):
    """Ensure that mongo service is up and responsive."""
    # `port_for` takes a container port and returns the corresponding host port
    port = docker_services.port_for("mongo", 27017)
    server = "{}:{}".format(docker_ip, port)
    print("mongo", server)
    docker_services.wait_until_responsive(
        timeout=30.0, pause=0.1, check=lambda: is_responsive_mongo(server)
    )
    return server
