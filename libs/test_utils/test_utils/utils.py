from db_plugins.db.mongo.initialization import init_mongo_database
from pymongo import MongoClient
import psycopg2
from confluent_kafka.admin import AdminClient, NewTopic


def is_responsive_mongo():
    try:
        client = MongoClient("localhost", 27017)
        client.server_info()  # check connection
    except Exception as e:
        print(e)
        return False
    # Create test test_user and test_db
    db = client.test_db
    db.command(
        "createUser",
        "test_user",
        pwd="test_password",
        roles=["dbOwner", "readWrite"],
    )
    # put credentials to init database (create collections and indexes)
    settings = {
        "HOST": "localhost",
        "USERNAME": "test_user",
        "PASSWORD": "test_password",
        "PORT": 27017,
        "DATABASE": "test_db",
        "AUTH_SOURCE": "test_db",
    }
    init_mongo_database(settings)
    db.list_collections()
    return True


def is_responsive_psql():
    try:
        conn = psycopg2.connect(
            "dbname='postgres' user='postgres' host=localhost password='postgres'"
        )
        conn.close()
        return True
    except Exception:
        return False


def is_responsive_kafka(url):
    client = AdminClient({"bootstrap.servers": url})
    topics = ["test_topic"]
    new_topics = [NewTopic(topic, num_partitions=1) for topic in topics]
    fs = client.create_topics(new_topics)
    for topic, f in fs.items():
        try:
            f.result()
        except Exception as e:
            print(f"Can't create topic {topic}")
            print(e)
            return False
    return True
