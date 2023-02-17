import os
import json
from apf.core.secret_manager import get_secret
from db_plugins.db.mongo.connection import _MongoConfig

def get_mongodb_credentials():
    secret_name = os.environ["MONGODB_SECRET_NAME"]
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    # check if config is valid
    # _MongoConfig will raise error if the config has missing parameters
    _MongoConfig(secret)
    secret["port"] = int(secret["port"])
    return secret