import os
import json
from apf.core.secret_manager import get_secret
from db_plugins.db.sql.connection import satisfy_keys


def validate_database_credentials(credentials: dict):
    if len(satisfy_keys(credentials.keys())) > 0:
        raise Exception("Invalid config")

    return True


def get_psql_credentials():
    secret_name = os.environ["PSQL_SECRET_NAME"]
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    # check if config is valid
    # _MongoConfig will raise error if the config has missing parameters
    validate_database_credentials(secret)
    secret["port"] = int(secret["port"])
    return secret
