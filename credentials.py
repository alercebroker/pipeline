import json

from apf.core.secret_manager import get_secret


def get_mongodb_credentials(mongo_secret_name):
    secret_name = mongo_secret_name
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    secret["port"] = int(secret["port"])
    return secret
