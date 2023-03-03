import os
import json

from apf.core.secret_manager import get_secret


def get_mongodb_credentials():
    secret_name = os.environ["MONGODB_SECRET_NAME"]
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    return secret
