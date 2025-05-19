import json
from collections import UserDict
from apf.core.secret_manager import get_secret


def get_credentials(secret_name):
    secret = get_secret(secret_name)
    secret = json.loads(secret)

    return secret
