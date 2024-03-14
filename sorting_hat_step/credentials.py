import os
import boto3
from botocore.exceptions import ClientError
import json
from collections import UserDict


# Copy pasted this class from dbplugins
# TODO remove when we can ensure that settings come with the correct format
class _MongoConfig(UserDict):
    """Special dictionary used to parse configuration dictionaries for mongodb.

    The required keys are described in `REQUIRED_KEYS`, but can come with any
    capitalization. It is possible to add extra keys for other parameters used
    by `pymongo.MongoClient`.

    All keys are converted from `snake_case` to `lowerCamelCase` format, as
    used by `pymongo`. The special key `database` is removed from the dictionary
    proper, but can be accessed through the property `db_name`.
    """

    REQUIRED_KEYS = {"host", "username", "password", "port", "database"}

    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        if self.REQUIRED_KEYS.difference(self.keys()):
            missing = ", ".join(
                value.upper() for value in self.REQUIRED_KEYS.difference(self.keys())
            )
            raise ValueError(f"Invalid configuration. Missing keys: {missing}")

    def __setitem__(self, key, value):
        """Converts keys from (case-insensitive) `snake_case` to `lowerCamelCase`"""
        klist = [
            w.lower() if i == 0 else w.title() for i, w in enumerate(key.split("_"))
        ]
        super().__setitem__("".join(klist), value)


def get_secret(secret_name: str) -> str:
    secret_name = secret_name
    region_name = "us-east-1"

    client = boto3.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    return get_secret_value_response["SecretString"]


def get_credentials(secret_name: str, secret_type="mongo") -> UserDict | dict:
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    # check if config is valid
    # _MongoConfig will raise error if the config has missing parameters
    if secret_type == "mongo":
        secret = _MongoConfig(secret)
        secret["port"] = int(secret["port"])
    return secret
