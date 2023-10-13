import json
from collections import UserDict
from apf.core.secret_manager import get_secret


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
                value.upper()
                for value in self.REQUIRED_KEYS.difference(self.keys())
            )
            raise ValueError(f"Invalid configuration. Missing keys: {missing}")

    def __setitem__(self, key, value):
        """Converts keys from (case-insensitive) `snake_case` to `lowerCamelCase`"""
        klist = [
            w.lower() if i == 0 else w.title()
            for i, w in enumerate(key.split("_"))
        ]
        super().__setitem__("".join(klist), value)

def get_credentials(secret_name, db_type="mongo"):
    secret = get_secret(secret_name)
    secret = json.loads(secret)
    # check if config is valid
    # _MongoConfig will raise error if the config has missing parameters
    if db_type == "mongo":
        secret = _MongoConfig(secret)
        secret["port"] = int(secret["port"])
    return secret
