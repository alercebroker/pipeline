import json


def mock_get_secret(secret_name):
    if secret_name == "sql_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "PORT": 5432,
                "DB_NAME": "postgres",
            }
        )
    elif secret_name == "mongo_secret":
        return json.dumps(
            {
                "HOST": "localhost",
                "USERNAME": "test_user",
                "PASSWORD": "test_password",
                "PORT": 27017,
                "DATABASE": "test_db",
                "AUTH_SOURCE": "test_db",
            }
        )
    else:
        raise ValueError("Unknown secret name")
