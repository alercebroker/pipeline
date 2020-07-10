##################################################
#       early_classifier   Settings File
##################################################

IP = '0.0.0.0'

DB_CONFIG = {
    "PSQL": {
        "HOST": IP,
        "USER": "postgres",
        "PASSWORD": "docker",
        "PORT": 5432,
        "DB_NAME": "test"
    }       
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
    "n_retry": 5,
    "clf_api": f"http://{IP}:5000/get_classification",
}

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "../some_avros"
}
