##################################################
#       early_classifier   Settings File
##################################################

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "../some_avros"
}
DB_CONFIG = {
    "PSQL": {
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "docker",
        "PORT": 5432,
        "DB_NAME": "test"
    }       
}
IP = '18.191.43.15'
STEP_CONFIG = {
        "DB_CONFIG": DB_CONFIG,
        "n_retry": 5,
        "clf_api": f"http://{IP}:5000/get_classification",
}
