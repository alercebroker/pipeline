##################################################
#       early_classifier   Settings File
##################################################

IP = '18.191.43.15'

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
    "TOPICS": ['ztf_20200618_programid1'],
    "PARAMS": {
        "bootstrap.servers": "3.94.222.35:9092,3.211.25.175:9092,52.45.218.185:9092",
        "group.id": "early_staging_old_model",
    }
}
