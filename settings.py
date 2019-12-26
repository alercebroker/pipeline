##################################################
#       early_classifier   Settings File
##################################################

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "../avro_files/ztf_public_20190302"
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
STEP_CONFIG = {
        "DB_CONFIG": DB_CONFIG,
        "n_retry": 5,
        "clf_api": "http://localhost:5000/get_classification",
}
