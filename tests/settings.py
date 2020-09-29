##################################################
#       early_classifier   Settings File
##################################################

DB_CONFIG = {
    "SQL": {
        "ENGINE": "postgres",
        "HOST": "localhost",
        "USER": "postgres",
        "PASSWORD": "docker",
        "PORT": 5432,
        "DB_NAME": "test"
    }       
}

STEP_CONFIG = {
    "DB_CONFIG": DB_CONFIG,
}

CONSUMER_CONFIG = {
    "DIRECTORY_PATH": "example_avros"
}
