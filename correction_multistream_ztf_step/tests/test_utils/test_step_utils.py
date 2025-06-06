



OUTPUT = {'oids': [36028997823087042, 36028949634494332], 'mids': [3074303351515015000, 3074303351815015012], 'dets': [10,21], 'ndets': [11,3]}

psql_config = {
    "ENGINE": "postgresql",
    "HOST": "localhost",
    "USER": "postgres",
    "PASSWORD": "postgres",
    "PORT": 5432,
    "DB_NAME": "correction_test",
    "SCHEMA": "multisurvey"
}

INSPECT = ['mag', 'e_mag', 'e_mag_corr', 
           "e_mag_corr_ext", "mag_corr", 
           "dubious", "corrected", "stellar"]
