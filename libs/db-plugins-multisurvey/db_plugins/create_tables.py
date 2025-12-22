from db_plugins.db.sql._connection import PsqlDatabase
from db_plugins.db.sql.models import Feature, FeatureNameLut 

#QUIMAL
# "HOST": 'quimal-db2.alerce.online',
#"PASSWORD": 'IG,eFKHi5['
#"DB_NAME": "ztf",


#STAGING
connection_config = {
    "HOST": '44.210.122.226',
    "USER": 'alerce',
    "PASSWORD": 'AmOeFXXfvzV1rMO',
    "PORT": 5432,
    "DB_NAME": "alercedb",
    "SCHEMA": "multisurvey_api",
}



# connection_config = {
#     "HOST": 'quimal-db2.alerce.online',
#     "USER": 'alerce',
#     "PASSWORD": 'IG,eFKHi5[',
#     "PORT": 5432,
#     "DB_NAME": "ztf",
#     "SCHEMA": "multisurvey",
# }


psql_connection = PsqlDatabase(connection_config)


# para crear las tablas
#psql_connection.create_db()

FeatureNameLut.__table__.create(psql_connection._engine, checkfirst=True)
Feature.__table__.create(psql_connection._engine, checkfirst=True)