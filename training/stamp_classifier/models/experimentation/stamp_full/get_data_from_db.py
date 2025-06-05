import json
import requests
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import inspect

def get_database_engine(environment: str, read=True):
    if read:
        if environment == 'production':
            url = 'https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json'
            params = requests.get(url).json()['params']
            engine = sa.create_engine(
                f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}"
            )
        
        elif environment == 'staging':
            with open('alerceread_db_staging.json', 'r', encoding='utf-8') as f:
                params = json.load(f)
            engine = sa.create_engine(
                f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
            )
        else:
            raise ValueError(f'Environment "{environment}" not defined')
    else:
        if environment == 'production':
            #with open('alerce_db_production.json', 'r', encoding='utf-8') as f:
            #    params = json.load(f)
            engine = sa.create_engine(
                f"postgresql+psycopg2://alerce:yR66ZK5tQJGj9qBeKSpXhTMS@10.1.24.217/alercedb"
            )
        
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print('Available Tables:\n', tables)
    return engine


def get_features(oids, engine, version=None):
    if version is not None:
        query_features = f"""
                        SELECT * FROM feature as f 
                        WHERE f.oid in ({','.join(oids)}) and f.version = '{version}';
                        """
    else:
        query_features = f"""
                        SELECT * FROM feature as f 
                        WHERE f.oid in ({','.join(oids)});
                        """       

    features = pd.read_sql_query(query_features, con=engine)
    return features


def get_detections(oids, engine, rb_filter=True):
    if rb_filter:
        query_detections = f"""
            SELECT * FROM detection
            WHERE oid IN ({','.join(oids)}) AND rb >= 0.55;
        """
    else:
        query_detections = f"""
            SELECT * FROM detection
            WHERE oid IN ({','.join(oids)});
        """    
    detections = pd.read_sql_query(query_detections, con=engine)
    return detections


def get_forced_photometry(oids, engine, procstatus_filter=True):
    if procstatus_filter:
        query_forced_photometry = f"""
            SELECT * FROM forced_photometry
            WHERE oid in ({','.join(oids)}) and procstatus in ('0', '57');
        """
    else:
        query_forced_photometry = f"""
            SELECT * FROM forced_photometry
            WHERE oid IN ({','.join(oids)});
        """    
    forced_photometry = pd.read_sql_query(query_forced_photometry, con=engine)
    return forced_photometry