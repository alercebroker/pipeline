from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
Session = sessionmaker()


def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        instance = model(**kwargs)
        session.add(instance)
        return instance, True


def get_session(db_config):
    psql_config = db_config["PSQL"]
    db_credentials = 'postgresql://{}:{}@{}:{}/{}'.format(
        psql_config["USER"], psql_config["PASSWORD"], psql_config["HOST"], psql_config["PORT"], psql_config["DB_NAME"])
    engine = create_engine(db_credentials)
    Session.configure(bind=engine)
    return Session()
