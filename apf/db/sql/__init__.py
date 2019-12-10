from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
Session = sessionmaker()

# TODO return only fields in filter_by + kwargs
def get_or_create(session, model, filter_by=None, **kwargs):
    instance = session.query(model).filter_by(**filter_by).first()
    if instance:
        return instance, False
    else:
        filter_by.update(kwargs)
        instance = model(**filter_by)
        session.add(instance)
        return instance, True


def get_session(db_config):
    psql_config = db_config["PSQL"]
    db_credentials = 'postgresql://{}:{}@{}:{}/{}'.format(
        psql_config["USER"], psql_config["PASSWORD"], psql_config["HOST"], psql_config["PORT"], psql_config["DB_NAME"])
    engine = create_engine(db_credentials)
    Base.metadata.create_all(engine)
    Session.configure(bind=engine)
    return Session()


def add_to_database(session, objects):
    if isinstance(objects, list):
        session.add_all(objects)
    else:
        session.add(objects)
    session.commit()


def bulk_insert(objects, model, session, batch_size=1000):
    batch = []
    insert_stmt = model.__table__.insert()
    for obj in objects:
        if len(batch) > batch_size:
            session.execute(insert_stmt, batch)
            batch.clear()
        batch.append(obj)
    if batch:
        session.execute(insert_stmt, batch)
