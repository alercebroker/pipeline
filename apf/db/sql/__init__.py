from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, load_only

Base = declarative_base()
Session = sessionmaker()

"""
    Check if record exists in database.

    :param session: The connection session
    :param model: The class of the model to be instantiated
    :param dict filter_by: attributes used to find object in the database
    :param dict kwargs: attributes used to create the object that are not used in filter_by

    :returns: True if object exists else False

"""
def check_exists(session, model, filter_by=None):
    return session.query(
                session.query(model).filter_by(**filter_by).exists()
            ).scalar()

def get_or_create(session, model, filter_by=None, **kwargs):
    """
    Initializes a model by creating it or getting it from the database if it exists

    :param session: The connection session
    :param model: The class of the model to be instantiated
    :param dict filter_by: attributes used to find object in the database
    :param dict kwargs: attributes used to create the object that are not used in filter_by

    :returns: Tuple with the instanced object and wether it was created or not
    """
    instance = session.query(model).options(load_only(*filter_by.keys())).filter_by(**filter_by).first()
    if instance:
        return instance, False
    else:
        filter_by.update(kwargs)
        instance = model(**filter_by)
        session.add(instance)
        return instance, True


def update(instance, args):
    """
    Updates an object
    :param instance: Object to be updatedoptions(load_only(*filter_by.keys()))
    :param dict args: Attributes updated

    :returns: The updated object instance
    """
    for key in args.keys():
        setattr(instance, key, args[key])
    return instance


def get_session(db_config):
    """
    Gets the database session

    :param dict db_config: Credentials to set up the database connection

    :returns: Session
    """
    psql_config = db_config["PSQL"]
    db_credentials = 'postgresql://{}:{}@{}:{}/{}'.format(
        psql_config["USER"], psql_config["PASSWORD"], psql_config["HOST"], psql_config["PORT"], psql_config["DB_NAME"])
    engine = create_engine(db_credentials)
    Session.configure(bind=engine)
    return Session()


def add_to_database(session, objects):
    """
    Adds objects to the database by adding them to the session.

    :param session: Session object connected to the database
    :param list/model objects: Model instances to be added
    """
    if isinstance(objects, list):
        session.add_all(objects)
    else:
        session.add(objects)
    session.commit()


def bulk_insert(objects, model, session):
    """
    Inserts multiple objects to the database improving performance

    :param list objects: Objects to be added
    :param model: Class of the model to be added
    :param session: Session instance
    """
    session.bulk_insert_mappings(model, objects)
