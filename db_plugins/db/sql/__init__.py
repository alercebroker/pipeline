from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, load_only, scoped_session


class DatabaseConnection:
    def __init__(self, db_credentials=None, base=None):
        self.session = None
        self.engine = create_engine(db_credentials) if db_credentials else None
        self.base = base

    def init_app(self, db_credentials, base):
        self.engine = create_engine(db_credentials)
        self.base = base

    def create_session(self, bind=None, **options):
        if bind:
            self.session = sessionmaker(bind=bind, **options)()
        else:
            self.session = sessionmaker(bind=self.engine, **options)()
        return self.session

    def create_scoped_session(self, bind=None):
        if bind:
            self.session = scoped_session(
                sessionmaker(bind=bind, autocommit=False, autoflush=False)
            )
        else:
            self.session = scoped_session(
                sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
            )
        return self.session

    def create_db(self):
        self.base.metadata.create_all(bind=self.engine)

    def drop_db(self):
        self.base.metadata.drop_all(bind=self.engine)

    def cleanup(self, resp_or_exc):
        self.session.remove()

    def query(self, models, offset=None, limit=None, total=None, sort_by=None, sort_desc="DESC", *params):
        """
        offset = None
        limit = None
        if page and page_size:
            offset = page_size * (page - 1)
            limit = page_size + offset
        sql_query = self.session.query(*models).filter(*params)
        if not total:
            total = sql_query.order_by(None).count()
        if sort_by is not None:
            if sort_desc == "DESC":
                sql_query = sql_query.order_by(sort_by.desc())
            else:
                sql_query = sql_query.order_by(sort_by.asc())
        results = sql_query[offset:limit]
        return {
            "total": total,
            "results": results
        }
        """
        sql_query = self.session.query(*models).filter(*params)
        total = sql_query.order_by(None).count() if not total else None
        if sort_by is not None:
            sql_query = sql_query.order_by(sort_by.desc()) if sort_desc == "DESC" else sql_query.order_by(sort_by.asc())
        results = sql_query[offset:limit]
        return {
            "total": total,
            "results": results
        }

    def check_exists(self, model, filter_by):
        """
        Check if record exists in database.

        :param session: The connection session
        :param model: The class of the model to be instantiated
        :param dict filter_by: attributes used to find object in the database
        :param dict kwargs: attributes used to create the object that are not used in filter_by

        :returns: True if object exists else False

        """
        return self.session.query(
            self.session.query(model).filter_by(**filter_by).exists()
        ).scalar()

    def get_or_create(self, model, filter_by=None, **kwargs):
        """
        Initializes a model by creating it or getting it from the database if it exists

        Parameters
        ----------

        session : Session
            The connection session
        model : Model
            The class of the model to be instantiated
        filter_by : dict
            attributes used to find object in the database
        kwargs : dict
            attributes used to create the object that are not used in filter_by

        Returns
        ----------
        instance, created
            Tuple with the instanced object and wether it was created or not
        """
        instance = (
            self.session.query(model)
            .options(load_only(*filter_by.keys()))
            .filter_by(**filter_by)
            .first()
        )
        if instance:
            return instance, False
        else:
            filter_by.update(kwargs)
            instance = model(**filter_by)
            self.session.add(instance)
            return instance, True

    def update(self, instance, args):
        """
        Updates an object

        Parameter
        -----------

        instance : Model
            Object to be updated
        args : dict
            Attributes updated

        Returns
        ----------
        instance
            The updated object instance
        """
        for key in args.keys():
            setattr(instance, key, args[key])
        return instance

    def bulk_insert(self, objects, model):
        """
        Inserts multiple objects to the database improving performance

        Parameters
        -----------

        objects : list
            Objects to be added
        model: Model
            Class of the model to be added
        session: Session
            Session instance 
        """
        self.session.bulk_insert_mappings(model, objects)
