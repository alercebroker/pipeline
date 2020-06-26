from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, load_only, scoped_session, Query
from sqlalchemy.ext.declarative import declarative_base
from ..generic import DatabaseConnection, BaseQuery, Pagination


Base = declarative_base()
from db_plugins.db.sql import models


class SQLQuery(BaseQuery, Query):
    def check_exists(self, model=None, filter_by=None):
        """
        Check if record exists in database.

        :param session: The connection session
        :param model: The class of the model to be instantiated
        :param dict filter_by: attributes used to find object in the database
        :param dict kwargs: attributes used to create the object that are not used in filter_by

        :returns: True if object exists else False

        """
        model = self._entities[0].mapper.class_ if self._entities else model
        query = self.session.query(model) if not self._entities else self
        return self.session.query(query.filter_by(**filter_by).exists()).scalar()

    def get_or_create(self, model=None, filter_by=None, **kwargs):
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
        model = self._entities[0].mapper.class_ if self._entities else model
        query = self.session.query(model) if not self._entities else self
        instance = (
            query.options(load_only(*filter_by.keys())).filter_by(**filter_by).first()
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

    def paginate(self, page=1, per_page=10, count=True):
        """
        Returns pagination object with the results

        Parameters
        -----------
        
        page : int
            page or offset of the query
        per_page : int
            number of items per each result page
        count : bool
            whether to count total elements in query
        """
        if page < 1:
            page = 1
        if per_page < 0:
            per_page = 10
        items = self.limit(per_page).offset((page - 1) * per_page).all()
        if not count:
            total = None
        else:
            total = self.order_by(None).count()
        return Pagination(self, page, per_page, total, items)

    def find_one(self, model=None, filter_by=None):
        """
        Finds one item of the specified model. 
        
        If there are more than one item an error occurs.
        If there are no items, then it returns None

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            attributes used to find object in the database
        """
        model = self._entities[0].mapper.class_ if self._entities else model
        query = self.session.query(model) if not self._entities else self
        return query.filter_by(**filter_by).one_or_none()

    def find(self, model=None, filter_by=None, paginate=True):
        """
        Finds list of items of the specified model. 
        
        If there are too many items a timeout can happen.

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            attributes used to find objects in the database
        paginate : bool
            whether to get a paginated result or not
        """
        model = self._entities[0].mapper.class_ if self._entities else model
        query = self.session.query(model) if not self._entities else self
        query = query.filter_by(**filter_by)
        if paginate:
            return query.paginate()
        else:
            return query.all()


class SQLConnection(DatabaseConnection):
    def __init__(self):
        self.config = None
        self.engine = None
        self.Base = None
        self.Session = None
        self.session = None

    def connect(
        self, config, base=None, session_options=None, use_scoped=False, scope_func=None
    ):
        self.config = config
        self.engine = create_engine(config["SQLALCHEMY_DATABASE_URL"])
        self.Base = base or Base
        session_options = session_options or {}
        session_options["query_cls"] = SQLQuery
        self.Session = sessionmaker(bind=self.engine, **session_options)
        if not use_scoped:
            self.create_session()
        else:
            self.create_scoped_session(scope_func)

    def create_session(self):
        self.session = self.Session()

    def create_scoped_session(self, scope_func=None):
        self.session = scoped_session(self.Session, scopefunc=scope_func)
        self.Base.query = self.session.query_property(query_cls=SQLQuery)

    def create_db(self):
        self.Base.metadata.create_all(bind=self.engine)

    def drop_db(self):
        self.Base.metadata.drop_all(bind=self.engine)

    def query(self, *args):
        return self.session.query(*args)
