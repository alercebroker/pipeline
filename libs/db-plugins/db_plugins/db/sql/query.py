from sqlalchemy.orm import Query
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from ..generic import BaseQuery, Pagination


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
        if isinstance(filter_by, dict):
            query = self.session.query(model).filter_by(**filter_by)
        else:
            query = self.session.query(model).filter(filter_by)
        result = query.first()
        created = False
        if result is not None:
            return result, created

        try:
            kwargs.update(filter_by)
            instance = model(**kwargs)
            self.session.add(instance)
            self.session.commit()
            result = instance
            created = True
        except IntegrityError:
            self.session.rollback()
            result = self.session.query(model).filter_by(**filter_by).first()
            created = False

        return result, created

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

    def bulk_insert(self, objects, model=None):
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

        if objects is None or len(objects) == 0:
            return

        model = self._entities[0].mapper.class_ if self._entities else model
        table = model.__table__
        insert_stmt = insert(table)
        engine = self.session.get_bind()
        columns = table.primary_key.columns
        names = [c.name for c in columns.values()]
        do_nothing_stmt = insert_stmt.on_conflict_do_nothing(index_elements=names)
        engine.execute(do_nothing_stmt, objects)

    def paginate(self, page=1, per_page=10, count=True, max_results=50000):
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
            total = self.order_by(None).limit(max_results + 1).count()
        return Pagination(self, page, per_page, total, items)

    def find(self, model=None, filter_by={}, paginate=True):
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
        if isinstance(filter_by, dict):
            query = query.filter_by(**filter_by)
        else:
            query = query.filter(filter_by)
        if paginate:
            return query.paginate()
        else:
            return query.all()

    def find_one(self, model=None, filter_by={}):
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
        if isinstance(filter_by, dict):
            query = query.filter_by(**filter_by)
        else:
            query = query.filter(filter_by)
        return query.one_or_none()

    def find_all(self, model=None, filter_by={}, paginate=True):
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
        if isinstance(filter_by, dict):
            query = query.filter_by(**filter_by)
        else:
            query = query.filter(filter_by)
        if paginate:
            return query.paginate()
        else:
            return query.all()
