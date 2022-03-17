from db_plugins.db.generic import BaseQuery
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.sql import SQLConnection


class MultiQuery(BaseQuery):
    def __init__(self, psql_driver: SQLConnection, mongo_driver: MongoConnection, read_from: str = "engine", model=None, *args, **kwargs):
        self.psql = psql_driver
        self.mongo = mongo_driver
        self.read_from = read_from
        self.engine = "mongo"
        if kwargs["engine"]:
            self.engine = kwargs["engine"]
        pass

    def check_exists(self, model, filter_by):
        """Check if a model exists in the database."""
        raise NotImplementedError()

    def get_or_create(self, model, filter_by, **kwargs):
        pass

    def update(self, instance, args):
        """Update a model instance with specified args."""
        raise NotImplementedError()

    def paginate(self, page=1, per_page=10, count=True):
        """Return a pagination object from this query."""
        raise NotImplementedError()

    def bulk_insert(self, objects, model):
        self.mongo.query().bulk_insert(objects, model)
        # objects will be change
        # model
        self.psql.query().bulk_insert(objects, model)

    def find_all(self):
        """Retrieve all items from the result of this query."""
        if self.engine is "mongo":
            return self.mongo.query().find_all()
        raise NotImplementedError()

    def find_one(self, filter_by={}, model=None, **kwargs):
        """Retrieve only one item from the result of this query.
        Returns None if result is empty.
        """
        raise NotImplementedError()
