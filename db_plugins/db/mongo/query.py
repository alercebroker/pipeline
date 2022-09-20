from itertools import zip_longest
from typing import Type

from pymongo import UpdateOne

from db_plugins.db.generic import BaseQuery, Pagination, PaginationNoCount
from db_plugins.db.mongo.models import BaseModel


class CollectionNotFound(Exception):
    pass


class MongoQuery(BaseQuery):
    def __init__(self, database, model: Type[BaseModel] = None, name: str = None):
        if model and name:
            raise ValueError("Only one of 'model' or 'name' can be defined")
        self.model = model
        self.collection = None
        self._db = database
        if name:
            # Ignore model and use pure pymongo API
            self.collection = self._db[name]
        elif model:
            # Using custom ORM API
            self.init_collection(model)

    def init_collection(self, model):
        if model:
            self.model = model
            self.collection = self._db[model._meta.tablename]
        elif self.collection is None:
            raise CollectionNotFound("A valid model must be provided at instantiation or in method call")

    def check_exists(self, filter_by: dict = None, model: Type[BaseModel] = None):
        """
        Check if record exists in database.

        Parameters
        ----------
        filter_by : dict
            Attributes for finding a document in database
        model : Type[BaseModel]
            Class of the model whose collection will be searched for

        Returns
        -------
        bool
            Whether an object exists in the collection or not
        """
        filter_by = filter_by or {}
        self.init_collection(model)
        return self.collection.count_documents(filter_by, limit=1) != 0

    def get_or_create(self, filter_by: dict = None, model: BaseModel = None, **kwargs):
        """Initialize a model by creating it or getting it from the database.

        Parameters
        ----------
        model : Type[BaseModel]
            Class of the model to be searched for or created
        filter_by : dict
            Attributes for finding a document in the database
        **kwargs
            Additional attributes for creating a document (will be overridden by ``filter_by`` attributes)

        Returns
        ----------
        tuple[dict, bool]
            Tuple with the document and whether it was created or not
        """
        filter_by = filter_by or {}
        self.init_collection(model)
        result = self.collection.find_one(filter_by)
        if result is not None:
            return result, False

        kwargs.update(filter_by)
        try:
            model_instance = self.model(**kwargs)
            result = self.collection.insert_one(model_instance)
        except Exception as e:
            raise AttributeError(e)

        return result, True

    def update(self, instance, attrs):
        """Update a document in collection.

        Parameters
        ----------
        instance : BaseModel
            Document to be updated
        attrs : dict
            Attributes to be updated

        Returns
        ----------
        dict
            The updated document
        """
        self.init_collection(type(instance))
        return self.collection.update_one(instance, {"$set": attrs})

    def bulk_update(self, instances: list = None, attrs: list = None, filter_fields: list = None):
        instances = instances or []
        if not len(instances):
            return
        model = instances[0].__class__
        if any(instance.__class__ != model for instance in instances):
            raise TypeError("All instances must have the same model class")
        attrs = attrs or []
        if len(instances) != len(attrs):
            raise ValueError("Length of instances and attributes must match")
        filter_fields = filter_fields or []
        if len(filter_fields) and len(filter_fields) != len(instances):
            raise ValueError("Length of filter_fields must be 0 or equal to instances")
        self.init_collection(model)

        requests = [UpdateOne(filters or instance, {"$set": attr})
                    for instance, attr, filters in zip_longest(instances, attrs, filter_fields)]
        return self.collection.bulk_write(requests=requests, ordered=False)

    def bulk_insert(self, documents: list = None, model: Type[BaseModel] = None):
        """Insert multiple objects to the database improving performance.

        Parameters
        -----------
        documents : list[dict]
            Documents to be added
        model: Type[BaseModel]
            Class of the model documents to be added
        """
        documents = documents or []
        if not len(documents):
            return

        self.init_collection(model)
        documents = [self.model(**doc) for doc in documents]
        return self.collection.insert_many(documents)

    def paginate(self, filter_by=None, page=1, per_page=10, count=True, max_results=50000):
        """Return pagination object with the results.

        https://arpitbhayani.me/blogs/benchmark-and-compare-pagination-approach-in-mongodb
        Parameters
        -----------
        filter_by : dict, list
            Attributes used to find documents in the database or aggregation pipeline
        page : int
            Page of the query
        per_page : int
            Number of items per page
        count : bool
            Whether to count total number of documents in query
        max_results: int
            If counting is used, only count up to this amount of documents

        Returns
        -------
        Pagination
            Paginated results
        """
        if page < 1:
            page = 1
        if per_page < 0:
            per_page = 10

        filter_by = filter_by or {}
        if isinstance(filter_by, dict):
            filter_by = [{"$match": filter_by}]

        # Calculate number of documents to skip
        filter_by.append({"$skip": per_page * (page - 1)})
        filter_by.append({"$limit": per_page if count else per_page + 1})
        cursor = self.collection.aggregate(filter_by)

        # Return documents
        items = list(cursor)
        if not count:
            has_next = len(items) > per_page
            items = items[:-1] if has_next else items
            return PaginationNoCount(self, page, per_page, items, has_next)
        else:
            filter_by.pop()  # Removes $limit step
            filter_by.pop()  # Removes $skip step
            filter_by.append({"$limit": max_results})
            filter_by.append({"$count": "n"})
            summary = list(self.collection.aggregate(filter_by))
            try:
                total = summary[0]["n"]
            except IndexError:
                total = 0
            return Pagination(self, page, per_page, total, items)

    def find_one(self, filter_by: dict = None, model: Type[BaseModel] = None, **kwargs):
        """Find one item of the specified model.

        If there is more than one item an error occurs.
        If there are no items, then it returns None

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            attributes used to find object in the database
        """
        filter_by = filter_by or {}
        self.init_collection(model)
        return self.collection.find_one(filter_by, **kwargs)

    def find_all(self, filter_by: dict = None, model: Type[BaseModel] = None, paginate=True, **kwargs):
        """Find list of items of the specified model.

        If there are too many items a timeout can happen.

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            Attributes used to find documents in the database or aggregation pipeline
        paginate : bool
            Whether to get a paginated result
        kwargs : dict
            All other arguments are passed to `paginate` and/or
            `pymongo.collection.Collection.aggregate`
        """
        self.init_collection(model)
        filter_by = filter_by or {}
        if isinstance(filter_by, dict):
            filter_by = [{"$match": filter_by}]

        if paginate:
            return self.paginate(filter_by, **kwargs)
        else:
            return self.collection.aggregate(filter_by, **kwargs)

    def __repr__(self):
        return f"MongoQuery(model={self.model})"
