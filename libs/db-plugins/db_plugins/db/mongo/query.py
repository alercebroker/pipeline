from itertools import zip_longest
from typing import Type

from pymongo import UpdateOne

from ..generic import BaseQuery, Pagination, PaginationNoCount
from .models import BaseModel


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

    def init_collection(self, model: Type[BaseModel] = None):
        """Sets the collection used by the various methods based on the model
        class provided. If no model is provided, it will use the collection
        provided during creation, if any.

        Raises a :class:`CollectionNotFound` error if no model is provided
        and no collection was selected during creation.

        Parameters
        ----------
        model : Type[BaseModel]
            Model class used for queries
        """
        if model:
            self.model = model
            self.collection = self._db[model._meta.tablename]
        elif self.collection is None:
            raise CollectionNotFound(
                "A valid model must be provided at instantiation or in method call"
            )

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

    def get_or_create(
        self, filter_by: dict = None, model: Type[BaseModel] = None, **kwargs
    ):
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
        tuple[dict or InsertOneResult, bool]
            Tuple with the document (if not created) and whether it was created or not
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
        pymongo.results.UpdateResult
        """
        self.init_collection(type(instance))
        return self.collection.update_one(instance, {"$set": attrs})

    def bulk_update(self, instances: list, attrs: list, filter_fields: list = None):
        """Update multiple documents in a collection at once.

        The length of `instances` and `attrs` must be the same.

        If `filter_fields` is defined, it must also match the length of
        `instances` and its contents will take precedence over the
        corresponding instance when querying for documents to update.

        Parameters
        ----------
        instances : list[BaseModel]
            List with documents to be updated
        attrs : list[dict]
            List with dictionary of fields to update for corresponding instance
        filter_fields : list[dict], optional
            Attributes for finding the document that needs to be updates

        Returns
        -------
        pymongo.results.BulkWriteResult
        """
        if len(instances) == 0:
            return
        model = type(instances[0])
        if any(type(instance) != model for instance in instances):
            raise TypeError("All instances must have the same model class")
        if len(instances) != len(attrs):
            raise ValueError("Length of instances and attributes must match")
        filter_fields = filter_fields or []
        if len(filter_fields) and len(filter_fields) != len(instances):
            raise ValueError("Length of filter_fields must be 0 or equal to instances")
        self.init_collection(model)

        requests = [
            UpdateOne(filters or instance, {"$set": attr})
            for instance, attr, filters in zip_longest(instances, attrs, filter_fields)
        ]
        return self.collection.bulk_write(requests=requests, ordered=False)

    def bulk_insert(self, documents: list, model: Type[BaseModel] = None):
        """Insert multiple documents to the database at once.

        Parameters
        -----------
        documents : list[dict]
            Documents to be added
        model: Type[BaseModel]
            Class of the model documents to be added

        Returns
        -------
        pymongo.results.InsertManyResult
        """
        if len(documents) == 0:
            return
        self.init_collection(model)

        documents = [self.model(**doc) for doc in documents]
        return self.collection.insert_many(documents)

    def paginate(
        self, filter_by=None, page=1, per_page=10, count=True, max_results=50000
    ):
        """Return pagination object with selected documents.

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
            filter_by.pop()  # Removes pagination $limit step
            filter_by.pop()  # Removes pagination $skip step
            # $sorting can be too slow and is fully irrelevant for counting
            filter_by = [step for step in filter_by if "$sort" not in step]
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

        If there are no items, then it returns None.

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            Attributes used to find object document in the database

        Returns
        -------
        dict, None
            First document that matches
        """
        filter_by = filter_by or {}
        self.init_collection(model)
        return self.collection.find_one(filter_by, **kwargs)

    def find_all(
        self,
        filter_by: dict = None,
        model: Type[BaseModel] = None,
        paginate=True,
        **kwargs,
    ):
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

        Returns
        -------
        pymongo.cursor.Cursor or Paginate
            Cursor to iterate over query results or a pagination object,
            depending on the `paginate` option
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
        return f"MongoQuery(model={self.model.__class__.__name__})"
