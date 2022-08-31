from db_plugins.db.generic import BaseQuery, Pagination, PaginationNoCount
from db_plugins.db.mongo.models import Base
from pymongo.collection import Collection as PymongoCollection
from pymongo import UpdateOne


def constructor_creator(collection_class):
    def constructor(self, database, model: Base = None, *args, **kwargs):
        """Get / create a Mongo collection.

        Raises :class:`TypeError` if `name` is not
        """
        self.model = model
        self.initialized_collection = False
        self._db = database
        if "name" in kwargs:
            # Ignore model and use pure pymongo API
            collection_class.__init__(
                self,
                database=database,
                *args,
                **kwargs,
            )
            self.initialized_collection = True
        elif model:
            # Using custom ORM API
            self.init_collection(model, **kwargs)
            self.initialized_collection = True

    return constructor


def init_collection_creator(collection_class):
    def init_collection(self, model, **kwargs):
        if self.initialized_collection:
            return
        if model:
            self.model = model
            collection_class.__init__(
                self,
                name=model._meta.tablename,
                database=self._db,
                **kwargs,
            )
            # get rid of temporary attribute
            self.__delattr__("_db")
            self.initialized_collection = True
            return
        if not self.model:
            self.raise_collection_not_found_error()

    return init_collection


def check_exists(
    self,
    model: Base = None,
    filter_by={},
):
    """
    Check if record exists in database.

    :param session: The connection session
    :param model: The class of the model to be instantiated
    :param dict filter_by: attributes used to find object in the database
    :param dict kwargs: attributes used to create the object that are not
    used in filter_by

    :returns: True if object exists else False

    """
    self.init_collection(model)
    return self.count_documents(filter_by, limit=1) != 0


def get_or_create_creator(collection_class):
    def get_or_create(self, filter_by={}, model: Base = None, **kwargs):
        """Initialize a model by creating it or getting it from the database.

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
        self.init_collection(model)
        result = collection_class.find_one(self, filter_by)
        created = False
        if result is not None:
            return result, created

        try:
            kwargs.update(filter_by)
            model_instance = self.model(**kwargs)
            result = self.insert_one(model_instance)
            created = True
        except Exception as e:
            raise AttributeError(e)
            created = False

        return result, created

    return get_or_create


def update_creator(collection_class):
    def update(self, instance, args):
        """Update an object.

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
        model = type(instance)
        self.init_collection(model)
        update_op = {"$set": args}
        return collection_class.update_one(self, instance, update_op)

    return update


def bulk_update_creator(collection_class):
    def bulk_update(self, instances: list, args: list, filter_fields: list = []):
        model = type(instances[0])
        self.init_collection(model)
        requests = []
        for i, instance in enumerate(instances):
            requests.append(
                UpdateOne(
                    filter_fields[i] if len(filter_fields) > 0 else instance,
                    {"$set": args[i]},
                )
            )
        return collection_class.bulk_write(self, requests=requests, ordered=False)

    return bulk_update


def bulk_insert_creator(collection_class):
    def bulk_insert(self, objects, model=None):
        """Insert multiple objects to the database improving performance.

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

        self.init_collection(model)
        objects = [self.model(**obj) for obj in objects]
        return collection_class.insert_many(self, objects)

    return bulk_insert


def paginate(
    self,
    filter_by={},
    page=1,
    per_page=10,
    count=True,
    max_results=50000,
):
    """Return pagination object with the results.

    https://arpitbhayani.me/blogs/benchmark-and-compare-pagination-approach-in-mongodb
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

    if isinstance(filter_by, dict):
        filter_by = [{"$match": filter_by}]

    # Calculate number of documents to skip
    filter_by.append({"$skip": per_page * (page - 1)})
    filter_by.append({"$limit": per_page if count else per_page + 1})
    # Skip and limit
    cursor = self.aggregate(filter_by)

    # Return documents
    items = list(cursor)
    if not count:
        has_next = len(items) > per_page
        items = items[:-1] if has_next else items
        return PaginationNoCount(self, page, per_page, items, has_next)
    else:
        filter_by.pop()
        filter_by.pop()
        filter_by.append({"$limit": max_results})
        filter_by.append({"$count": "n"})
        summary = list(self.aggregate(filter_by))
        try:
            total = summary[0]["n"]
        except IndexError:
            total = 0
        return Pagination(self, page, per_page, total, items)


def find_one_creator(collection_class):
    def find_one(self, filter_by={}, model: Base = None):
        """Find one item of the specified model.

        If there are more than one item an error occurs.
        If there are no items, then it returns None

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            attributes used to find object in the database
        """
        self.init_collection(model)
        return collection_class.find_one(self, filter_by)

    return find_one


def find_all_creator(collection_class):
    def find_all(self, model: Base = None, filter_by={}, paginate=True, **kwargs):
        """Find list of items of the specified model.

        If there are too many items a timeout can happen.

        Parameters
        -----------
        model : Model
            Class of the model to be retrieved
        filter_by : dict
            attributes used to find objects in the database
        paginate : bool
            whether to get a paginated result or not
        kwargs : dict
            all other arguments are passed to `paginate` and/or
            `pymongo.collection.Collection.aggregate`
        """
        self.init_collection(model)

        if isinstance(filter_by, dict):
            filter_by = [{"$match": filter_by}]

        if paginate:
            return self.paginate(filter_by, **kwargs)
        else:
            return collection_class.aggregate(self, filter_by, **kwargs)

    return find_all


def raise_collection_not_found_error(self):
    class CollectionNotFound(Exception):
        pass

    raise CollectionNotFound(
        "You should provide model argument to either the MongoQuery instance or this method call"
    )


def __repr__(self):
    if not self.initialized_collection:
        return f"MongoQuery(model={self.model})"
    else:
        return super().__repr__()


def mongo_query_creator(collection_class=PymongoCollection) -> BaseQuery:
    class_dict = {
        "__init__": constructor_creator(collection_class),
        "init_collection": init_collection_creator(collection_class),
        "check_exists": check_exists,
        "get_or_create": get_or_create_creator(collection_class),
        "update": update_creator(collection_class),
        "bulk_update": bulk_update_creator(collection_class),
        "bulk_insert": bulk_insert_creator(collection_class),
        "paginate": paginate,
        "find_one": find_one_creator(collection_class),
        "find_all": find_all_creator(collection_class),
    }
    return type(
        "MongoQuery",
        (collection_class, BaseQuery),
        class_dict,
    )
