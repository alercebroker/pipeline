from pymongo.errors import WriteConcernError, WriteError

from ..generic import BaseQuery, Pagination

class MongoQuery(BaseQuery):

    def __init__(self, connection, collection=None, **kwargs):
        """Get / create a Mongo collection.

        Raises :class:`TypeError` if `name` is not
        """
        self.__connection = connection
        self.__collection = collection

    def check_exists(self, collection=None, filter_by={}):
        """
        Check if record exists in database.

        :param session: The connection session
        :param model: The class of the model to be instantiated
        :param dict filter_by: attributes used to find object in the database
        :param dict kwargs: attributes used to create the object that are not used in filter_by

        :returns: True if object exists else False

        """
        self.__collection = self.__collection if self.__collection else collection
        return self.__collection.count_documents(filter_by, limit=1) != 0

    def get_or_create(self, collection=None, filter_by=None, **kwargs):
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

        self.__collection = self.__collection if self.__collection else collection
        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)
        result = mycolObj.find_one(filter_by)
        created = False
        if result is not None:
            return result, created

        try:
            kwargs.update(filter_by)
            result = mycolObj.insertOne(kwargs)
            created = True
        except Exception as e:
            print("An exception occurred ::", e)
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

        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)

        mycolObj.update_one(instance, args)

        return instance

    def bulk_insert(self, objects, collection=None):
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

        self.__collection = self.__collection if self.__collection else collection
        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)
        mycolObj.insert_many(objects)


    def paginate(self, page=1, per_page=10, count=True, max_results=50000):
        """
        Returns pagination object with the results
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

        # Calculate number of documents to skip
        skips = per_page * (page - 1)

        # Skip and limit
        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)
        cursor = mycolObj.find().skip(skips).limit(per_page)

        # Return documents
        items = [x for x in cursor]
        if not count:
            total = None
        else:
            all_docs = mycolObj.count_documents()
            total = all_docs if all_docs < max_results else max_results
        return Pagination(self, page, per_page, total, items)

    def find_one(self, collection=None, filter_by={}):
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

        self.__collection = self.__collection if self.__collection else collection
        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)
        return mycolObj.find_one(filter_by)

    def find_all(self, collection=None, filter_by={}, paginate=True):
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

        self.__collection = self.__collection if self.__collection else collection
        mycolObj = self.__connection.db.__getitem__(self.__collection.__tablename__)

        if paginate:
            return self.paginate()
        else:
            return mycolObj.find(filter_by)
