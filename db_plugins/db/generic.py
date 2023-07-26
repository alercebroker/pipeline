import abc
from math import ceil
from typing import Type


class DatabaseCreator(abc.ABC):
    """Abstract DatabaseConnection creator class.

    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @classmethod
    @abc.abstractmethod
    def create_database(cls):
        """Abstract factory method.

        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        raise NotImplementedError()


class DatabaseConnection(abc.ABC):
    """Abstract DatabaseConnection class.

    Main database connection interface declares common functionality
    to all databases
    """

    @abc.abstractmethod
    def connect(self, config):
        """Initiate the database connection."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_db(self):
        """Create database collections or tables."""
        raise NotImplementedError()

    @abc.abstractmethod
    def drop_db(self):
        """Remove database collections or tables."""
        raise NotImplementedError()

    @abc.abstractmethod
    def query(self, *args):
        """Get a query object."""
        raise NotImplementedError()


def new_DBConnection(creator: Type[DatabaseCreator]) -> DatabaseConnection:
    """Create a new database connection.

    The client code works with an instance of a concrete creator,
    albeit through its base interface. As long as the client keeps working with
    the creator via the base interface, you can pass it any creator's subclass.
    """
    return creator.create_database()


class BaseQuery(abc.ABC):
    """Abstract Query class."""

    @abc.abstractmethod
    def check_exists(self, model, filter_by):
        """Check if a model exists in the database."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_or_create(self, model, filter_by, **kwargs):
        """Create a model if it doesn't exist in the database.

        It always returns a model instance and whether it was created or not.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, instance, args):
        """Update a model instance with specified args."""
        raise NotImplementedError()

    @abc.abstractmethod
    def paginate(self, page=1, per_page=10, count=True):
        """Return a pagination object from this query."""
        raise NotImplementedError()

    @abc.abstractmethod
    def bulk_insert(self, objects, model):
        """Insert multiple objects to the database."""
        raise NotImplementedError()

    @abc.abstractmethod
    def find_all(self):
        """Retrieve all items from the result of this query."""
        raise NotImplementedError()

    @abc.abstractmethod
    def find_one(self, filter_by={}, model=None, **kwargs):
        """Retrieve only one item from the result of this query.

        Returns None if result is empty.
        """
        raise NotImplementedError()


class Pagination:
    """Paginate responses from the database."""

    def __init__(self, query, page, per_page, total, items):
        """Set attributes from args."""
        self.query = query
        self.page = page
        self.per_page = per_page
        self.total = total
        self.items = items

    @property
    def pages(self):
        """Get total number of pages."""
        if self.per_page == 0 or self.total is None:
            pages = 0
        else:
            pages = int(ceil(self.total / float(self.per_page)))
        return pages

    def prev(self):
        """Return a :class:`Pagination` object for the previous page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page - 1, self.per_page)

    @property
    def prev_num(self):
        """Get number of the previous page."""
        if not self.has_prev:
            return None
        return self.page - 1

    @property
    def has_prev(self):
        """Check if a previous page exists."""
        return self.page > 1

    def next(self):
        """Return a :class:`Pagination` object for the next page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page + 1, self.per_page)

    @property
    def has_next(self):
        """Check if a next page exists."""
        return self.page < self.pages

    @property
    def next_num(self):
        """Get number of the next page."""
        if not self.has_next:
            return None
        return self.page + 1


class PaginationNoCount(Pagination):
    def __init__(self, query, page, per_page, items, has_next):
        super().__init__(query, page, per_page, None, items)
        self._has_next = has_next

    @property
    def has_next(self):
        """Check if a previous page exists."""
        return self._has_next
