"""Main classes for using the Mongo ORM"""
from pymongo import MongoClient, IndexModel


class Metadata:
    database = "DEFAULT_DATABASE"
    collections = {}

    def create_all(self, client: MongoClient, database: str = None):
        self.database = database or self.database
        db = client[self.database]
        for c in self.collections:
            collection = db[c]
            collection.create_indexes(self.collections[c]["indexes"])

    def drop_all(self, client: MongoClient, database: str = None):
        self.database = database or self.database
        client.drop_database(self.database)


class BaseMetaClass(type):
    """Metaclass for Base model class that creates mappings."""

    metadata = Metadata()

    def __new__(cls, name, bases, attrs):
        """Create class with mappings and other metadata."""
        if name == "Base":
            return type.__new__(cls, name, bases, attrs)
        indexes = []
        tablename = name
        fields = {}
        class_dict = {}
        for k, v in attrs.items():
            if k == "__table_args__":
                for arg in v:
                    if isinstance(arg, IndexModel):
                        indexes.append(arg)
            if k == "__tablename__":
                tablename = v
            if isinstance(v, Field):
                fields[v.name] = v

        class_dict["__indexes__"] = indexes
        class_dict["__fields__"] = fields
        class_dict["__tablename__"] = tablename
        cls.metadata.collections[tablename] = {
            "indexes": indexes,
            "fields": fields,
        }
        return type.__new__(cls, name, bases, class_dict)


def base_creator():
    """Create a Base class for declarative model definition."""

    class Base(dict, metaclass=BaseMetaClass):
        def __init__(self, *args, **kwargs):
            model = {}
            for field in self.__fields__:
                try:
                    if isinstance(self.__fields__[field], SpecialField):
                        model[field] = self.__fields__[field].callback(**kwargs)
                    else:
                        model[field] = kwargs[field]
                except KeyError:
                    raise AttributeError(
                        "{} model needs {} attribute".format(
                            self.__class__.__name__, field
                        )
                    )
            super(Base, self).__init__(*args, **model)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(
                    "{} has no attribute {}".format(
                        self.__class__.__name__,
                        key,
                    )
                )

        def __setattr__(self, key, value):
            self[key] = value

        def __str__(self):
            return dict.__str__(self)

        def __repr__(self):
            return dict.__repr__(self)

    return Base


class Field:
    def __init__(self, name):
        self.name = name


class SpecialField(Field):
    def __init__(self, name, callback):
        super(SpecialField, self).__init__(name)
        self.callback = callback
