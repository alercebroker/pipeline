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


class ModelMetadata:
    def __init__(self, tablename: str, fields: dict, indexes: list):
        self.tablename = tablename
        self.fields = fields
        self.indexes = indexes

    def __repr__(self):
        dict_repr = {
            "tablename": self.tablename,
            "fields": self.fields,
            "indexes": self.indexes,
        }
        return str(dict_repr)


class BaseMetaClass(type):
    """Metaclass for Base model class that creates mappings."""

    metadata = Metadata()

    def __new__(cls, name, bases, attrs):
        """Create class with mappings and other metadata."""
        if name == "Base":
            return type.__new__(cls, name, bases, attrs)
        class_dict = {"_meta": ModelMetadata(name, {}, [])}
        for k, v in attrs.items():
            if k == "__table_args__":
                for arg in v:
                    if isinstance(arg, IndexModel):
                        class_dict["_meta"].indexes.append(arg)
            if k == "__tablename__":
                class_dict["_meta"].tablename = v
            if isinstance(v, Field):
                class_dict["_meta"].fields[k] = v

        cls.metadata.collections[class_dict["_meta"].tablename] = {
            "indexes": class_dict["_meta"].indexes,
            "fields": class_dict["_meta"].fields,
        }
        return type.__new__(cls, name, bases, class_dict)


def base_creator():
    """Create a Base class for declarative model definition."""

    class Base(dict, metaclass=BaseMetaClass):
        def __init__(self, **kwargs):
            model = {}
            for field in self._meta.fields:
                try:
                    if isinstance(self._meta.fields[field], SpecialField):
                        model[field] = self._meta.fields[field].callback(
                            Model=self.__class__, **kwargs
                        )
                    else:
                        model[field] = kwargs[field]
                except KeyError:
                    raise AttributeError(
                        "{} model needs {} attribute".format(
                            self.__class__.__name__, field
                        )
                    )
            super(Base, self).__init__(**model)

        @classmethod
        def set_database(cls, database):
            cls.metadata.database = database

        def __str__(self):
            return dict.__str__(self)

        def __repr__(self):
            return dict.__repr__(self)

    return Base


class Field:
    def __init__(self):
        pass


class SpecialField(Field):
    def __init__(self, callback):
        super(SpecialField, self).__init__()
        self.callback = callback
