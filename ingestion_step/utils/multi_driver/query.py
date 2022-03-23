from db_plugins.db.generic import BaseQuery
from db_plugins.db.mongo import MongoConnection
from db_plugins.db.sql import SQLConnection
from typing import List
from sqlalchemy import and_
from sqlalchemy.sql.expression import bindparam

import db_plugins.db.mongo.models as mongo_models
import db_plugins.db.sql.models as psql_models

MODELS = {
    "psql": {
        "Object": psql_models.Object,
        "Detection": psql_models.Detection,
        "NonDetection": psql_models.NonDetection,
        "MagStats": psql_models.MagStats,
        "Ps1_ztf": psql_models.Ps1_ztf,
        "Ss_ztf": psql_models.Ss_ztf,
        "Reference": psql_models.Reference,
        "Dataquality": psql_models.Dataquality,
        "Gaia_ztf": psql_models.Gaia_ztf,
    },
    "mongo": {
        "Object": mongo_models.Object,
        "Detection": mongo_models.Detection,
        "NonDetection": mongo_models.NonDetection
    }
}


def filter_to_psql(model: object, filter_by: dict):
    filters = []
    for attribute, _filter in filter_by.items():
        if "$in" in _filter:
            attribute = "oid" if attribute == "aid" else attribute
            f = getattr(model, attribute).in_(_filter["$in"])
            filters.append(f)
    if len(filters) == 1:
        return filters[0]
    elif len(filters) > 1:
        return and_(*filters)
    return {}


def update_to_psql(model: object, filter_by: List[dict]):
    filters = []
    for attribute, _filter in filter_by[0].items():
        if attribute == "_id" and (model is psql_models.Object or model is psql_models.Ps1_ztf):
            f = getattr(model, "oid") == bindparam("oid")
            filters.append(f)
    if len(filters) == 1:
        return filters[0]
    elif len(filters) > 1:
        return and_(*filters)
    return {}


class MultiQuery(BaseQuery):
    def __init__(self,
                 psql_driver: SQLConnection,
                 mongo_driver: MongoConnection,
                 model=None,
                 mapper=None,
                 *args,
                 **kwargs):
        self.model = model
        self.psql = psql_driver
        self.mongo = mongo_driver
        self.engine = kwargs.get("engine", "mongo")
        self.mapper = mapper

    def check_exists(self, model, filter_by):
        """Check if a model exists in the database."""
        raise NotImplementedError()

    def get_or_create(self, model, filter_by, **kwargs):
        pass

    def update(self, instance, args):
        """Update a model instance with specified args."""
        raise NotImplementedError()

    def bulk_update(self, to_update: List, filter_by: List[dict]):
        try:
            model = MODELS[self.engine][self.model]
        except Exception as e:
            raise Exception(f"Indicates model on query() method: {e}")

        if self.engine == "mongo":
            to_update = [model(**x) for x in to_update]
            return self.mongo.query().bulk_update(to_update, to_update, filter_fields=filter_by)
        elif self.engine == "psql":
            to_update = self.mapper.convert(to_update, model)
            bind_object = to_update[0]
            where_clause = update_to_psql(model, filter_by)
            params_statement = dict(zip(bind_object.keys(), map(bindparam, bind_object.keys())))
            statement = model.__table__.update().where(where_clause).values(params_statement)
            return self.psql.engine.execute(statement, to_update)
        raise NotImplementedError()

    def paginate(self, page=1, per_page=10, count=True):
        """Return a pagination object from this query."""
        raise NotImplementedError()

    def bulk_insert(self, objects: List[dict]):
        try:
            model = MODELS[self.engine][self.model]
        except Exception as e:
            raise Exception(f"Indicates model on query() method: {e}")

        if self.engine == "mongo":
            self.mongo.query().bulk_insert(objects, model)
        elif self.engine == "psql":
            psql_data = self.mapper.convert(objects, model)
            self.psql.query().bulk_insert(psql_data, model)
        else:
            raise NotImplementedError()

    def find_all(self, filter_by={}, paginate=False):
        """Retrieve all items from the result of this query."""
        try:
            model = MODELS[self.engine][self.model]
        except Exception as e:
            raise Exception(f"Indicates model on query() method: {e}")

        if self.engine == "mongo":
            cursor = self.mongo.query().find_all(model=model, filter_by=filter_by, paginate=paginate)
            return [x for x in cursor]
        elif self.engine == "psql":
            filter_by = filter_to_psql(model, filter_by)
            response = self.psql.query().find_all(model=model, filter_by=filter_by, paginate=paginate)
            response = [x.__dict__ for x in response]
            for x in response:
                del x["_sa_instance_state"]
            return response
        else:
            raise NotImplementedError()

    def find_one(self, filter_by={}, model=None, **kwargs):
        """Retrieve only one item from the result of this query.
        Returns None if result is empty.
        """
        raise NotImplementedError()
