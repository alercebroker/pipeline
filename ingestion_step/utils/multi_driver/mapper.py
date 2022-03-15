from db_plugins.db.mongo import models as mongo_models
from db_plugins.db.sql import models as psql_models


class Mapper:
    def __init__(self):
        pass

    def _convert_object(self, object_: mongo_models.Object) -> psql_models.Object:
        pass

    def convert(self, mongo_model):
        pass
