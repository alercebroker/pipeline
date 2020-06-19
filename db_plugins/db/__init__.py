import abc
from .sql import SQLConnection

class ALeRCEDatabase(abc.ABC):
    @abc.abstractmethod
    def create_database(self):
        pass

    def connect(self, **kwargs):
        connection = self.create_database()
        connection.start(**kwargs)

    def create_db(self):
        connection = self.create_database()
        connection.create_db()

    def drop_db(self):
        connection = self.create_database()
        connection.drop_db()


class SQLDatabase(ALeRCEDatabase):
    conn = None
    def create_database(self):
        if not self.conn:
            self.conn = SQLConnection()
        return self.conn

    @property
    def session(self):
        return self.conn.session
        