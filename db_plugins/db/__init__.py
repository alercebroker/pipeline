import abc
from db_plugins.db.sql import SQLConnection

class DatabaseConnectionCreator(abc.ABC):
    @abc.abstractmethod
    def create_connection(self):
        pass

    def connect(self, **kwargs):
        connection = self.create_connection()
        connection.start(**kwargs)

    def create_db(self):
        connection = self.create_connection()
        connection.create_db()

    def drop_db(self):
        connection = self.create_connection()
        connection.drop_db()

    def query(self, *args):
        connection = self.create_connection()
        return connection.query(*args)


class SQLConnectionCreator(DatabaseConnectionCreator):
    conn = None
    def create_connection(self):
        if not self.conn:
            self.conn = SQLConnection()
        return self.conn

    @property
    def session(self):
        return self.conn.session


