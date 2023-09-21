from typing import Callable, List

from db_plugins.db.sql.models import (
    Object,
    Detection,
    NonDetection,
    Feature,
    MagStats,
    Probability,
    Base
)

from .connection import PSQLConnection, Session
from mongo_scribe.sql.command.commands import Command

class CommandHandler:
    def __init__(self, table, query_function: Callable[[Session, Base, List], None]):
        self.query = query_function
        self.data = []
        self.table = table

    def add_command(self, command: Command):
        self.data.append(command.get_operations())

    def execute(self, connection: PSQLConnection):
        if self.data == []:
            return
        
        with connection.session() as session:
            self.query(session, self.table, self.data)

class SQLCommandExecutor:
    def __init__(self, config: dict) -> None:
        self.connection = PSQLConnection(config["PSQL"])

    def _add_command(self):
        pass

    def _execute_commands(self):
        pass

    def bulk_execute(self, commands: List[Command]):
        pass