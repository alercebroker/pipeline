from typing import Callable, Dict, List


from .connection import PSQLConnection, Session
from sql_scribe.sql.command.commands import (
    Command,
    ZTFCorrectionCommand, 
    ZTFMagstatCommand,
    LSSTMagstatCommand,
    LSSTUpdateDiaObjectCommand,
    LSSTFeatureCommand,
    LSSTCorrectionCommand
)

class CommandHandler:
    def __init__(self, query_function: Callable[[Session, List], None]):
        self.query = query_function
        self.data = []

    def add_command(self, command: Command):
        if isinstance(command.data, list):
            self.data.extend(command.data)
            return

        self.data.append(command.data)

    def execute(self, connection: PSQLConnection):
        if self.data == []:
            return
        with connection.session() as session:
            self.query(session, self.data)
            session.commit()
        self.data = []


class SQLCommandExecutor:
    def __init__(self, config: dict) -> None:
        self.connection = PSQLConnection(config)
        commands_list = (
            ZTFCorrectionCommand,
            ZTFMagstatCommand,
            LSSTMagstatCommand,
            LSSTUpdateDiaObjectCommand,
            LSSTFeatureCommand,
            LSSTCorrectionCommand
        )
        self.handlers: Dict[str, CommandHandler] = {
            c.type: CommandHandler(c.db_operation) for c in commands_list
        }

    def _add_command(self, command: Command):
        self.handlers[command.type].add_command(command)

    def _execute_commands(self):
        for handler in self.handlers.values():
            handler.execute(self.connection)

    def bulk_execute(self, commands: List[Command]):
        for command in commands:
            self._add_command(command)

        self._execute_commands()
