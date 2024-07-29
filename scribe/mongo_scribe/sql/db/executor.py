from typing import Callable, Dict, List


from .connection import PSQLConnection, Session
from mongo_scribe.sql.command.commands import (
    Command,
    InsertObjectCommand,
    UpdateObjectFromStatsCommand,
    InsertDetectionsCommand,
    InsertForcedPhotometryCommand,
    UpdateObjectStatsCommand,
    UpsertFeaturesCommand,
    UpsertNonDetectionsCommand,
    UpsertProbabilitiesCommand,
    UpsertXmatchCommand,
    UpsertScoreCommand,
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
        self.connection = PSQLConnection(config["PSQL"])
        commands = (
            InsertObjectCommand,
            UpdateObjectFromStatsCommand,
            InsertDetectionsCommand,
            InsertForcedPhotometryCommand,
            UpdateObjectStatsCommand,
            UpsertFeaturesCommand,
            UpsertNonDetectionsCommand,
            UpsertProbabilitiesCommand,
            UpsertXmatchCommand,
            UpsertScoreCommand,
        )
        self.handlers: Dict[str, CommandHandler] = {
            c.type: CommandHandler(c.db_operation) for c in commands
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
