from abc import ABC, abstractmethod
from typing import List

from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from db_plugins.db.sql.models import (
    Object,
    Detection,
    NonDetection,
    Feature,
    MagStats,
    Probability,
    Base,
)

from .commons import ValidCommands


class Command(ABC):
    type: str

    def __init__(self, data, criteria=None, options=None):
        self._check_inputs(data, criteria)
        self.data = data
        self.criteria = criteria or {}
        self.options = options

    def _check_inputs(self, data, criteria):
        if not data:
            raise

    @staticmethod
    @abstractmethod
    def db_operation(cls, session: Session, data: List):
        pass


class InsertObjectCommand(Command):
    type = ValidCommands.insert_object

    @staticmethod
    @abstractmethod
    def db_operation(cls, session: Session, data: List):
        return session.connection().execute(insert(Object).values(data))


class InsertDetectionsCommand(Command):
    type: ValidCommands.insert_detections

    @staticmethod
    @abstractmethod
    def db_operation(cls, session: Session, data: List):
        return session.connection().execute(insert(Detection).values(data))


class UpsertFeaturesCommand(Command):
    type: ValidCommands.upsert_features

    @staticmethod
    @abstractmethod
    def db_operation(cls, session: Session, data: List):
        insert_stmt = insert(Feature).values(data)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint=Feature.primary_key, set_=insert_stmt.excluded.value
        )

        return session.connection().execute(insert_stmt)
