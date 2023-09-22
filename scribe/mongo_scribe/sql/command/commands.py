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
        self.data = self._format_data(data)
        self.criteria = criteria or {}
        self.options = options

    def _format_data(self, data):
        return data

    def _check_inputs(self, data, criteria):
        if not data:
            raise ValueError("Not data provided in command")

    @staticmethod
    @abstractmethod
    def db_operation(session: Session, data: List):
        pass


class InsertObjectCommand(Command):
    type = ValidCommands.insert_object

    @staticmethod
    def db_operation(session: Session, data: List):
        return session.connection().execute(
            insert(Object).values(data).on_conflict_do_nothing()
        )


class InsertDetectionsCommand(Command):
    type = ValidCommands.insert_detections

    @staticmethod
    def db_operation(session: Session, data: List):
        return session.connection().execute(
            insert(Detection).values(data).on_conflict_do_nothing()
        )


class UpsertFeaturesCommand(Command):
    type = ValidCommands.upsert_features

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if "features" not in data:
            raise ValueError("No features provided in command")
        if "features_version" not in data:
            raise ValueError("No feature version provided in command")
        if "features_group" not in data:
            raise ValueError("No feature group provided in command")

    def _format_data(self, data):
        return [
            {**feat, "version": data["features_version"]} for feat in data["features"]
        ]

    @staticmethod
    def db_operation(session: Session, data: List):
        insert_stmt = insert(Feature).values(data)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint=Feature.primary_key, set_=insert_stmt.excluded.value
        )

        return session.connection().execute(insert_stmt)
