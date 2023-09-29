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
)

from .commons import ValidCommands


class Command(ABC):
    type: str

    def __init__(self, data, criteria=None, options=None):
        self._check_inputs(data, criteria)
        self.criteria = criteria or {}
        self.options = options
        self.data = self._format_data(data)

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


class UpdateObjectStatsCommand(Command):
    type = ValidCommands.update_object_stats

    def _check_inputs(self, data, criteria):
        if "oid" not in criteria:
            raise ValueError("Not oid provided in command")
        if "magstats" not in data:
            raise ValueError("Magstats not provided in the commands data")

    def _format_data(self, data):
        magstats = data.pop("magstats")
        data["oid"] = self.criteria["oid"]
        for magstat in magstats:
            magstat.pop("sid")
            fid_map = {"g": 1, "r": 2, "i": 3}

            magstat["oid"] = self.criteria["oid"]
            magstat["fid"] = fid_map[magstat["fid"]]
            if "step_id_corr" not in magstat:
                magstat["step_id_corr"] = "ALeRCE ZTF"

        return (data, magstats)

    @staticmethod
    def db_operation(session: Session, data: List):
        # data should be a tuple where idx 0 is objstats and 1 is magstats
        objstats, magstats = map(list, zip(*data))
        # list flatten
        magstats = sum(magstats, [])
        print(objstats)
        for stat in objstats:
            oid = stat.pop("oid")
            update_stmt = update(Object).where(Object.oid == oid).values(stat)
            session.connection().execute(update_stmt)

        upsert_stmt = insert(MagStats).values(magstats)
        upsert_stmt = upsert_stmt.on_conflict_do_update(
            constraint="magstat_pkey", set_=upsert_stmt.excluded
        )
        return session.connection().execute(upsert_stmt)


class UpsertNonDetectionsCommand(Command):
    type = ValidCommands.upsert_non_detections

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if any([field not in criteria for field in ["oid", "fid", "mjd"]]):
            raise ValueError("Needed 'oid', 'mjd' and 'fid' as criteria")
        self.criteria = criteria

    def _format_data(self, data):
        return [{**self.criteria, "diffmaglim": data["diffmaglim"]}]

    @staticmethod
    def db_operation(session: Session, data: List):
        insert_stmt = insert(NonDetection).values(data)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="non_detection_pkey",
            set_=dict(diffmaglim=insert_stmt.excluded.diffmaglim),
        )
        return session.connection().execute(insert_stmt)


class UpsertFeaturesCommand(Command):
    type = ValidCommands.upsert_features

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if "oid" not in criteria:
            raise ValueError("No oids were provided in command")
        if "features" not in data:
            raise ValueError("No features provided in command")
        if "features_version" not in data:
            raise ValueError("No feature version provided in command")
        if "features_group" not in data:
            raise ValueError("No feature group provided in command")

    def _format_data(self, data):
        return [
            {**feat, "version": data["features_version"], "oid": i}
            for feat in data["features"]
            for i in self.criteria["oid"]
        ]

    @staticmethod
    def db_operation(session: Session, data: List):
        insert_stmt = insert(Feature).values(data)

        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="feature_pkey", set_=dict(value=insert_stmt.excluded.value)
        )

        return session.connection().execute(insert_stmt)


class UpsertProbabilitiesCommand(Command):
    type = ValidCommands.upsert_probabilities

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if "oid" not in criteria:
            raise ValueError("No oids were provided in command")


    def _format_data(self, data):
        return [{**data, "oid": i} for i in self.criteria["oid"]]

    @staticmethod
    def db_operation(session: Session, data: List):
        insert_stmt = insert(Probability).values(data)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="probability_pkey",
            set_=dict(
                ranking=insert_stmt.excluded.ranking,
                probability=insert_stmt.excluded.probability,
            ),
        )

        return session.connection().execute(insert_stmt)
