from abc import ABC, abstractmethod
from typing import Dict, List

from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from db_plugins.db.sql.models import (
    Object,
    ForcedPhotometry,
    Detection,
    NonDetection,
    Feature,
    MagStats,
    Probability,
    Xmatch,
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

    def _check_inputs(self, data, criteria):
        return super()._check_inputs(data, criteria)

    def _format_data(self, data: Dict):
        exclude = [
            "aid",
            "sid",
            "tid",
            "extra_fields",
            "e_dec",
            "e_ra",
            "stellar",
        ]
        fid_map = {"g": 1, "r": 2, "i": 3}

        field_mapping = {
            "mag": "magpsf",
            "e_mag": "sigmapsf",
            "mag_corr": "magpsf_corr",
            "e_mag_corr": "sigmapsf_corr",
            "e_mag_corr_ext": "sigmapsf_corr_ext",
        }
        _extra_fields = [
            "nid",
            "magap",
            "sigmagap",
            "rfid",
            "diffmaglim",
            "distnr",
            "magapbig",
            "rb",
            "rbversion",
            "sigmagapbig",
            "drb",
            "drbversion",
        ]
        new_data = data.copy()
        # rename some fields
        for k, v in field_mapping.items():
            new_data[v] = new_data.pop(k)

        # add fields from extra_fields
        for field in _extra_fields:
            if field in data["extra_fields"]:
                new_data[field] = data["extra_fields"][field]

        new_data = {k: v for k, v in new_data.items() if k not in exclude}
        new_data["step_id_corr"] = data.get("step_id_corr", "ALeRCE v3")
        new_data["parent_candid"] = (
            int(data["parent_candid"])
            if data["parent_candid"] != "None"
            else None
        )
        new_data["fid"] = fid_map[new_data["fid"]]

        return {**new_data, "candid": int(self.criteria["candid"])}

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["candid"], el["oid"]): el for el in data}
        unique = list(unique.values())
        stmt = insert(Detection).values(unique)
        return session.connection().execute(
            stmt.on_conflict_do_update(
                constraint="detection_pkey", set_=dict(oid=stmt.excluded.oid)
            )
        )


class InsertForcedPhotometryCommand(Command):
    type = ValidCommands.insert_forced_photo

    def _format_data(self, data: Dict):
        exclude = [
            "aid",
            "sid",
            "candid",
            "tid",
            "e_dec",
            "e_ra",
            "stellar",
            "extra_fields",
        ]
        fid_map = {"g": 1, "r": 2, "i": 3}

        extra_fields = data["extra_fields"]
        extra_fields.pop("brokerIngestTimestamp", "")
        extra_fields.pop("surveyPublishTimestamp", "")
        extra_fields.pop("parent_candid", "")
        extra_fields.pop("forcediffimfluxunc", "")
        new_data = data.copy()
        new_data = {k: v for k, v in new_data.items() if k not in exclude}
        new_data["fid"] = fid_map[new_data["fid"]]

        return {**new_data, **extra_fields}

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["pid"], el["oid"]): el for el in data}
        unique = list(unique.values())
        statement = insert(ForcedPhotometry)
        return session.connection().execute(
            statement.on_conflict_do_update(
                constraint="forced_photometry_pkey", set_=statement.excluded
            ),
            unique,
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
        for stat in objstats:
            oid = stat.pop("oid")
            update_stmt = update(Object).where(Object.oid == oid).values(stat)
            session.connection().execute(update_stmt)

        unique_magstats = {(el["oid"], el["fid"]): el for el in magstats}
        unique_magstats = list(unique_magstats.values())
        upsert_stmt = insert(MagStats).values(unique_magstats)
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
        fid_map = {"g": 1, "r": 2, "i": 3}
        self.criteria["fid"] = fid_map[self.criteria["fid"]]
        return [{**self.criteria, "diffmaglim": data["diffmaglim"]}]

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["oid"], el["fid"], el["mjd"]): el for el in data}
        unique = list(unique.values())
        insert_stmt = insert(NonDetection).values(unique)
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
        FID_MAP = {None: 0, "": 0, "g": 1, "r": 2, "gr": 12, "rg": 12}
        return [
            {
                **feat,
                "version": data["features_version"],
                "oid": i,
                "fid": FID_MAP[feat["fid"]]
                if isinstance(feat["fid"], str) or feat["fid"] is None
                else feat["fid"],
            }
            for feat in data["features"]
            for i in self.criteria["oid"]
        ]

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["oid"], el["name"], el["fid"]): el for el in data}
        unique = list(unique.values())
        insert_stmt = insert(Feature).values(unique)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="feature_pkey",
            set_=dict(value=insert_stmt.excluded.value),
        )

        return session.connection().execute(insert_stmt)


class UpsertProbabilitiesCommand(Command):
    type = ValidCommands.upsert_probabilities

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if "oid" not in criteria:
            raise ValueError("No oids were provided in command")

    def _format_data(self, data):
        classifier_name = data.pop("classifier_name")
        classifier_version = data.pop("classifier_version")

        parsed = [
            {
                "classifier_name": classifier_name,
                "classifier_version": classifier_version,
                "class_name": class_name,
                "probability": value,
            }
            for class_name, value in data.items()
        ]
        parsed.sort(key=lambda e: e["probability"], reverse=True)
        parsed = [{**el, "ranking": i + 1} for i, el in enumerate(parsed)]
        return [
            {**el, "oid": oid} for el in parsed for oid in self.criteria["oid"]
        ]

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {
            (el["oid"], el["classifier_name"], el["class_name"]): el
            for el in data
        }
        unique = list(unique.values())
        insert_stmt = insert(Probability).values(unique)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="probability_pkey",
            set_=dict(
                ranking=insert_stmt.excluded.ranking,
                probability=insert_stmt.excluded.probability,
            ),
        )

        return session.connection().execute(insert_stmt)


class UpsertXmatchCommand(Command):
    type = ValidCommands.upsert_xmatch

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)
        if "oid" not in criteria or criteria["oid"] == []:
            raise ValueError("No oids were provided in command")

    def _format_data(self, data):
        data["xmatch"]["oid_catalog"] = data["xmatch"].pop("catoid")
        return [{**data["xmatch"], "oid": oid} for oid in self.criteria["oid"]]

    @staticmethod
    def db_operation(session: Session, data: List):
        uniques = {el["oid"]: el for el in data}
        uniques = list(uniques.values())
        insert_stmt = insert(Xmatch).values(uniques)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="xmatch_pkey",
            set_=dict(
                oid_catalog=insert_stmt.excluded.oid_catalog,
                dist=insert_stmt.excluded.dist,
            ),
        )

        return session.connection().execute(insert_stmt)
