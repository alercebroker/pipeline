import copy
from abc import ABC, abstractmethod
from typing import Dict, List
from importlib.metadata import version
import logging

from db_plugins.db.sql.models import (
    Detection,
    Feature,
    ForcedPhotometry,
    MagStats,
    NonDetection,
    Object,
    Probability,
    Xmatch,
    Score,
)
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .commons import ValidCommands

step_version = version("scribe")


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
        logging.debug("Inserting %s objects", len(data))
        return session.connection().execute(
            insert(Object).values(data).on_conflict_do_nothing()
        )


class UpsertScoreCommand(Command):
    type = ValidCommands.upsert_score

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)

        data_keys = data.keys()

        if not "detector_name" in data_keys:
            raise ValueError(f"missing field detector_name")
        if not "detector_version" in data_keys:
            raise ValueError(f"missing field detector_version")
        if not "categories" in data_keys:
            raise ValueError(f"missing field categories")
        else:
            if len(data["categories"]) < 1:
                raise ValueError(f"Categories in data with no content")

    def _format_data(self, data):

        principal_list = []

        for cat_dict in data["categories"]:

            principal_list.append(
                {
                    "detector_name": data["detector_name"],
                    "oid": self.criteria["_id"],
                    "detector_version": data["detector_version"],
                    "category_name": cat_dict["name"],
                    "score": cat_dict["score"],
                }
            )

        return principal_list

    @staticmethod
    def db_operation(session: Session, data: List):
        logging.debug("Inserting %s objects", len(data))

        insert_stmt = insert(Score)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="score_pkey",
            set_=dict(
                score=insert_stmt.excluded.score,
            ),
        )
        return session.connection().execute(insert_stmt, data)


class UpdateObjectFromStatsCommand(Command):
    type = ValidCommands.update_object_from_stats
    valid_attributes = set(
        [
            "ndethist",
            "ncovhist",
            "mjdstarthist",
            "mjdendhist",
            "corrected",
            "stellar",
            "ndet",
            "g_r_max",
            "g_r_max_corr",
            "g_r_mean",
            "g_r_mean_corr",
            "meanra",
            "meandec",
            "sigmara",
            "sigmadec",
            "deltajd",
            "firstmjd",
            "lastmjd",
            "step_id_corr",
            "diffpos",
        ]
    )

    def _check_inputs(self, data, criteria):
        super()._check_inputs(data, criteria)

    def _format_data(self, data):

        if not set(data.keys()).issubset(self.valid_attributes):
            bad_inputs = set(data.keys()).difference(self.valid_attributes)
            logging.debug(f"Invalid keys provided {bad_inputs}")
            for k in bad_inputs:
                data.pop(k)

        return {"oid": self.criteria["oid"], **data}

    @staticmethod
    def db_operation(session: Session, data: List):
        # upsert_stmt = update(Object)
        logging.debug("Updating or inserting %s objects", len(data))
        return session.bulk_update_mappings(Object, data)


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
        new_data = copy.deepcopy(data)
        # rename some fields
        for k, v in field_mapping.items():
            new_data[v] = new_data.pop(k)
        # add fields from extra_fields
        for field in _extra_fields:
            if field in new_data["extra_fields"]:
                new_data[field] = new_data["extra_fields"][field]
        new_data = {k: v for k, v in new_data.items() if k not in exclude}
        new_data["step_id_corr"] = new_data.get("step_id_corr", step_version)
        new_data["parent_candid"] = (
            int(new_data["parent_candid"])
            if new_data["parent_candid"] != "None"
            else None
        )
        new_data["fid"] = fid_map[new_data["fid"]]
        return {**new_data, **self.criteria}

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["candid"], el["oid"]): el for el in data}
        unique = list(unique.values())
        logging.debug("Inserting %s detections", len(unique))
        stmt = insert(Detection)
        return session.execute(
            stmt.on_conflict_do_update(
                constraint="detection_pkey", set_=stmt.excluded
            ),
            unique,
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

        data = copy.deepcopy(data)
        extra_fields = data["extra_fields"]
        extra_fields.pop("brokerIngestTimestamp", "")
        extra_fields.pop("surveyPublishTimestamp", "")
        extra_fields.pop("parent_candid", "")
        extra_fields.pop("forcediffimfluxunc", "")
        new_data = {k: v for k, v in data.items() if k not in exclude}
        new_data["fid"] = fid_map[new_data["fid"]]

        return {**new_data, **extra_fields}
        super()._check_inputs(data, criteria)

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["pid"], el["oid"]): el for el in data}
        unique = list(unique.values())
        logging.debug("Inserting %s forced photometry", len(unique))
        statement = insert(ForcedPhotometry)
        return session.execute(
            statement.on_conflict_do_update(
                constraint="forced_photometry_pkey", set_=statement.excluded
            ),
            unique,
        )


class UpdateObjectStatsCommand(Command):
    type = ValidCommands.update_object_stats

    def _check_inputs(self, data, criteria):
        if "magstats" not in data:
            raise ValueError("Magstats not provided in the commands data")

    def _format_data(self, data):
        fid_map = {"g": 1, "r": 2, "i": 3}
        magstats = data.pop("magstats")
        data["oid"] = self.criteria["_id"]
        for magstat in magstats:
            magstat.pop("sid")
            magstat["oid"] = self.criteria["_id"]
            magstat["fid"] = fid_map[magstat["fid"]]
            magstat["stellar"] = bool(magstat.get("stellar"))
            if "step_id_corr" not in magstat:
                magstat["step_id_corr"] = step_version

        return (data, magstats)

    @staticmethod
    def db_operation(session: Session, data: List):
        # data should be a tuple where idx 0 is objstats and 1 is magstats
        objstats, magstats = map(list, zip(*data))
        logging.debug("Updating object stats")
        for stat in objstats:
            oid = stat.pop("oid")
            update_stmt = update(Object).where(Object.oid == oid)
            session.execute(update_stmt, stat)
        logging.debug("Insert magstats")
        magstats = sum(magstats, [])
        unique_magstats = {(el["oid"], el["fid"]): el for el in magstats}
        unique_magstats = list(unique_magstats.values())
        upsert_stmt = insert(MagStats)
        upsert_stmt = upsert_stmt.on_conflict_do_update(
            constraint="magstat_pkey", set_=upsert_stmt.excluded
        )
        return session.execute(upsert_stmt, unique_magstats)


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
        logging.debug("Updating or inserting %s non detections", len(unique))
        insert_stmt = insert(NonDetection)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="non_detection_pkey",
            set_=dict(diffmaglim=insert_stmt.excluded.diffmaglim),
        )
        return session.execute(insert_stmt, unique)


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
        FID_MAP = {None: 0, "": 0, "g": 1, "r": 2, "gr": 12, "rg": 12}
        return [
            {
                **feat,
                "version": data["features_version"],
                "oid": self.criteria["_id"],
                "fid": FID_MAP[feat["fid"]]
                if isinstance(feat["fid"], str) or feat["fid"] is None
                else feat["fid"],
            }
            for feat in data["features"]
        ]

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {(el["oid"], el["name"], el["fid"]): el for el in data}
        unique = list(unique.values())
        logging.debug("Upserting %s features", len(unique))
        insert_stmt = insert(Feature)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="feature_pkey",
            set_=dict(value=insert_stmt.excluded.value),
        )

        return session.execute(insert_stmt, unique)


class UpsertProbabilitiesCommand(Command):
    type = ValidCommands.upsert_probabilities

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
        return [{**el, "oid": self.criteria["_id"]} for el in parsed]

    @staticmethod
    def db_operation(session: Session, data: List):
        unique = {
            (el["oid"], el["classifier_name"], el["class_name"]): el
            for el in data
        }
        unique = list(unique.values())
        logging.debug("Upserting %s probabilities", len(unique))
        insert_stmt = insert(Probability)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="probability_pkey",
            set_=dict(
                ranking=insert_stmt.excluded.ranking,
                probability=insert_stmt.excluded.probability,
            ),
        )

        return session.execute(insert_stmt, unique)


class UpsertXmatchCommand(Command):
    type = ValidCommands.upsert_xmatch

    def _format_data(self, data):
        formatted_data = []
        for catalog in data["xmatch"]:
            catalog_data = {
                "oid": self.criteria["_id"],
                "catid": catalog,
                "oid_catalog": data["xmatch"][catalog]["catoid"],
                "dist": data["xmatch"][catalog]["dist"],
            }
            formatted_data.append(catalog_data)
        return formatted_data

    @staticmethod
    def db_operation(session: Session, data: list):
        unique = {(d["oid"], d["catid"]): d for d in data}
        unique = list(unique.values())
        insert_stmt = insert(Xmatch)
        insert_stmt = insert_stmt.on_conflict_do_update(
            constraint="xmatch_pkey",
            set_=dict(
                oid_catalog=insert_stmt.excluded.oid_catalog,
                dist=insert_stmt.excluded.dist,
            ),
        )
        logging.debug("Upserting %s xmatches", len(unique))
        return session.execute(insert_stmt, unique)
