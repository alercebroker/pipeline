import copy
from abc import ABC, abstractmethod
from typing import Dict, List
from importlib.metadata import version
import logging


## En el import desde los models se deben traer todos los modelos que se modififiquen en los steps. 
## Al menos desde correction hacia atras.
from db_plugins.db.sql.models import (
    Object,
    Detection,
    ZtfDetection,
    ForcedPhotometry,
    ZtfForcedPhotometry,
    ZtfSS,
    ZtfPS1,
    ZtfGaia,
    ZtfDataquality,
    ZtfReference,
    MagStat,
    LsstDiaObject,
)
from sqlalchemy import update, bindparam
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from .parser import (
    parse_det,
    parse_ztf_det,
    parse_ztf_gaia,
    parse_fp,
    parse_ztf_fp,
    parse_ztf_ps1,
    parse_ztf_ss,
    parse_ztf_dq,
    parse_ztf_refernece,
    parse_obj_stats,
    parse_magstats,
    parse_dia_object,
)


step_version = version("scribe-multisurvey")


class Command(ABC):
    type: str

    def __init__(self, payload, criteria=None, options=None):
        self._check_inputs(payload, criteria)
        self.criteria = criteria or {}
        self.options = options
        self.data = self._format_data(payload)

    def _format_data(self, data):
        return data

    def _check_inputs(self, data, criteria):
        if not data:
            raise ValueError("Not data provided in command")

    @staticmethod
    @abstractmethod
    def db_operation(session: Session, data: List):
        pass

class ZTFCorrectionCommand(Command):
    type = "ZTFCorrectionCommand"

    def _format_data(self, data):
        """
        Generate a dictionary with  data for the tables:f
        ztf_detection
        ztf_forced photometry
        ps1_ztf
        ss_ztf
        gaia_ztf
        """

        oid = data["oid"]
        candidate_measurement_id = data["measurement_id"][0]

        # detections
        candidate = {}        
        ## potential issue, deduplicate candidates inside detections and forced
        detections = []
        ztf_detections = []
        forced_photometries = []
        ztf_forced_photometries = []

        for detection in data["detections"]:
            if detection["new"]:
                # forced photometry
                # correct! if detection["forced"]: 
                if detection["extra_fields"].get("forcediffimflux", None):
                    forced_photometries.append(parse_fp(detection, oid))
                    ztf_forced_photometries.append(parse_ztf_fp(detection, oid))
                # detection
                else:
                    if detection["measurement_id"] == candidate_measurement_id:
                        candidate = detection
                    detections.append(parse_det(detection, oid))
                    ztf_detections.append(parse_ztf_det(detection, oid))
        


        parsed_ps1 = parse_ztf_ps1(candidate, oid)
        parsed_ss = parse_ztf_ss(candidate, oid)
        parsed_gaia = parse_ztf_gaia(candidate, oid)
        parsed_dq = parse_ztf_dq(candidate, oid)
        parsed_reference = parse_ztf_refernece(candidate, oid)

        return {
            "detections": detections,
            "ztf_detections": ztf_detections,
            "forced_photometries": forced_photometries,
            "ztf_forced_photometries": ztf_forced_photometries,
            "ps1": parsed_ps1,
            "ss": parsed_ss,
            "gaia": parsed_gaia,
            "dq": parsed_dq,
            "reference": parsed_reference,
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        # forget deduplication!!!! :D
        detections = []
        ztf_detections = []
        forced_pothometries = []
        ztf_forced_pothometries = []
        ps1 = []
        ss = []
        gaia = []
        dq = []
        reference = []

        for single_data in data:
            detections.extend(single_data["detections"])
            ztf_detections.extend(single_data["ztf_detections"])
            forced_pothometries.extend(single_data["forced_photometries"])
            ztf_forced_pothometries.extend(single_data["ztf_forced_photometries"])
            ps1.append(single_data["ps1"])
            ss.append(single_data["ss"])
            gaia.append(single_data["gaia"])
            dq.append(single_data["dq"])
            reference.append(single_data["reference"])            
        
        # insert detections
        if len(detections) > 0:
            detections_stmt = insert(Detection)
            detections_result = session.connection().execute(
                detections_stmt.on_conflict_do_update(
                    constraint="pk_detection_oid_measurementid",
                    set_=detections_stmt.excluded
                ),
                detections
            )
        
        if len(ztf_detections) > 0:
            ztf_detections_stmt = insert(ZtfDetection)
            ztf_detections_result = session.connection().execute(
                ztf_detections_stmt.on_conflict_do_update(
                    constraint="pk_ztfdetection_oid_measurementid",
                    set_=ztf_detections_stmt.excluded
                ),
                ztf_detections
            )

        # insert forced photometry
        if len(forced_pothometries) > 0:
            fp_stmt = insert(ForcedPhotometry)
            fp_result = session.connection().execute(
                fp_stmt.on_conflict_do_update(
                    constraint="pk_forcedphotometry_oid_measurementid",
                    set_=fp_stmt.excluded
                ),
                forced_pothometries
            )

        if len(ztf_forced_pothometries) > 0:
            ztf_fp_stmt = insert(ZtfForcedPhotometry)
            ztf_fp_result = session.connection().execute(
                ztf_fp_stmt.on_conflict_do_update(
                    constraint="pk_ztfforcedphotometry_oid_measurementid",
                    set_=ztf_fp_stmt.excluded
                ),
                ztf_forced_pothometries
            )

        # insert ps1_ztf
        if len(ps1) > 0:
            ps_stmt = insert(ZtfPS1)
            ps_result = session.connection().execute(
                ps_stmt.on_conflict_do_update(
                    constraint="pk_ztfps1_oid_measurement_id",
                    set_=ps_stmt.excluded
                ),
                ps1
            )

        # insert ss_ztf
        if len(ss) > 0:
            ss_stmt = insert(ZtfSS)
            ss_result = session.connection().execute(
                ss_stmt.on_conflict_do_update(
                    constraint="pk_ztfss_oid_measurement_id",
                    set_=ss_stmt.excluded
                ),
                ss
            )

        # insert gaia_ztf
        if len(gaia) > 0:
            gaia_stmt = insert(ZtfGaia)
            gaia_result = session.connection().execute(
                gaia_stmt.on_conflict_do_update(
                    constraint="pk_ztfgaia_oid",
                    set_=gaia_stmt.excluded
                ),
                gaia
            )

        # insert dataquaility_ztf
        if len(dq) > 0:
            dq_stmt = insert(ZtfDataquality)
            dq_result = session.connection().execute(
                dq_stmt.on_conflict_do_update(
                    constraint="pk_ztfdataqualit_oid_measurement_id",
                    set_=dq_stmt.excluded
                ),
                dq
            )

        # insert reference_ztf
        if len(reference) > 0:
            reference_stmt = insert(ZtfReference)
            reference_result = session.connection().execute(
                reference_stmt.on_conflict_do_update(
                    constraint="pk_ztfreference_oid_rfid",
                    set_=reference_stmt.excluded
                ),
                reference
            )


class ZTFMagstatCommand(Command):
    type = "ZTFMagstatCommand"

    def _format_data(self, data):
        
        oid = data["oid"]

        object_stats = parse_obj_stats(data, oid)
        magstats_list = [
            parse_magstats(ms, oid)
            for ms in data["magstats"]
        ]

        return {
            "object_stats": object_stats,
            "magstats": magstats_list
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        objectstat_list = []
        magstat_list = []

        for single_data in data:
            objectstat_list.append(single_data["object_stats"])
            magstat_list.extend(single_data["magstats"])      
        
        # update object
        if len(objectstat_list) > 0:
            object_stmt = update(Object)
            object_result = session.connection().execute(
                object_stmt
                .where(Object.oid == bindparam('_oid'))
                .values({
                    "meanra": bindparam("meanra"),
                    "meandec": bindparam("meandec"),
                    "sigmara": bindparam("sigmara"),
                    "sigmadec": bindparam("sigmadec"),
                    "firstmjd": bindparam("firstmjd"),
                    "lastmjd": bindparam("lastmjd"),
                    "deltamjd": bindparam("deltamjd"),
                    "n_det": bindparam("n_det"),
                    "n_forced": bindparam("n_forced"),
                    "n_non_det": bindparam("n_non_det"),
                    "corrected": bindparam("corrected"),
                    "stellar": bindparam("stellar"),
                }),
                objectstat_list
            )
        
        # insert magstats
        if len(magstat_list) > 0:
            magstats_stmt = insert(MagStat)
            magstats_result = session.connection().execute(
                magstats_stmt.on_conflict_do_update(
                    constraint="pk_magstat_oid_band",
                    set_=magstats_stmt.excluded
                ),
                magstat_list
            )


class LSSTMagstatCommand(Command):
    type = "LSSTMagstatCommand"

    def _format_data(self, data):
        print('wiwiwi2')
        print(data)
        oid = data["oid"]

        object_stats = parse_obj_stats(data, oid)
        magstats_list = []

        return {
            "object_stats": object_stats,
            "magstats": magstats_list
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        objectstat_list = []
        magstat_list = []

        for single_data in data:
            objectstat_list.append(single_data["object_stats"])
            magstat_list.extend(single_data["magstats"])      
        
        # update object
        if len(objectstat_list) > 0:
            object_stmt = update(Object)
            object_result = session.connection().execute(
                object_stmt
                .where(Object.oid == bindparam('_oid'))
                .values({
                    "meanra": bindparam("meanra"),
                    "meandec": bindparam("meandec"),
                    "sigmara": bindparam("sigmara"),
                    "sigmadec": bindparam("sigmadec"),
                    "firstmjd": bindparam("firstmjd"),
                    "lastmjd": bindparam("lastmjd"),
                    "deltamjd": bindparam("deltamjd"),
                    "n_det": bindparam("n_det"),
                    "n_forced": bindparam("n_forced"),
                    "n_non_det": bindparam("n_non_det"),
                    "corrected": bindparam("corrected"),
                    "stellar": bindparam("stellar"),
                }),
                objectstat_list
            )


class LSSTUpdateDiaObjectCommand(Command):
    type = "LSSTUpdateDiaObjectCommand"

    def _format_data(self, data):
        
        oid = data["oid"]

        dia_object = parse_dia_object(data, oid)
        

        return {
            "dia_object": dia_object,
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        objects_update_list = []

        for single_data in data:
            objects_update_list.append(single_data["dia_object"])
        
        # update dia object
        if len(objects_update_list) > 0:
            object_stmt = update(LsstDiaObject)
            object_result = session.connection().execute(
                object_stmt
                .where(LsstDiaObject.oid == bindparam('_oid'))
                .values({
                    "validityStartMjdTai": bindparam("validityStartMjdTai"),
                    "ra": bindparam("ra"),
                    "raErr": bindparam("raErr"),
                    "dec": bindparam("dec"),
                    "decErr": bindparam("decErr"),
                    "ra_dec_Cov": bindparam("ra_dec_Cov"),
                    "u_psfFluxMean": bindparam("u_psfFluxMean"),
                    "u_psfFluxMeanErr": bindparam("u_psfFluxMeanErr"),
                    "u_psfFluxSigma": bindparam("u_psfFluxSigma"),
                    "u_psfFluxNdata": bindparam("u_psfFluxNdata"),
                    "u_fpFluxMean": bindparam("u_fpFluxMean"),
                    "u_fpFluxMeanErr": bindparam("u_fpFluxMeanErr"),
                    "g_psfFluxMean": bindparam("g_psfFluxMean"),
                    "g_psfFluxMeanErr": bindparam("g_psfFluxMeanErr"),
                    "g_psfFluxSigma": bindparam("g_psfFluxSigma"),
                    "g_psfFluxNdata": bindparam("g_psfFluxNdata"),
                    "g_fpFluxMean": bindparam("g_fpFluxMean"),
                    "g_fpFluxMeanErr": bindparam("g_fpFluxMeanErr"),
                    "r_psfFluxMean": bindparam("r_psfFluxMean"),
                    "r_psfFluxMeanErr": bindparam("r_psfFluxMeanErr"),
                    "r_psfFluxSigma": bindparam("r_psfFluxSigma"),
                    "r_psfFluxNdata": bindparam("r_psfFluxNdata"),
                    "r_fpFluxMean": bindparam("r_fpFluxMean"),
                    "r_fpFluxMeanErr": bindparam("r_fpFluxMeanErr"),
                    "i_psfFluxMean": bindparam("i_psfFluxMean"),
                    "i_psfFluxMeanErr": bindparam("i_psfFluxMeanErr"),
                    "i_psfFluxSigma": bindparam("i_psfFluxSigma"),
                    "i_psfFluxNdata": bindparam("i_psfFluxNdata"),
                    "i_fpFluxMean": bindparam("i_fpFluxMean"),
                    "i_fpFluxMeanErr": bindparam("i_fpFluxMeanErr"),
                    "z_psfFluxMean": bindparam("z_psfFluxMean"),
                    "z_psfFluxMeanErr": bindparam("z_psfFluxMeanErr"),
                    "z_psfFluxSigma": bindparam("z_psfFluxSigma"),
                    "z_psfFluxNdata": bindparam("z_psfFluxNdata"),
                    "z_fpFluxMean": bindparam("z_fpFluxMean"),
                    "z_fpFluxMeanErr": bindparam("z_fpFluxMeanErr"),
                    "y_psfFluxMean": bindparam("y_psfFluxMean"),
                    "y_psfFluxMeanErr": bindparam("y_psfFluxMeanErr"),
                    "y_psfFluxSigma": bindparam("y_psfFluxSigma"),
                    "y_psfFluxNdata": bindparam("y_psfFluxNdata"),
                    "y_fpFluxMean": bindparam("y_fpFluxMean"),
                    "y_fpFluxMeanErr": bindparam("y_fpFluxMeanErr"),
                    "u_scienceFluxMean": bindparam("u_scienceFluxMean"),
                    "u_scienceFluxMeanErr": bindparam("u_scienceFluxMeanErr"),
                    "g_scienceFluxMean": bindparam("g_scienceFluxMean"),
                    "g_scienceFluxMeanErr": bindparam("g_scienceFluxMeanErr"),
                    "r_scienceFluxMean": bindparam("r_scienceFluxMean"),
                    "r_scienceFluxMeanErr": bindparam("r_scienceFluxMeanErr"),
                    "i_scienceFluxMean": bindparam("i_scienceFluxMean"),
                    "i_scienceFluxMeanErr": bindparam("i_scienceFluxMeanErr"),
                    "z_scienceFluxMean": bindparam("z_scienceFluxMean"),
                    "z_scienceFluxMeanErr": bindparam("z_scienceFluxMeanErr"),
                    "y_scienceFluxMean": bindparam("y_scienceFluxMean"),
                    "y_scienceFluxMeanErr": bindparam("y_scienceFluxMeanErr"),
                    "u_psfFluxMin": bindparam("u_psfFluxMin"),
                    "u_psfFluxMax": bindparam("u_psfFluxMax"),
                    "u_psfFluxMaxSlope": bindparam("u_psfFluxMaxSlope"),
                    "u_psfFluxErrMean": bindparam("u_psfFluxErrMean"),
                    "g_psfFluxMin": bindparam("g_psfFluxMin"),
                    "g_psfFluxMax": bindparam("g_psfFluxMax"),
                    "g_psfFluxMaxSlope": bindparam("g_psfFluxMaxSlope"),
                    "g_psfFluxErrMean": bindparam("g_psfFluxErrMean"),
                    "r_psfFluxMin": bindparam("r_psfFluxMin"),
                    "r_psfFluxMax": bindparam("r_psfFluxMax"),
                    "r_psfFluxMaxSlope": bindparam("r_psfFluxMaxSlope"),
                    "r_psfFluxErrMean": bindparam("r_psfFluxErrMean"),
                    "i_psfFluxMin": bindparam("i_psfFluxMin"),
                    "i_psfFluxMax": bindparam("i_psfFluxMax"),
                    "i_psfFluxMaxSlope": bindparam("i_psfFluxMaxSlope"),
                    "i_psfFluxErrMean": bindparam("i_psfFluxErrMean"),
                    "z_psfFluxMin": bindparam("z_psfFluxMin"),
                    "z_psfFluxMax": bindparam("z_psfFluxMax"),
                    "z_psfFluxMaxSlope": bindparam("z_psfFluxMaxSlope"),
                    "z_psfFluxErrMean": bindparam("z_psfFluxErrMean"),
                    "y_psfFluxMin": bindparam("y_psfFluxMin"),
                    "y_psfFluxMax": bindparam("y_psfFluxMax"),
                    "y_psfFluxMaxSlope": bindparam("y_psfFluxMaxSlope"),
                    "y_psfFluxErrMean": bindparam("y_psfFluxErrMean"),
                    "firstDiaSourceMjdTai": bindparam("firstDiaSourceMjdTai"),
                    "lastDiaSourceMjdTai": bindparam("lastDiaSourceMjdTai"),
                    "nDiaSources": bindparam("nDiaSources"),
                }),
                objects_update_list
            )


### ###### ###
### LEGACY ###
### ###### ###
"""
class ZtfCorrectionCommand(Command):
    pass

class InsertObjectCommand(Command):

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
  /commands.py", line 155, in Upsert      upsert_stmt = upsert_stmt.on_conflict_do_update(
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
"""
