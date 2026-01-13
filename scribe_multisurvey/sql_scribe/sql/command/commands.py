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
    ZtfObject,
    ZtfSS,
    ZtfPS1,
    ZtfGaia,
    ZtfDataquality,
    ZtfReference,
    MagStat,
    LsstDiaObject,
    Feature,
    LsstMpcOrbits,
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
    parse_ztf_object,
    parse_ztf_magstats,
    parse_ztf_objstats
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
        reference_ztf
        """

        oid = data["oid"]
        candidate_measurement_id = data["measurement_id"][0]

        candidate = {}        
        ## TODO check potential issue, multiple candidates in the same message ? => Whats the solution?
        detections = []
        ztf_detections = []
        
        for detection in data["detections"]:
            if detection["new"]:
                if detection["measurement_id"] == candidate_measurement_id:
                    candidate = detection
                detections.append(parse_det(detection, oid))
                ztf_detections.append(parse_ztf_det(detection, oid))
        
        for detection in data["previous_detections"]:
            if detection["new"]:
                detections.append(parse_det(detection, oid))
                ztf_detections.append(parse_ztf_det(detection, oid))

        fp_dict = {}
        ztf_fp_dict = {}
        for forced in data["forced_photometries"]:
            if forced["new"]:
                fp = parse_fp(forced, oid)
                ztf_fp = parse_ztf_fp(forced, oid)
                
                key = (fp['oid'], fp['measurement_id'], fp['sid'])
                ztf_key = (ztf_fp['oid'], ztf_fp['measurement_id'])
                
                if key not in fp_dict or fp['mjd'] > fp_dict[key]['mjd']:
                    fp_dict[key] = fp
                if ztf_key not in ztf_fp_dict or ztf_fp['mjd'] > ztf_fp_dict[ztf_key]['mjd']:
                    ztf_fp_dict[ztf_key] = ztf_fp

        parsed_ps1 = parse_ztf_ps1(candidate, oid)
        parsed_ss = parse_ztf_ss(candidate, oid)
        parsed_gaia = parse_ztf_gaia(candidate, oid)
        parsed_dq = parse_ztf_dq(candidate, oid)
        parsed_reference = parse_ztf_refernece(candidate, oid)
        parsed_ztf_obj = parse_ztf_object(candidate, oid)

        return {
            "detections": detections,
            "ztf_detections": ztf_detections,
            "forced_photometries": list(fp_dict.values()),
            "ztf_forced_photometries": list(ztf_fp_dict.values()),
            "ps1": parsed_ps1,
            "ss": parsed_ss,
            "gaia": parsed_gaia,
            "dq": parsed_dq,
            "reference": parsed_reference,
            "ztf_object": parsed_ztf_obj
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        detections = []
        ztf_detections = []
        forced_photometries = []
        ztf_forced_photometries = []
        ps1 = []
        ss = []
        gaia = []
        dq = []
        reference = []
        ztf_obj_update = []

        for single_data in data:
            detections.extend(single_data["detections"])
            ztf_detections.extend(single_data["ztf_detections"])
            forced_photometries.extend(single_data["forced_photometries"])
            ztf_forced_photometries.extend(single_data["ztf_forced_photometries"])
            ps1.append(single_data["ps1"])
            ss.append(single_data["ss"])
            gaia.append(single_data["gaia"])
            dq.append(single_data["dq"])
            reference.append(single_data["reference"])
            ztf_obj_update.append(single_data["ztf_object"])            
        
        fp_dedup_dict = {}
        for fp, ztf_fp in zip(forced_photometries, ztf_forced_photometries):
            key = (fp['oid'], fp['measurement_id'], fp['sid'])
            if key not in fp_dedup_dict or fp['mjd'] > fp_dedup_dict[key]['fp']['mjd']:
                fp_dedup_dict[key] = {'fp': fp, 'ztf_fp': ztf_fp}
    
        forced_photometries = [item['fp'] for item in fp_dedup_dict.values()]
        ztf_forced_photometries = [item['ztf_fp'] for item in fp_dedup_dict.values()]
    
        det_dedup_dict = {}
        for det, ztf_det in zip(detections, ztf_detections):
            key = (det['oid'], det['measurement_id'], det['sid'])
            if key not in det_dedup_dict or det['mjd'] > det_dedup_dict[key]['det']['mjd']:
                det_dedup_dict[key] = {'det': det, 'ztf_det': ztf_det}
    
        detections = [item['det'] for item in det_dedup_dict.values()]
        ztf_detections = [item['ztf_det'] for item in det_dedup_dict.values()]
    
        conn = session.connection()

        if detections:
            detections_stmt = insert(Detection).on_conflict_do_update(
                constraint="pk_detection_oid_measurementid_sid",
                set_=insert(Detection).excluded
            )
            conn.execute(detections_stmt, detections)
        
        if ztf_detections:
            ztf_detections_stmt = insert(ZtfDetection).on_conflict_do_update(
                constraint="pk_ztfdetection_oid_measurementid",
                set_=insert(ZtfDetection).excluded
            )
            conn.execute(ztf_detections_stmt, ztf_detections)

        if forced_photometries:
            fp_stmt = insert(ForcedPhotometry).on_conflict_do_update(
                constraint="pk_forcedphotometry_oid_measurementid_sid",
                set_=insert(ForcedPhotometry).excluded
            )
            conn.execute(fp_stmt, forced_photometries)

        if ztf_forced_photometries:
            ztf_fp_stmt = insert(ZtfForcedPhotometry).on_conflict_do_update(
                constraint="pk_ztfforcedphotometry_oid_measurementid",
                set_=insert(ZtfForcedPhotometry).excluded
            )
            conn.execute(ztf_fp_stmt, ztf_forced_photometries)

        ztfobj_dict = {}
        for obj in ztf_obj_update:
            key = obj['oid']
            mjd = obj.get('_mjd', 0)
            if key not in ztfobj_dict or mjd > ztfobj_dict[key][0]:
                ztfobj_dict[key] = (mjd, obj)
        
        if ztfobj_dict:
            ztfobj_list = []
            for mjd, obj in ztfobj_dict.values():
                clean_obj = {k: v for k, v in obj.items() if k != '_mjd'}
                clean_obj['_oid'] = clean_obj['oid']
                ztfobj_list.append(clean_obj)

            ztf_object_stmt = update(ZtfObject).where(
                ZtfObject.oid == bindparam('_oid')
            ).values({
                "ndethist": bindparam("ndethist"),
                "ncovhist": bindparam("ncovhist"),
                "mjdstarthist": bindparam("mjdstarthist"),
                "mjdendhist": bindparam("mjdendhist"),
            })
            conn.execute(ztf_object_stmt, ztfobj_list)

        ps1_dict = {}
        for p in ps1:
            key = (p['oid'], p.get('measurement_id'))
            mjd = p.get('_mjd', 0)
            if key not in ps1_dict or mjd > ps1_dict[key][0]:
                ps1_dict[key] = (mjd, {k: v for k, v in p.items() if k != '_mjd'})

        if ps1_dict:
            ps1_clean = [obj for mjd, obj in ps1_dict.values()]
            ps_stmt = insert(ZtfPS1).on_conflict_do_update(
                constraint="pk_ztfps1_oid_measurement_id",
                set_=insert(ZtfPS1).excluded
            )
            conn.execute(ps_stmt, ps1_clean)
    
        ss_dict = {}
        for s in ss:
            key = (s['oid'], s.get('measurement_id'))
            mjd = s.get('_mjd', 0)
            if key not in ss_dict or mjd > ss_dict[key][0]:
                ss_dict[key] = (mjd, {k: v for k, v in s.items() if k != '_mjd'})

        if ss_dict:
            ss_clean = [obj for mjd, obj in ss_dict.values()]
            ss_stmt = insert(ZtfSS).on_conflict_do_update(
                constraint="pk_ztfss_oid_measurement_id",
                set_=insert(ZtfSS).excluded
            )
            conn.execute(ss_stmt, ss_clean)
        
        gaia_dict = {}
        for g in gaia:
            key = g['oid']
            mjd = g.get('_mjd', 0)
            if key not in gaia_dict or mjd > gaia_dict[key][0]:
                gaia_dict[key] = (mjd, {k: v for k, v in g.items() if k != '_mjd'})

        if gaia_dict:
            gaia_clean = [obj for mjd, obj in gaia_dict.values()]
            gaia_stmt = insert(ZtfGaia).on_conflict_do_update(
                constraint="pk_ztfgaia_oid",
                set_=insert(ZtfGaia).excluded
            )
            conn.execute(gaia_stmt, gaia_clean)

        dq_dict = {}
        for d in dq:
            key = (d['oid'], d.get('measurement_id'))
            mjd = d.get('_mjd', 0)
            if key not in dq_dict or mjd > dq_dict[key][0]:
                dq_dict[key] = (mjd, {k: v for k, v in d.items() if k != '_mjd'})
            
        if dq_dict:
            dq_clean = [obj for mjd, obj in dq_dict.values()]
            dq_stmt = insert(ZtfDataquality).on_conflict_do_update(
                constraint="pk_ztfdataquality_oid_measurement_id",
                set_=insert(ZtfDataquality).excluded
            )
            conn.execute(dq_stmt, dq_clean)

        ref_dict = {}
        for r in reference:
            key = (r['oid'], r.get('rfid'))
            mjd = r.get('_mjd', 0)
            if key not in ref_dict or mjd > ref_dict[key][0]:
                ref_dict[key] = (mjd, {k: v for k, v in r.items() if k != '_mjd'})
 
        if ref_dict:
            reference_clean = [obj for mjd, obj in ref_dict.values()]
            reference_stmt = insert(ZtfReference).on_conflict_do_update(
                constraint="pk_ztfreference_oid_rfid",
                set_=insert(ZtfReference).excluded
            )
            conn.execute(reference_stmt, reference_clean)
            
class ZTFMagstatCommand(Command):
    type = "ZTFMagstatCommand"

    def _format_data(self, data):
        
        oid = data["oid"]
        sid = data["sid"]
        object_stats = parse_ztf_objstats(data, oid, sid)
        magstats_list = [
            parse_ztf_magstats(ms, oid, sid)
            for ms in data["magstats"]["0"]
        ]

        return {
            "object_stats": object_stats,
            "magstats": magstats_list
        }

    @staticmethod
    def db_operation(session: Session, data: List):
        objectstat_list = []
        dedup_magstats = {} # We must deduplicate magstats and since none of its keys is good we will use  objstats stuff

        for single_data in data:
            obj = single_data["object_stats"]
            objectstat_list.append(obj)

            lastmjd = obj["lastmjd"]

            for ms in single_data["magstats"]:
                key = (ms["oid"], ms["sid"], ms["band"])

                if key not in dedup_magstats:
                    dedup_magstats[key] = (lastmjd, ms)
                else:
                    prev_lastmjd, _ = dedup_magstats[key]
                    if lastmjd > prev_lastmjd:
                        dedup_magstats[key] = (lastmjd, ms)

        magstat_list = [ms for _, ms in dedup_magstats.values()]

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
                }),
                objectstat_list
            )
            
            # update ztf_object columns stellar and corrected
            ztf_object_stmt = update(ZtfObject)
            ztf_object_result = session.connection().execute(
                ztf_object_stmt
                .where(ZtfObject.oid == bindparam('_oid'))
                .values({
                    "corrected": bindparam("corrected"),
                    "stellar": bindparam("stellar"),
                    "reference_change": bindparam("reference_change"),
                    "diffpos": bindparam("diffpos"),
                }),
                objectstat_list
            )
        
        # insert magstats
        if len(magstat_list) > 0:
            magstats_stmt = insert(MagStat)
            magstats_result = session.connection().execute(
                magstats_stmt.on_conflict_do_update(
                    constraint="pk_magstat_oid_sid_band",
                    set_=magstats_stmt.excluded
                ),
                magstat_list
            )  




class LSSTMagstatCommand(Command):
    type = "LSSTMagstatCommand"

    def _format_data(self, data):
        oid = data["oid"]
        sid = data["sid"]
        object_stats = parse_obj_stats(data, oid, sid)
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
                }),
                objectstat_list
            )

class LSSTFeatureCommand(Command):
    type = "LSSTFeatureCommand"

    def _format_data(self, data):
        
        oid = data["oid"]
        sid = data["sid"]
        feature_version = data["features_version"].split('.')[0] if isinstance(data["features_version"], str) else data["features_version"]
        
        deduplication_dict = {}

        for feature in data["features"]:
            key = (oid, sid, feature["feature_id"], feature["band"])
            deduplication_dict[key] = {
                "oid": oid,
                "sid": sid,
                "feature_id": feature["feature_id"],
                "band": feature["band"],
                "version": feature_version,
                "value": feature["value"],
            }
        
        return list(deduplication_dict.values())

    @staticmethod
    def db_operation(session: Session, data: List):
        
        if len(data) > 0:
            dedup = {}
            for row in data:
                key = (row["oid"], row["sid"], row["feature_id"], row["band"])
                dedup[key] = row 

            
            deduplicated_data = list(dedup.values())

            features_stmt = insert(Feature).on_conflict_do_update(
                constraint="pk_feature_oid_featureid_band",
                set_=insert(Feature).excluded, 
            )

            session.connection().execute(features_stmt, deduplicated_data)


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
