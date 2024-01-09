from typing import Dict, List, Tuple

from apf.core.step import GenericStep
from metadata_step.utils.parse import format_detection
from metadata_step.utils.database import (
    PSQLConnection,
    insert_metadata,
    get_gaia_catalog,
    get_ps1_catalog,
)


class MetadataStep(GenericStep):
    def __init__(self, config, db_sql: PSQLConnection, **step_args):
        super().__init__(config=config, **step_args)
        self.db = db_sql

    def _format_detection(self, d: Dict, catalogs: Dict):
        FID = {"g": 1, "r": 2}
        d = d.copy()
        d["fid"] = FID[d["fid"]]
        extra_fields = d.pop("extra_fields")
        extra_fields.pop("fp_hists", {})
        extra_fields.pop("prv_candidates", {})
        return format_detection({**d, **extra_fields}, catalogs)

    def _write_metadata_into_db(self, result: List[Dict], ps1: Dict[str, List]):
        ps1_updates = []
        flattened = sum(ps1.values(), [])
        for el in flattened:
            if el["updated"]:
                ps1_updates.append(el)

        with self.db.session() as session:
            insert_metadata(session, result, ps1_updates)

    # Output format: [{oid: OID, ss: SS_DATA, ...}]
    def execute(self, messages: List[Dict]):
        unique = {message["oid"]: message for message in messages}
        oids = list(unique.keys())
        messages = list(unique.values())
        catalogs = {"ps1": {}, "gaia": {}}
        with self.db.session() as session:
            catalogs["ps1"] = get_ps1_catalog(session, oids)
            catalogs["gaia"] = get_gaia_catalog(session, oids)

        result = [self._format_detection(message, catalogs) for message in messages]
        return result, catalogs["ps1"]

    def post_execute(self, result: Tuple[List[Dict], List[Dict]]):
        data, ps1 = result
        self._write_metadata_into_db(data, ps1)
        return []
