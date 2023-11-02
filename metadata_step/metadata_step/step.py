from typing import Dict, List

from apf.core.step import GenericStep, get_class
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
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])
        self.db = db_sql

    def _format_detection(self, d: Dict, catalogs: Dict):
        d = d.copy()
        extra_fields = d.pop("extra_fields")
        extra_fields.pop("fp_hists", {})
        extra_fields.pop("prv_candidates", {})
        return format_detection({**d, **extra_fields}, catalogs)

    def _write_metadata_into_db(self, result: List[Dict]):
        with self.db.session() as session:
            insert_metadata(session, result)

    # Output format: [{oid: OID, ss: SS_DATA, ...}]
    def execute(self, messages: List[Dict]):
        oids = list(set([message["oid"] for message in messages]))
        catalogs = {"ps1": {}, "gaia": {}}
        with self.db.session() as session:
            catalogs["ps1"] = get_ps1_catalog(session, oids)
            catalogs["gaia"] = get_gaia_catalog(session, oids)

        result = [self._format_detection(message, catalogs) for message in messages]
        return result

    def post_execute(self, result: List[Dict]):
        self._write_metadata_into_db(result)
        return []
