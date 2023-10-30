from typing import Dict, List

from apf.core.step import GenericStep, get_class
from utils.parse import format_detection


class MetadataStep(GenericStep):
    def __init__(self, config, **step_args):
        super().__init__(config=config, **step_args)
        cls = get_class(config["SCRIBE_PRODUCER_CONFIG"]["CLASS"])
        self.scribe_producer = cls(config["SCRIBE_PRODUCER_CONFIG"])

    def _format_detection(self, d: Dict):
        d = d.copy()
        extra_fields = d.pop("extra_fields")
        extra_fields.pop("fp_hists", {})
        extra_fields.pop("prv_candidates", {})
        return format_detection({**d, **extra_fields})

    def _write_metadata_into_db(self, result: List[Dict]):
        pass

    # Output format: [{oid: OID, ss: SS_DATA, ...}]
    def execute(self, messages: List[Dict]):
        catalogs = {"ps1": {}, "ss": {}}
        result = [self._format_detection(message, catalogs) for message in messages]
        return result

    def post_execute(self, result: List[Dict]):
        self._write_metadata_into_db(result)
        return []
