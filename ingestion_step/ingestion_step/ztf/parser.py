from typing import Any

import pandas as pd

from ingestion_step.core.parser_interface import ParsedData, ParserInterface
from ingestion_step.ztf import extractor
from ingestion_step.ztf.parsers.candidates import parse_candidates
from ingestion_step.ztf.parsers.fp_hists import parse_fp_hists
from ingestion_step.ztf.parsers.prv_candidates import parse_prv_candidates


class ZTFParser(ParserInterface):
    """
    Parser implementation for ZTF.
    """

    def parse(self, messages: list[dict[str, Any]]) -> ParsedData:
        """
        Parser method for ZTF.

        Extracts `candidates`, `prv_candidates` y `fp_hists`, then parses
        each of those into its respective `objects`, `detections`,
        `non_detections` and `forced_photometries`.
        """
        msg_data = extractor.extract(messages)

        objects, detections = parse_candidates(msg_data["candidates"])
        prv_detections, non_detections = parse_prv_candidates(
            msg_data["prv_candidates"]
        )
        forced_photometries = parse_fp_hists(msg_data["fp_hists"])

        return ParsedData(
            objects=objects,
            detections=pd.concat([detections, prv_detections]),
            non_detections=non_detections,
            forced_photometries=forced_photometries,
        )
