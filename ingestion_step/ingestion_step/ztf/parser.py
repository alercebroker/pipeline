from typing import Any

import pandas as pd

from ingestion_step.core.parsed_data import ParsedData
from ingestion_step.core.parser_interface import ParserInterface
from ingestion_step.ztf import extractor
from ingestion_step.ztf.parsers.candidates import parse_candidates
from ingestion_step.ztf.parsers.fp_hist import parse_fp_hist
from ingestion_step.ztf.parsers.prv_candidates import parse_prv_candidates


class ZTFParser(ParserInterface):
    def parse(self, messages: list[dict[str, Any]]) -> ParsedData:
        # Separate messages into candidates, prv_candidates and fp_hist
        msg_data = extractor.extract(messages)

        # Parse the separated data into objs, dets, non_dets, and fps
        # each split into common and survey specific data frames.
        objects, detections = parse_candidates(msg_data["candidates"])
        prv_detections, non_detections = parse_prv_candidates(
            msg_data["prv_candidates"]
        )
        forced_photometries = parse_fp_hist(msg_data["fp_hist"])

        return ParsedData(
            objects=objects,
            detections=pd.concat([detections, prv_detections]),
            non_detections=non_detections,
            forced_photometries=forced_photometries,
        )
