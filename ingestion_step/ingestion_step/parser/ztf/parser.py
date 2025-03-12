from typing import Any, NamedTuple

import pandas as pd

from ingestion_step.parser.core.parser_interface import ParserInterface


class ExtractedData(NamedTuple):
    candidates: pd.DataFrame
    prv_candidates: pd.DataFrame
    fp_hist: pd.DataFrame


class ZTFParser(ParserInterface):
    common_objects: pd.DataFrame
    survey_objects: pd.DataFrame

    common_detections: pd.DataFrame
    survey_detections: pd.DataFrame

    survey_non_detections: pd.DataFrame

    common_forced_photometries: pd.DataFrame
    survey_forced_photometries: pd.DataFrame

    def has_stamp(self, message: dict[str, Any]) -> bool:
        return (
            "cutoutScience" in message
            and "cutoutTemplate" in message
            and "cutoutDifference" in message
        )

    def extract(self, messages: list[dict[str, Any]]) -> ExtractedData:
        candidates = [
            {
                "oid": message["objectId"],
                "candid": message["candid"],
                "has_stamp": self.has_stamp(message),
                **message["candidate"],
            }
            for message in messages
        ]

        prv_candidates = [
            {
                "oid": message["objectId"],
                "parent_candid": message["candid"],
                **prv_candidate,
            }
            for message in messages
            for prv_candidate in message["prv_candidates"]
        ]

        fp_hist = [
            {"oid": message["objectId"], "candid": message["candid"], **fp_hist}
            for message in messages
            for fp_hist in message["fp_hists"]
        ]

        return ExtractedData(
            candidates=pd.DataFrame(candidates),
            prv_candidates=pd.DataFrame(prv_candidates),
            fp_hist=pd.DataFrame(fp_hist),
        )

    def process_candidates(
        self, candidates: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Extract relevant columns for common and survey specific dataframes for objects
        common_obj_cols = ["oid", "candid", "ra", "dec", "jd"]
        survey_obj_cols = ["oid", "candid"]

        common_objs = candidates[common_obj_cols]
        survey_objs = candidates[survey_obj_cols]

        # Insert telescope ID. ZTF has only one telescope so we add 0 to all entries
        common_objs.insert(loc=-1, column="tid", value=0)

        # Insert survey ID. ZTF is sid=0
        common_objs.insert(loc=-1, column="sid", value=0)

        # TODO: oid -> int based oid (instead of str)
        # TODO: jd -> mjd

        # Extract relevant columns for common and survey specific dataframes for detections
        common_det_cols = ["oid", "candid", "ra", "dec", "fid", "jd"]
        survey_det_cols = [
            "oid",
            "candid",
            "pid",
            "diffmaglim",
            "isdiffpos",
            "nid",
            "magpsf",
            "sigmapsf",
            "magap",
            "sigmagap",
            "distnr",
            "rb",
            "rbversion",
            "drb",
            "drbversion",
            "magapbig",
            "sigmagapbig",
            # "parent_candid",
            # "rband",
            # "magpsf_corr",
            # "sigmapsf_corr",
            # "sigmapsf_corr_ext",
            # "corrected",
            # "dubious",
            "has_stamp",
            # "step_id_corr",
        ]

        common_dets = candidates[common_det_cols]
        survey_dets = candidates[survey_det_cols]

        # TODO: oid -> int based oid (instead of str)
        # TODO: measurment_id
        # TODO: jd -> mjd
        # TODO: fid -> band

        return common_objs, survey_objs, common_dets, survey_dets

    def process_prv_candidates(
        self, prv_candidates: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Split prv_candidates into dets and non_dets
        det_prv_candidates = prv_candidates[prv_candidates["candid"].notnull()]
        non_det_prv_candidates = prv_candidates[prv_candidates["candid"].isnull()]

        # Extract relevant columns for common and survey specific dataframes for detections
        common_det_cols = ["oid", "candid", "ra", "dec", "fid", "jd"]
        survey_det_cols = [
            "oid",
            "candid",
            "pid",
            "diffmaglim",
            "isdiffpos",
            "nid",
            "magpsf",
            "sigmapsf",
            "magap",
            "sigmagap",
            "distnr",
            "rb",
            "rbversion",
            "drb",
            "drbversion",
            "magapbig",
            "sigmagapbig",
            "parent_candid",
            # "rband",
            # "magpsf_corr",
            # "sigmapsf_corr",
            # "sigmapsf_corr_ext",
            # "corrected",
            # "dubious",
            # "has_stamp",
            # "step_id_corr",
        ]

        common_dets = det_prv_candidates[common_det_cols]
        survey_dets = det_prv_candidates[survey_det_cols]

        # TODO: oid -> int based oid (instead of str)
        # TODO: measurment_id
        # TODO: jd -> mjd
        # TODO: fid -> band

        # Extract relevant columns for common and survey specific dataframes for non detections
        non_det_cols = ["oid", "fid", "jd", "diffmaglim"]

        non_dets = non_det_prv_candidates[non_det_cols]

        # TODO: oid -> int based oid (instead of str)
        # TODO: measurment_id
        # TODO: jd -> mjd

        return common_dets, survey_dets, non_dets

    def parse(self, messages: list[dict[str, Any]]) -> None:
        extracted_data = self.extract(messages)

        self.process_candidates(extracted_data.candidates)
        self.process_prv_candidates(extracted_data.prv_candidates)

    def get_common_objects(self) -> pd.DataFrame:
        return self.common_objects

    def get_survey_objects(self) -> pd.DataFrame:
        return self.survey_objects

    def get_common_detections(self) -> pd.DataFrame:
        return self.common_detections

    def get_survey_detections(self) -> pd.DataFrame:
        return self.survey_detections

    def get_survey_non_detections(self) -> pd.DataFrame:
        return self.survey_non_detections

    def get_common_forced_photometries(self) -> pd.DataFrame:
        return self.common_forced_photometries

    def get_survey_forced_photometries(self) -> pd.DataFrame:
        return self.survey_forced_photometries
