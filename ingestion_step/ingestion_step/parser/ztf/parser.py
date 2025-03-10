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

    def build_objects(self, data: ExtractedData) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_cols = ["oid", "candid"]
        survey_cols = ["oid", "candid"]

        common_objects = data.candidates[common_cols]
        survey_objects = data.candidates[survey_cols]

        # Insert telescope ID. ZTF has only one telescope so we add 0 to all entries
        common_objects.insert(loc=-1, column="tid", value=0)

        # Insert survey ID. ZTF is sid=0
        common_objects.insert(loc=-1, column="sid", value=0)

        return common_objects, survey_objects

    def build_detections(
        self, data: ExtractedData
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_candidate_cols = ["oid", "candid", "ra", "dec", "fid"]
        survey_candidate_cols = [
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
            # "has_stamp",
            # "step_id_corr",
        ]

        common_candidate_dets = data.candidates[common_candidate_cols]
        survey_candidate_dets = data.candidates[survey_candidate_cols]

        common_prv_candidate_cols = common_candidate_cols
        survey_prv_candidate_cols = survey_candidate_cols + ["parent_candid"]

        common_prv_candidate_dets = data.prv_candidates[common_prv_candidate_cols]
        survey_prv_candidate_dets = data.prv_candidates[survey_prv_candidate_cols]

        survey_candidate_dets.insert(loc=-1, column="parent_candid", value=None)

        common_dets = pd.concat([common_candidate_dets, common_prv_candidate_dets])
        survey_dets = pd.concat([survey_candidate_dets, survey_prv_candidate_dets])

        return common_dets, survey_dets

    def build_non_detections(self, extracted_data: ExtractedData) -> pd.DataFrame:
        pass

    def build_forced_photometries(
        self, extracted_data: ExtractedData
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def parse(self, messages: list[dict[str, Any]]) -> None:
        extracted_data = self.extract(messages)

        self.common_objects, self.survey_objects = self.build_objects(extracted_data)

        self.common_detections, self.survey_detections = self.build_detections(
            extracted_data
        )

        self.survey_non_detections = self.build_non_detections(extracted_data)

        self.common_forced_photometries, self.survey_forced_photometries = (
            self.build_forced_photometries(extracted_data)
        )

        pass

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
