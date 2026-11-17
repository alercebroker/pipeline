"""lsst survey: constants, preprocessor, extractor, AstroObject building, scribe and output."""
from lc_classifier.features.core.base import AstroObject  # object type
import pandas as pd  # references table type

from features.surveys.base import Survey  # survey contract
from features.surveys.lsst.astro_object import detections_to_astro_object_lsst  # sources -> AstroObject
from features.surveys.lsst.bands import LSST_BANDS  # lsst band table
from features.surveys.lsst.output import parse_output_lsst  # output messages
from features.surveys.lsst.scribe import parse_scribe_payload_lsst  # scribe commands


class LSSTSurvey(Survey):
    """LSST survey definition."""

    name = "lsst"  # survey name
    sid = 1  # survey id
    tid = 1  # type id
    needs_references = False  # lsst does not read references
    bands = LSST_BANDS  # u 6, g 1, r 2, i 3, z 4, y 5; pairs 61 12 23 34 45

    def make_preprocessor(self):
        """Return the lsst lightcurve preprocessor."""
        from lc_classifier.features.preprocess.lsst import LSSTLightcurvePreprocessor  # lazy heavy import

        return LSSTLightcurvePreprocessor()  # same arguments as step.py today

    def make_extractor(self):
        """Return the lsst feature extractor."""
        from lc_classifier.features.composites.lsst import LSSTFeatureExtractor  # lazy heavy import

        return LSSTFeatureExtractor()  # same arguments as step.py today

    def epochs_from_message(self, message: dict) -> list[dict]:
        """Return sources and previous sources, keeping the rows whose band is not None."""
        dets = message.get('sources', []) + message.get('previous_sources', [])  # alert sources, then earlier sources
        dets = [elem for elem in dets if elem.get('band') is not None]  # drop the rows without band
        return dets  # same filter as pre_execute today

    def forced_from_message(self, message: dict) -> list[dict] | None:
        """Return the message's forced_sources unchanged, None when the message has none."""
        return message.get("forced_sources", None)  # passed through as execute does today

    def build_astro_object(  # signature spans several lines
        self,  # survey instance
        detections: list[dict],  # epochs from epochs_from_message, with index_column
        forced: list[dict] | None,  # forced sources from forced_from_message
        xmatches: dict | None,  # allwise match or None
        references_db: pd.DataFrame | None,  # unused: lsst reads no references
    ) -> AstroObject:  # returns the lsst AstroObject
        """Build the AstroObject from the sources, the forced sources and the xmatch, as execute does today."""
        return detections_to_astro_object_lsst(detections, forced, xmatches)  # moved parser, no references passed

    def scribe_commands(  # signature spans several lines
        self,  # survey instance
        astro_objects: list[AstroObject],  # objects with computed features
        features_version,  # version id
        features_group,  # extractor name, unused by lsst today
        feature_name_lut: dict,  # feature id -> name
    ) -> dict:  # returns the upserting_features list
        """Return the features scribe commands of the objects; lsst has no update-object command."""
        return parse_scribe_payload_lsst(astro_objects, features_version, features_group, feature_name_lut)  # moved body

    def output_messages(  # signature spans several lines
        self,  # survey instance
        astro_objects: list[AstroObject],  # objects with computed features
        messages: list[dict],  # messages as pre_execute left them, same order
        candids: dict,  # oid -> measurement ids accumulated by execute
    ) -> list[dict]:  # returns one output message per object
        """Return the output messages of the objects."""
        return parse_output_lsst(astro_objects, messages, candids)  # moved body
