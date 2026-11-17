"""ztf survey: constants, preprocessor, extractor, AstroObject building, scribe and output."""
from lc_classifier.features.core.base import AstroObject, discard_bogus_detections  # object type and rb filter
import pandas as pd  # references table type

from features.surveys.base import Survey  # survey contract
from features.surveys.ztf.astro_object import detections_to_astro_object  # epochs -> AstroObject
from features.surveys.ztf.bands import ZTF_BANDS  # ztf band table
from features.surveys.ztf.output import parse_output  # output messages
from features.surveys.ztf.scribe import parse_scribe_payload  # scribe commands


class ZTFSurvey(Survey):
    """ZTF survey definition."""

    name = "ztf"  # survey name
    sid = 0  # survey id
    tid = 0  # type id
    needs_references = True  # ztf reads references from the db
    bands = ZTF_BANDS  # g 1, r 2, g,r 12; i only has a letter

    def make_preprocessor(self):
        """Return the ztf lightcurve preprocessor."""
        from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor  # lazy heavy import

        return ZTFLightcurvePreprocessor(drop_bogus=True)  # same arguments as step.py today

    def make_extractor(self):
        """Return the ztf feature extractor."""
        from lc_classifier.features.composites.ztf import ZTFFeatureExtractor  # lazy heavy import

        return ZTFFeatureExtractor()  # same arguments as step.py today

    def epochs_from_message(self, message: dict) -> list[dict]:
        """Return detections, previous detections and forced photometry with bogus epochs discarded."""
        epochs = (  # the three epoch arrays, None counted as empty
            (message.get("detections") or [])  # alert detections
            + (message.get("previous_detections") or [])  # earlier detections
            + (message.get("forced_photometries") or [])  # forced photometry, flagged inline
        )  # end of the concatenation
        return discard_bogus_detections(epochs)  # same filter as pre_execute today

    def forced_from_message(self, message: dict) -> list[dict]:
        """Return []: ztf forced epochs flow inline in the epochs with a per-row forced flag."""
        return []  # detections_to_astro_object requires it empty

    def build_astro_object(  # signature spans several lines
        self,  # survey instance
        detections: list[dict],  # epochs from epochs_from_message, with index_column
        forced: list[dict],  # must be empty for ztf
        xmatches: dict | None,  # allwise match or None
        references_db: pd.DataFrame | None,  # references read from the db
    ) -> AstroObject:  # returns the ztf AstroObject
        """Stamp aid = oid on every epoch, as execute does today, then build the AstroObject."""
        # No ZTF multisurvey record has an `aid`; the parser indexes on it.
        detections = [{**det, "aid": det["oid"]} for det in detections]  # new dicts, input untouched
        return detections_to_astro_object(detections, forced, xmatches, references_db)  # moved parser

    def scribe_commands(  # signature spans several lines
        self,  # survey instance
        astro_objects: list[AstroObject],  # objects with computed features
        features_version,  # version id
        features_group,  # extractor name, unused by ztf today
        feature_name_lut: dict,  # feature id -> name
    ) -> dict:  # returns update_object and upserting_features lists
        """Return the update-object and features scribe commands of the objects."""
        return parse_scribe_payload(astro_objects, features_version, features_group, feature_name_lut)  # moved body

    def output_messages(  # signature spans several lines
        self,  # survey instance
        astro_objects: list[AstroObject],  # objects with computed features
        messages: list[dict],  # messages as pre_execute left them, same order
        candids: dict,  # oid -> candids accumulated by execute
    ) -> list[dict]:  # returns one output message per object
        """Return the output messages of the objects."""
        return parse_output(astro_objects, messages, candids)  # moved body
