"""Tests for the shared survey helpers in features/surveys/common.py.

The two "equals today" tests use the Task 1 golden fixtures as their oracle:
the stub feature rows and LUTs are imported from test_surveys_golden, and the
expected scribe feature lists are read from fixtures/golden_{ztf,lsst}.json.
"""
import json  # read the golden fixtures
import math  # check NaN results
import pathlib  # locate the fixtures dir
import unittest  # test case base class

import pandas as pd  # build detection and feature frames

from features.surveys.common import (  # helpers under test
    flatten_features_for_output,  # name + band suffix -> value dict for the output topic
    last_mjd_of,  # max mjd over detections and forced photometry
    prepare_features_for_db,  # feature rows for the scribe upsert
    wise_magnitudes,  # W1-W4 from an allwise xmatch
)  # end of the helper imports
from features.surveys.lsst import LSSTSurvey  # lsst band table
from features.surveys.ztf import ZTFSurvey  # ztf band table
from lc_classifier.features.core.base import AstroObject  # object the helpers read

from ..message_factory_ztf_ms import allwise_match  # allwise xmatch in the shape the step receives
from .test_surveys_golden import LUT, STUB_FEATURES, normalize  # Task 1 stub rows, LUTs and normalizer

FIXTURES = pathlib.Path(__file__).parent / "fixtures"  # where the golden json files live
INF = float("inf")  # infinite feature value
NAN = float("nan")  # missing feature value


def astro_object_with(rows):
    """Return a minimal AstroObject whose features are `rows` of (name, fid, value)."""
    detections = pd.DataFrame(  # two-row detections frame, the columns __post_init__ requires
        {"oid": [1, 1], "sid": [0, 0], "fid": ["g", "r"]}  # one object, two bands
    )  # end of the detections frame
    metadata = pd.DataFrame([["oid", 1]], columns=["name", "value"])  # metadata in the name/value layout
    ao = AstroObject(metadata=metadata, detections=detections)  # minimal object
    ao.features = pd.DataFrame(  # features in the extractor frame layout, as the golden stub writes them
        [(name, value, fid, 0, "stub") for name, fid, value in rows],  # reorder to name, value, fid
        columns=["name", "value", "fid", "sid", "version"],  # columns of empty_normal_dataframe
    )  # end of the features frame
    return ao  # ready for the helpers


def golden_features_command(survey):
    """Return the payload `features` list of the first "features" scribe command in golden_<survey>.json."""
    golden = json.loads((FIXTURES / f"golden_{survey}.json").read_text())  # whole fixture
    commands = [c for c in golden["scribe"] if c["value"]["step"] == "features"]  # features upserts only
    return commands[0]["value"]["payload"]["features"]  # first object's feature rows


class PrepareFeaturesForDbTestCase(unittest.TestCase):
    """prepare_features_for_db against the golden scribe payloads and its edge cases."""

    def test_prepare_keep_name_equals_today_ztf(self):
        """ztf rows with keep_name=True, minus name, equal the golden ztf features payload (inf -> None included)."""
        ao = astro_object_with(STUB_FEATURES["ztf"])  # stub rows include Skew=inf and Mean=NaN
        result = prepare_features_for_db(ao, LUT["ztf"], ZTFSurvey.bands, keep_name=True)  # helper under test
        got = normalize(result.drop(columns=["name"]).to_dict("records"))  # name is dropped before the upsert today
        self.assertEqual(got, golden_features_command("ztf"))  # same rows as today's scribe command
        self.assertIn({"value": None, "band": 2, "feature_id": 8}, got)  # Skew inf became None

    def test_prepare_keep_name_has_lut_names(self):
        """keep_name=True keeps a name column with the LUT spelling (Power_rate_1/4)."""
        ao = astro_object_with(STUB_FEATURES["ztf"])  # stub ztf rows
        result = prepare_features_for_db(ao, LUT["ztf"], ZTFSurvey.bands, keep_name=True)  # helper under test
        self.assertEqual(  # names after the backward-compat replace, unknown and NaN rows gone
            list(result["name"]),  # names in row order
            ["Amplitude", "Amplitude", "g-r_max", "g-r_mean", "g-r_max_corr",  # band and color features
             "g-r_mean_corr", "Power_rate_1/4", "Skew"],  # renamed and inf features
        )  # end of the names assertion
        self.assertTrue(set(result["name"]) <= set(LUT["ztf"].values()))  # every name is a LUT name

    def test_prepare_drop_name_equals_today_lsst(self):
        """lsst rows with keep_name=False equal the golden lsst features payload."""
        ao = astro_object_with(STUB_FEATURES["lsst"])  # stub lsst rows, no inf among them
        result = prepare_features_for_db(ao, LUT["lsst"], LSSTSurvey.bands, keep_name=False)  # helper under test
        self.assertNotIn("name", result.columns)  # name dropped
        self.assertNotIn("fid", result.columns)  # fid dropped
        got = normalize(result.to_dict("records"))  # json-safe rows
        self.assertEqual(got, golden_features_command("lsst"))  # same rows as today's scribe command

    def test_prepare_drop_name_inf_becomes_none_lsst(self):
        """An inf value on the keep_name=False path becomes None, like the ztf path."""
        ao = astro_object_with([("Skew", "z", INF), ("Amplitude", "u,g", 1.0)])  # inf row plus a finite color row
        lut = {1: "Amplitude", 5: "Skew"}  # both names known
        result = prepare_features_for_db(ao, lut, LSSTSurvey.bands, keep_name=False)  # helper under test
        got = normalize(result.to_dict("records"))  # json-safe rows
        self.assertEqual(  # inf -> None, bands mapped with the lsst table
            got,  # rows from the helper
            [{"value": None, "band": 4, "feature_id": 5},  # z -> 4, inf -> None
             {"value": 1.0, "band": 61, "feature_id": 1}],  # u,g -> 61
        )  # end of the rows assertion

    def test_prepare_drops_unknown_with_warning(self):
        """A name absent from the LUT is dropped and logged on alerce.FeatureStep."""
        ao = astro_object_with(STUB_FEATURES["ztf"])  # contains Unknown_feat
        with self.assertLogs("alerce.FeatureStep", "WARNING") as logs:  # capture the warning
            result = prepare_features_for_db(ao, LUT["ztf"], ZTFSurvey.bands, keep_name=True)  # helper under test
        self.assertNotIn("Unknown_feat", list(result["name"]))  # unmapped row dropped
        self.assertEqual(len(logs.output), 1)  # one warning
        self.assertIn("['Unknown_feat']", logs.output[0])  # the warning names the dropped feature

    def test_prepare_empty_lut_gives_empty_frame(self):
        """With an empty LUT every row is unmapped and the result is empty."""
        ao = astro_object_with(STUB_FEATURES["lsst"])  # stub lsst rows
        with self.assertLogs("alerce.FeatureStep", "WARNING"):  # every name is unknown, so it warns
            result = prepare_features_for_db(ao, {}, LSSTSurvey.bands, keep_name=False)  # empty LUT
        self.assertEqual(len(result), 0)  # nothing left to upsert
        self.assertEqual(list(result.columns), ["value", "band", "feature_id"])  # same columns as a full result


class WiseMagnitudesTestCase(unittest.TestCase):
    """wise_magnitudes for missing, non-allwise and allwise xmatches."""

    def test_wise_none_and_other_catalog(self):
        """None and a non-allwise catalog give four NaN."""
        other = allwise_match(1)  # allwise-shaped match
        other["catalog"] = "gaia"  # but from another catalog
        for xmatches in (None, other):  # both cases
            result = wise_magnitudes(xmatches)  # helper under test
            self.assertEqual(len(result), 4)  # W1-W4
            self.assertTrue(all(math.isnan(w) for w in result))  # all NaN

    def test_wise_allwise(self):
        """An allwise match gives its four w<n>mpro values."""
        self.assertEqual(wise_magnitudes(allwise_match(1)), (15.1, 14.9, 12.5, 9.1))  # factory defaults


class LastMjdTestCase(unittest.TestCase):
    """last_mjd_of over detections and forced photometry."""

    def test_last_mjd(self):
        """Max over both lists; NaN when no entry has an mjd."""
        self.assertEqual(last_mjd_of([{"mjd": 1.0}], [{"mjd": 3.0}]), 3.0)  # forced holds the max
        self.assertTrue(math.isnan(last_mjd_of([], [])))  # nothing at all
        self.assertTrue(math.isnan(last_mjd_of([{"x": 1}], [])))  # entries without mjd


class FlattenFeaturesTestCase(unittest.TestCase):
    """flatten_features_for_output against today's parse_output bodies."""

    def test_flatten_ztf(self):
        """ztf: suffix per band, sorted by name, '-' -> '_', NaN and inf -> None."""
        ao = astro_object_with(STUB_FEATURES["ztf"])  # stub rows with NaN and inf
        result = flatten_features_for_output(ao, ZTFSurvey.bands)  # helper under test
        expected = {  # same dict today's parse_output builds (golden output features)
            "Amplitude_1": 1.5, "Amplitude_2": 2.5, "Mean_1": None,  # band features, NaN -> None
            "Power_rate_1_4": 0.7, "Skew_2": None, "Unknown_feat_1": 1.0,  # no band, inf -> None, unmapped kept
            "g_r_max_12": 0.3, "g_r_max_corr_12": 0.25, "g_r_mean_12": 0.2,  # colors renamed
            "g_r_mean_corr_12": 0.15,  # last color
        }  # end of the expected ztf dict
        self.assertEqual(result, expected)  # same keys and values
        self.assertEqual(list(result), list(expected))  # same order: sorted by name
        self.assertTrue(all(v is None or type(v) is float for v in result.values()))  # values are float or None

    def test_flatten_lsst(self):
        """lsst: u -> _6, u,g -> _61, NaN and inf -> None, unknown band -> no suffix."""
        rows = STUB_FEATURES["lsst"] + [("Skew", "y", INF), ("Odd", "q", 2.0)]  # add inf and an unknown band
        ao = astro_object_with(rows)  # stub rows plus the two extras
        result = flatten_features_for_output(ao, LSSTSurvey.bands)  # helper under test
        expected = {  # sorted by the suffixed name
            "Amplitude_1": 1.2, "Amplitude_6": 1.1, "Mean_3": None,  # g, u, NaN -> None
            "Odd": 2.0, "Power_rate_1_2": 0.6, "Skew_5": None,  # unknown band no suffix, inf -> None
            "Unknown_feat_2": 1.0, "u_g_mean_61": 0.4,  # unmapped kept, color renamed
        }  # end of the expected lsst dict
        self.assertEqual(result, expected)  # same keys and values
        self.assertEqual(list(result), list(expected))  # same order: sorted by name
