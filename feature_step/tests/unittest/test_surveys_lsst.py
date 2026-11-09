"""Tests for the lsst survey package in features/surveys/lsst.

The oracle is the Task 1 golden fixture fixtures/golden_lsst.json: the
AstroObject frames as the parser returned them, the scribe commands and the
output messages. Messages, stub features, LUT and the normalizer are imported
from test_surveys_golden so both tests build the same inputs.

The split dicts carry index, columns and values but no dtypes, so the dtypes
have their own oracle, fixtures/golden_lsst_dtypes.json: column dtypes, index
dtype and index names of the same frames, recorded from the pre-move parser
(features/utils/parsers.py at c9bc3feb8) at the same point, before preprocessing.
"""
import json  # read the fixture and round-trip scribe commands like the step does
import pathlib  # locate the fixtures dir
import unittest  # test case base class

import pandas as pd  # feature frames with padded bands

from features.surveys.common import prepare_features_for_db  # shared feature rows for the db
from features.surveys.lsst import LSSTSurvey  # survey under test
from features.surveys.ztf import ZTFSurvey  # other caller of the shared helpers, must stay as it is

from ..message_factory_ztf_ms import allwise_match  # allwise-shaped xmatch dict
from .test_surveys_golden import (  # Task 1 inputs and normalizer
    FRAME_NAMES,  # detections, forced_photometry, metadata, reference
    LUT,  # feature id -> name tables
    StubExtractor,  # writes STUB_FEATURES into each AstroObject
    messages_for,  # the golden input batch
    normalize,  # NaN -> "NaN", numpy -> python, frames -> split dicts
    references_df,  # one-row reference table for the ztf guard cases
)  # end of the golden imports

FIXTURES = pathlib.Path(__file__).parent / "fixtures"  # where the golden json files live

PADDED_FEATURES = [  # (name, fid, value) rows whose band strings carry whitespace, plus the 0 cases
    ("Amplitude", " g ", 1.0),  # padded on both sides
    ("Amplitude", "g ", 2.0),  # padded on the right
    ("u-g_mean", " u,g", 3.0),  # padded band pair
    ("Amplitude", "g", 4.0),  # plain band
    ("Power_rate_1_2", None, 0.5),  # no band
    ("Amplitude", "other", 6.0),  # unknown band
    ("Amplitude", 1, 7.0),  # int band
]  # end of the padded rows
PADDED_DB_ROWS = [  # scribe feature rows fid_mapper_for_db_lsst gave for PADDED_FEATURES at c9bc3feb8
    {"band": 1, "feature_id": 1, "value": 1.0},  # " g " stripped to g
    {"band": 1, "feature_id": 1, "value": 2.0},  # "g " stripped to g
    {"band": 61, "feature_id": 2, "value": 3.0},  # " u,g" stripped to u,g
    {"band": 1, "feature_id": 1, "value": 4.0},  # g
    {"band": 0, "feature_id": 3, "value": 0.5},  # None band
    {"band": 0, "feature_id": 1, "value": 6.0},  # unknown band
    {"band": 0, "feature_id": 1, "value": 7.0},  # int band, str "1" is no band
]  # end of the expected rows


def golden():
    """Return the parsed golden_lsst.json fixture."""
    return json.loads((FIXTURES / "golden_lsst.json").read_text())  # whole fixture


def epochs_as_execute_stamps(message):
    """Return the message's epochs as execute hands them to the parser: filtered, with index_column."""
    epochs = LSSTSurvey().epochs_from_message(message)  # pre_execute's filtering
    return [  # execute stamps "<measurement_id>_<oid>" on each epoch
        {**epoch, "index_column": f"{epoch['measurement_id']}_{epoch['oid']}"}  # new dict, original untouched
        for epoch in epochs  # every kept epoch
    ]  # end of the stamped list


def build(message):
    """Return the AstroObject LSSTSurvey builds for `message`, forced sources and xmatch taken as execute takes them."""
    survey = LSSTSurvey()  # survey under test
    epochs = epochs_as_execute_stamps(message)  # what execute passes in
    forced = survey.forced_from_message(message)  # forced_sources as received
    return survey.build_astro_object(epochs, forced, message.get("xmatches"), None)  # lsst reads no references


def frames_of(ao):
    """Return the four AstroObject frames as the golden RecordingPreprocessor stored them."""
    return normalize({  # same normalization the fixture went through
        name: (None if getattr(ao, name) is None else getattr(ao, name).to_dict("split"))  # None stays null
        for name in FRAME_NAMES  # detections, forced_photometry, metadata, reference
    })  # end of the frames dict


def golden_dtypes():
    """Return the parsed golden_lsst_dtypes.json fixture: frame dtypes recorded from the pre-move parser."""
    return json.loads((FIXTURES / "golden_lsst_dtypes.json").read_text())  # whole dtype fixture


def dtypes_of(ao):
    """Return, per AstroObject frame, the column dtypes in column order, the index dtype and the index names."""
    described = {}  # frame name -> description, None for an absent frame
    for name in FRAME_NAMES:  # detections, forced_photometry, metadata, reference
        frame = getattr(ao, name)  # frame as the parser returns it
        described[name] = None if frame is None else {  # None stays null, as in the fixture
            "columns": [[str(column), str(dtype)] for column, dtype in frame.dtypes.items()],  # ordered [name, dtype] pairs
            "index": {"dtype": str(frame.index.dtype), "names": list(frame.index.names)},  # index dtype and level names
        }  # end of the frame description
    return described  # same layout as golden_lsst_dtypes.json


class BuildAstroObjectTestCase(unittest.TestCase):
    """LSSTSurvey.build_astro_object against the golden AstroObject frames."""

    def setUp(self):
        """Load the fixture and show full diffs."""
        self.golden = golden()  # oracle
        self.maxDiff = None  # whole diff on mismatch

    def test_build_astro_object_equals_golden(self):
        """Message with an allwise xmatch and forced sources gives the golden frames."""
        message = messages_for("lsst")[0]  # generate_message_lsst(seed=7, with_xmatch=True)
        got = frames_of(build(message))  # frames as the parser returns them
        self.assertEqual(got, self.golden["astro_objects"]["1152921504606847001"])  # equals today's frames
        self.assertIsNone(got["reference"])  # lsst builds no reference frame

    def test_build_astro_object_no_xmatch_equals_golden(self):
        """Message without xmatch gives the golden frames (W1-W4 NaN)."""
        message = messages_for("lsst")[1]  # generate_message_lsst(oid=...002, seed=8)
        got = frames_of(build(message))  # frames as the parser returns them
        self.assertEqual(got, self.golden["astro_objects"]["1152921504606847002"])  # equals today's frames
        metadata = dict(got["metadata"]["data"])  # name -> value
        self.assertEqual([metadata[w] for w in ("W1", "W2", "W3", "W4")], ["NaN"] * 4)  # no allwise match


class BuildAstroObjectDtypesTestCase(unittest.TestCase):
    """LSSTSurvey.build_astro_object dtypes against the ones the pre-move parser produced."""

    def setUp(self):
        """Load the dtype fixture and show full diffs."""
        self.golden = golden_dtypes()  # oracle recorded from features/utils/parsers.py at c9bc3feb8
        self.maxDiff = None  # whole diff on mismatch

    def test_dtype_fixture_covers_the_golden_objects(self):
        """The dtype fixture describes the same oids and the same four frames as golden_lsst.json."""
        self.assertEqual(sorted(self.golden), sorted(golden()["astro_objects"]))  # same oids as the value oracle
        for oid, frames in self.golden.items():  # each recorded object
            self.assertEqual(sorted(frames), sorted(FRAME_NAMES), oid)  # all four frames described
            self.assertIsNone(frames["reference"], oid)  # the pre-move lsst parser returned no reference frame

    def test_build_astro_object_dtypes_equal_golden(self):
        """Message with an allwise xmatch and forced sources keeps the pre-move column dtypes, index dtype and index names."""
        message = messages_for("lsst")[0]  # generate_message_lsst(seed=7, with_xmatch=True)
        got = dtypes_of(build(message))  # dtypes of the frames as the parser returns them
        self.assertEqual(got, self.golden["1152921504606847001"])  # equals the pre-move dtypes

    def test_build_astro_object_no_xmatch_dtypes_equal_golden(self):
        """Message without xmatch keeps the pre-move dtypes, the all-NaN W1-W4 metadata included."""
        message = messages_for("lsst")[1]  # generate_message_lsst(oid=...002, seed=8)
        got = dtypes_of(build(message))  # dtypes of the frames as the parser returns them
        self.assertEqual(got, self.golden["1152921504606847002"])  # equals the pre-move dtypes


class EpochsTestCase(unittest.TestCase):
    """LSSTSurvey.epochs_from_message and forced_from_message."""

    def test_epochs_drops_no_band(self):
        """The source with band None is gone; every other source and previous source stays, in order."""
        message = messages_for("lsst")[0]  # sources[1] has band None
        n_sources = len(message["sources"])  # 5 alert sources
        n_previous = len(message["previous_sources"])  # 3 previous sources
        no_band = [s["measurement_id"] for s in message["sources"] if s["band"] is None]  # ids without band
        self.assertEqual(len(no_band), 1)  # the factory pins exactly one
        epochs = LSSTSurvey().epochs_from_message(message)  # survey under test
        self.assertTrue(all(e["band"] is not None for e in epochs))  # no epoch without band left
        self.assertNotIn(no_band[0], [e["measurement_id"] for e in epochs])  # the band None source is gone
        self.assertEqual(len(epochs), n_sources - 1 + n_previous)  # everything else is kept
        kept = [s for s in message["sources"] + message["previous_sources"] if s["band"] is not None]  # today's filter
        self.assertEqual(epochs, kept)  # sources first, then previous sources, forced sources excluded

    def test_epochs_from_message_tolerates_absent_arrays(self):
        """An absent sources or previous_sources key counts as empty, as pre_execute's `.get(key, [])` does."""
        message = messages_for("lsst")[0]  # good message
        del message["previous_sources"]  # absent array
        epochs = LSSTSurvey().epochs_from_message(message)  # survey under test
        self.assertEqual(len(epochs), len(message["sources"]) - 1)  # only the sources with a band remain
        self.assertEqual(LSSTSurvey().epochs_from_message({}), [])  # no arrays at all gives no epochs

    def test_forced_from_message_passes_forced_sources_through(self):
        """forced_from_message returns the message's forced_sources object itself, unchanged."""
        message = messages_for("lsst")[0]  # message with 4 forced sources
        forced = LSSTSurvey().forced_from_message(message)  # survey under test
        self.assertIs(forced, message["forced_sources"])  # same list, not a copy
        self.assertEqual(len(forced), 4)  # the factory's n_forced

    def test_forced_from_message_is_none_without_forced_sources(self):
        """A message without forced_sources gives None, as execute's `message.get("forced_sources", None)` does today."""
        message = messages_for("lsst")[0]  # good message
        del message["forced_sources"]  # absent array
        self.assertIsNone(LSSTSurvey().forced_from_message(message))  # None passed through, not []


class ScribeAndOutputTestCase(unittest.TestCase):
    """LSSTSurvey.scribe_commands and output_messages against the golden scribe and output."""

    def setUp(self):
        """Build the first golden object with stub features and keep the message as pre_execute left it."""
        self.golden = golden()  # oracle
        self.maxDiff = None  # whole diff on mismatch
        message = messages_for("lsst")[0]  # first golden message
        self.oid = message["oid"]  # 1152921504606847001
        self.ao = build(message)  # AstroObject as the parser returns it
        StubExtractor("lsst").compute_features_batch([self.ao])  # stub features, as the golden run
        self.prepared = {**message, "detections": LSSTSurvey().epochs_from_message(message)}  # pre_execute's message
        self.candids = {self.oid: list(message["measurement_id"])}  # execute's candids accumulation

    def golden_scribe(self, step):
        """Return the golden scribe value of `step` for this oid."""
        return next(  # the only matching command
            c["value"] for c in self.golden["scribe"]  # every recorded command
            if c["value"]["step"] == step and c["key"] == str(self.oid)  # this step, this oid
        )  # end of the lookup

    def test_scribe_commands_equal_golden(self):
        """scribe_commands gives the golden features command and no update-object command."""
        commands = LSSTSurvey().scribe_commands([self.ao], 1, "LSSTFeatureExtractor", LUT["lsst"])  # survey under test
        as_sent = lambda cmd: normalize(json.loads(json.dumps(cmd)))  # the step json-encodes each command
        self.assertEqual(list(commands), ["upserting_features"])  # lsst sends no update_object
        self.assertEqual(len(commands["upserting_features"]), 1)  # one command per object
        self.assertEqual(as_sent(commands["upserting_features"][0]), self.golden_scribe("features"))  # feature rows

    def test_output_messages_equal_golden(self):
        """output_messages gives the golden output message with the lsst keys."""
        out = LSSTSurvey().output_messages([self.ao], [self.prepared], self.candids)  # survey under test
        self.assertEqual(len(out), 1)  # one message per object
        self.assertEqual(normalize(out[0]), self.golden["output"][0])  # golden first output message
        self.assertEqual(sorted(out[0]), ["detections", "features", "measurement_id", "oid"])  # keys today


class ReexportTestCase(unittest.TestCase):
    """features.utils.parsers keeps exporting the moved names."""

    def test_parsers_reexports_moved_names(self):
        """Every moved lsst name is importable from parsers and is the moved object itself."""
        from features.surveys.lsst import astro_object, output, scribe  # new homes
        from features.utils import parsers  # old home, still imported by step.py and tests

        moved = {  # name -> new module
            "flux_err_2_mag_err": astro_object,  # flux error -> magnitude error
            "fluxnjy2mag": astro_object,  # nJy flux -> AB magnitude
            "add_mag_and_flux_lsst": astro_object,  # lsst magnitude and flux rows
            "detections_to_astro_object_lsst": astro_object,  # the lsst parser
            "fid_mapper_for_db_lsst": scribe,  # band -> db code
            "parse_scribe_payload_lsst": scribe,  # scribe commands
            "parse_output_lsst": output,  # output messages
        }  # end of the moved names
        for name, module in moved.items():  # each moved name
            self.assertIs(getattr(parsers, name), getattr(module, name), name)  # same object in both places

    def test_parsers_keeps_the_prepare_shims(self):
        """The two prepare_ao_features_for_db names existing tests import are still in parsers."""
        from features.utils import parsers  # old home

        self.assertTrue(callable(parsers.prepare_ao_features_for_db))  # ztf shim still there
        self.assertTrue(callable(parsers.prepare_ao_features_for_db_lsst))  # lsst shim still there

    def test_fid_mapper_for_db_lsst_is_band_table(self):
        """fid_mapper_for_db_lsst keeps today's codes, the None and unknown 0, and the strip of the band string."""
        from features.surveys.lsst.scribe import fid_mapper_for_db_lsst  # moved function under test

        singles = [fid_mapper_for_db_lsst(b) for b in ("u", "g", "r", "i", "z", "y")]  # single bands
        self.assertEqual(singles, [6, 1, 2, 3, 4, 5])  # today's single codes
        pairs = [fid_mapper_for_db_lsst(b) for b in ("u,g", "g,r", "r,i", "i,z", "z,y")]  # band pairs
        self.assertEqual(pairs, [61, 12, 23, 34, 45])  # today's pair codes
        self.assertEqual([fid_mapper_for_db_lsst(b) for b in (None, "other", 1)], [0, 0, 0])  # missing, unknown, int
        self.assertEqual(fid_mapper_for_db_lsst(" g "), 1)  # today's body strips the band string


def with_padded_features(ao):
    """Replace the features of `ao` with PADDED_FEATURES, in the extractor frame layout, and return it."""
    sid = ao.detections["sid"].values[0]  # same sid the scribe payload reads
    ao.features = pd.DataFrame(  # padded rows in the extractor frame layout
        [(name, value, fid, sid, "stub") for name, fid, value in PADDED_FEATURES],  # reorder to name, value, fid
        columns=["name", "value", "fid", "sid", "version"],  # columns of empty_normal_dataframe
    )  # end of the features frame
    return ao  # same object, features replaced


def build_with_xmatch(message, xmatches):
    """Return the AstroObject LSSTSurvey builds for `message` with `xmatches` in place of the message's own."""
    survey = LSSTSurvey()  # survey under test
    epochs = epochs_as_execute_stamps(message)  # what execute passes in
    return survey.build_astro_object(epochs, survey.forced_from_message(message), xmatches, None)  # explicit xmatch


def wise_of(ao):
    """Return W1-W4 of an AstroObject's metadata, NaN as "NaN"."""
    metadata = dict(zip(ao.metadata["name"], ao.metadata["value"]))  # name -> value
    return normalize([metadata[w] for w in ("W1", "W2", "W3", "W4")])  # comparable list


class PaddedBandTestCase(unittest.TestCase):
    """lsst scribe write path strips the band string, as fid_mapper_for_db_lsst did on multisurvey (c9bc3feb8)."""

    def setUp(self):
        """Build the first golden object with the padded-band features."""
        self.maxDiff = None  # whole diff on mismatch
        message = messages_for("lsst")[0]  # first golden message
        self.oid = message["oid"]  # 1152921504606847001
        self.ao = with_padded_features(build(message))  # AstroObject with padded bands
        self.prepared = {**message, "detections": LSSTSurvey().epochs_from_message(message)}  # pre_execute's message
        self.candids = {self.oid: list(message["measurement_id"])}  # execute's candids accumulation

    def test_scribe_command_strips_padded_bands(self):
        """scribe_commands sends band 1, 1, 61 for fid " g ", "g ", " u,g" and 0 for None, unknown and int bands."""
        commands = LSSTSurvey().scribe_commands([self.ao], 1, "LSSTFeatureExtractor", LUT["lsst"])  # survey under test
        features = json.loads(json.dumps(commands["upserting_features"][0]["payload"]["features"]))  # as sent
        self.assertEqual([f["band"] for f in features[:3]], [1, 1, 61])  # the three padded bands
        self.assertEqual(features, PADDED_DB_ROWS)  # every row as on multisurvey

    def test_prepare_features_for_db_with_lsst_bands_strips_padded_bands(self):
        """Spec criterion 7 on padded bands: keep_name=False with the lsst band table equals the multisurvey lsst rows."""
        result = prepare_features_for_db(self.ao, LUT["lsst"], LSSTSurvey.bands, keep_name=False)  # four-argument call
        self.assertEqual(result.to_dict("records"), PADDED_DB_ROWS)  # same rows as prepare_ao_features_for_db_lsst gave

    def test_parsers_shim_strips_padded_bands(self):
        """parsers.prepare_ao_features_for_db_lsst, still imported by other tests, gives the multisurvey rows too."""
        from features.utils import parsers  # old home of the lsst helper

        result = parsers.prepare_ao_features_for_db_lsst(self.ao, LUT["lsst"])  # shim under test
        self.assertEqual(result.to_dict("records"), PADDED_DB_ROWS)  # same rows as on multisurvey

    def test_band_table_fid_for_db_is_the_multisurvey_mapper(self):
        """LSSTSurvey.bands.fid_for_db and fid_mapper_for_db_lsst agree on padded, missing, unknown and unhashable bands."""
        from features.surveys.lsst.scribe import fid_mapper_for_db_lsst  # moved function

        cases = {" g ": 1, "g ": 1, " u,g": 61, " g,r": 12, "g": 1, None: 0, "other": 0, 1: 0, float("nan"): 0}  # band -> code
        for band, code in cases.items():  # each band
            self.assertEqual(LSSTSurvey.bands.fid_for_db(band), code, repr(band))  # band table
            self.assertEqual(fid_mapper_for_db_lsst(band), code, repr(band))  # function name kept by parsers
        self.assertEqual(LSSTSurvey.bands.fid_for_db(["g"]), 0)  # str(["g"]) is no band; multisurvey did not raise

    def test_output_does_not_strip_padded_bands(self):
        """output_messages adds no suffix to a padded band, as the multisurvey suffix_map did, and values are floats."""
        out = LSSTSurvey().output_messages([self.ao], [self.prepared], self.candids)  # survey under test
        self.assertEqual(  # features measured at c9bc3feb8 for the same rows
            out[0]["features"],  # name + suffix -> value
            {"Amplitude": 7.0, "Amplitude_1": 4.0, "Power_rate_1_2": 0.5, "u_g_mean": 3.0},  # only plain g has a suffix
        )  # end of the features assertion
        self.assertEqual({type(v) for v in out[0]["features"].values()}, {float})  # python floats, as on multisurvey
        self.assertEqual(LSSTSurvey.bands.suffix(" g "), "")  # the band table strips for the db code only


class XmatchCatalogTestCase(unittest.TestCase):
    """lsst parser requires the xmatch `catalog` key, as the multisurvey body did (c9bc3feb8)."""

    def setUp(self):
        """Keep the first golden message and an allwise-shaped match for it."""
        self.message = messages_for("lsst")[0]  # first golden message
        self.allwise = allwise_match(self.message["oid"])  # catalog, distance, match_id, metadata

    def test_xmatch_without_catalog_raises_key_error(self):
        """An allwise-shaped xmatch without `catalog` raises KeyError('catalog')."""
        no_catalog = {k: v for k, v in self.allwise.items() if k != "catalog"}  # same shape, key missing
        with self.assertRaises(KeyError) as raised:  # the multisurvey body indexed xmatches['catalog']
            build_with_xmatch(self.message, no_catalog)  # survey under test
        self.assertEqual(raised.exception.args, ("catalog",))  # the missing key is catalog

    def test_empty_xmatch_raises_key_error(self):
        """An empty xmatch dict raises KeyError('catalog'): {} is not None."""
        with self.assertRaises(KeyError) as raised:  # the multisurvey body indexed xmatches['catalog']
            build_with_xmatch(self.message, {})  # survey under test
        self.assertEqual(raised.exception.args, ("catalog",))  # the missing key is catalog

    def test_xmatch_none_gives_nan(self):
        """xmatches=None gives W1-W4 NaN and no error."""
        self.assertEqual(wise_of(build_with_xmatch(self.message, None)), ["NaN"] * 4)  # no match attached

    def test_xmatch_from_another_catalog_gives_nan(self):
        """An xmatch whose catalog is not allwise gives W1-W4 NaN and no error."""
        other = {**self.allwise, "catalog": "other"}  # same shape, other catalog
        self.assertEqual(wise_of(build_with_xmatch(self.message, other)), ["NaN"] * 4)  # not read

    def test_allwise_xmatch_gives_the_magnitudes(self):
        """An allwise xmatch gives its w<n>mpro values."""
        self.assertEqual(wise_of(build_with_xmatch(self.message, self.allwise)), [15.1, 14.9, 12.5, 9.1])  # factory defaults


class ZtfUnchangedTestCase(unittest.TestCase):
    """ztf keeps its own multisurvey behaviour on the same inputs: no strip, no KeyError."""

    def test_ztf_band_mapper_does_not_strip(self):
        """ztf fid_for_db gives 0 for a padded band, as the multisurvey fid_mapper_for_db did."""
        got = [ZTFSurvey.bands.fid_for_db(b) for b in (" g ", "g ", " g,r", "g", None)]  # padded, plain, missing
        self.assertEqual(got, [0, 0, 0, 1, 0])  # only the exact string has a code

    def test_ztf_xmatch_without_catalog_gives_nan(self):
        """ztf build_astro_object gives W1-W4 NaN for an xmatch without `catalog` and for {}, as on multisurvey."""
        message = messages_for("ztf")[0]  # first ztf golden message
        survey = ZTFSurvey()  # the other caller of wise_magnitudes
        epochs = [  # execute stamps "<measurement_id>_<oid>" on each epoch
            {**epoch, "index_column": f"{epoch['measurement_id']}_{epoch['oid']}"}  # new dict, original untouched
            for epoch in survey.epochs_from_message(message)  # pre_execute's filtering
        ]  # end of the stamped list
        no_catalog = {k: v for k, v in allwise_match(message["oid"]).items() if k != "catalog"}  # key missing
        for xmatches in (no_catalog, {}):  # both cases that raise for lsst
            ao = survey.build_astro_object(epochs, [], xmatches, references_df())  # ztf parser
            self.assertEqual(wise_of(ao), ["NaN"] * 4)  # ztf used .get: no error, no magnitudes
