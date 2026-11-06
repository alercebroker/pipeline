"""Tests for the ztf survey package in features/surveys/ztf.

The oracle is the Task 1 golden fixture fixtures/golden_ztf.json: the
AstroObject frames as the parser returned them, the scribe commands and the
output messages. Messages, stub features, LUT, references and the normalizer
are imported from test_surveys_golden so both tests build the same inputs.
"""
import json  # read the fixture and round-trip scribe commands like the step does
import pathlib  # locate the fixtures dir
import unittest  # test case base class

from features.surveys.ztf import ZTFSurvey  # survey under test

from .test_surveys_golden import (  # Task 1 inputs and normalizer
    FRAME_NAMES,  # detections, forced_photometry, metadata, reference
    LUT,  # feature id -> name tables
    StubExtractor,  # writes STUB_FEATURES into each AstroObject
    messages_for,  # the golden input batch
    normalize,  # NaN -> "NaN", numpy -> python, frames -> split dicts
    references_df,  # one-row reference table the step's mocked query returns
)  # end of the golden imports

FIXTURES = pathlib.Path(__file__).parent / "fixtures"  # where the golden json files live


def golden():
    """Return the parsed golden_ztf.json fixture."""
    return json.loads((FIXTURES / "golden_ztf.json").read_text())  # whole fixture


def epochs_as_execute_stamps(message):
    """Return the message's epochs as execute hands them to the parser: filtered, with index_column."""
    epochs = ZTFSurvey().epochs_from_message(message)  # pre_execute's filtering
    return [  # execute stamps "<measurement_id>_<oid>" on each epoch
        {**epoch, "index_column": f"{epoch['measurement_id']}_{epoch['oid']}"}  # new dict, original untouched
        for epoch in epochs  # every kept epoch
    ]  # end of the stamped list


def build(message):
    """Return the AstroObject ZTFSurvey builds for `message` with the golden references table."""
    epochs = epochs_as_execute_stamps(message)  # what execute passes in
    return ZTFSurvey().build_astro_object(epochs, [], message.get("xmatches"), references_df())  # survey under test


def frames_of(ao):
    """Return the four AstroObject frames as the golden RecordingPreprocessor stored them."""
    return normalize({  # same normalization the fixture went through
        name: (None if getattr(ao, name) is None else getattr(ao, name).to_dict("split"))  # None stays null
        for name in FRAME_NAMES  # detections, forced_photometry, metadata, reference
    })  # end of the frames dict


class BuildAstroObjectTestCase(unittest.TestCase):
    """ZTFSurvey.build_astro_object against the golden AstroObject frames."""

    def setUp(self):
        """Load the fixture and show full diffs."""
        self.golden = golden()  # oracle
        self.maxDiff = None  # whole diff on mismatch

    def test_build_astro_object_equals_golden(self):
        """Message with an allwise xmatch and forced epochs gives the golden frames."""
        message = messages_for("ztf")[0]  # generate_message(seed=1, with_xmatch=True)
        got = frames_of(build(message))  # frames as the parser returns them
        self.assertEqual(got, self.golden["astro_objects"]["36028941624528297"])  # equals today's frames

    def test_build_astro_object_no_xmatch_no_forced_equals_golden(self):
        """Message without xmatch and without forced epochs gives the golden frames (W1-W4 NaN, forced empty)."""
        message = messages_for("ztf")[1]  # generate_message(oid=...298, seed=2, n_forced=0)
        got = frames_of(build(message))  # frames as the parser returns them
        self.assertEqual(got, self.golden["astro_objects"]["36028941624528298"])  # equals today's frames
        metadata = dict(got["metadata"]["data"])  # name -> value
        self.assertEqual([metadata[w] for w in ("W1", "W2", "W3", "W4")], ["NaN"] * 4)  # no allwise match
        self.assertEqual(got["forced_photometry"]["data"], [])  # no forced epochs


class EpochsTestCase(unittest.TestCase):
    """ZTFSurvey.epochs_from_message and forced_from_message."""

    def test_epochs_from_message_drops_bogus(self):
        """A message whose every epoch has rb=0.1 keeps no detection; forced epochs are judged by procstatus."""
        message = messages_for("ztf")[2]  # the all-bogus golden message
        epochs = ZTFSurvey().epochs_from_message(message)  # survey under test
        self.assertEqual([e for e in epochs if not e["forced"]], [])  # every rb=0.1 detection discarded
        self.assertEqual(len(epochs), len(message["forced_photometries"]))  # procstatus "0" forced epochs stay

    def test_epochs_from_message_tolerates_missing_arrays(self):
        """None or absent epoch arrays count as empty, as pre_execute's `or []` does."""
        message = messages_for("ztf")[0]  # good message
        message["previous_detections"] = None  # null array
        del message["forced_photometries"]  # absent array
        epochs = ZTFSurvey().epochs_from_message(message)  # survey under test
        self.assertEqual(len(epochs), len(message["detections"]))  # only the detections remain

    def test_forced_is_empty(self):
        """ztf forced epochs flow inline with the detections, so forced_from_message is []."""
        self.assertEqual(ZTFSurvey().forced_from_message(messages_for("ztf")[0]), [])  # always empty


class ScribeAndOutputTestCase(unittest.TestCase):
    """ZTFSurvey.scribe_commands and output_messages against the golden scribe and output."""

    def setUp(self):
        """Build the first golden object with stub features and keep the message as pre_execute left it."""
        self.golden = golden()  # oracle
        self.maxDiff = None  # whole diff on mismatch
        message = messages_for("ztf")[0]  # first golden message
        self.oid = message["oid"]  # 36028941624528297
        self.ao = build(message)  # AstroObject as the parser returns it
        StubExtractor("ztf").compute_features_batch([self.ao])  # stub features, as the golden run
        self.prepared = {**message, "detections": ZTFSurvey().epochs_from_message(message)}  # pre_execute's message
        self.candids = {self.oid: list(message["measurement_id"])}  # execute's candids accumulation

    def golden_scribe(self, step):
        """Return the golden scribe value of `step` for this oid."""
        return next(  # the only matching command
            c["value"] for c in self.golden["scribe"]  # every recorded command
            if c["value"]["step"] == step and c["key"] == str(self.oid)  # this step, this oid
        )  # end of the lookup

    def test_scribe_commands_equal_golden(self):
        """scribe_commands gives the golden update-object and features commands."""
        commands = ZTFSurvey().scribe_commands([self.ao], 1, "ZTFFeatureExtractor", LUT["ztf"])  # survey under test
        as_sent = lambda cmd: normalize(json.loads(json.dumps(cmd)))  # the step json-encodes each command
        self.assertEqual(as_sent(commands["update_object"][0]), self.golden_scribe("update-ztf-object-features"))  # object colors
        self.assertEqual(as_sent(commands["upserting_features"][0]), self.golden_scribe("features"))  # feature rows

    def test_output_messages_equal_golden(self):
        """output_messages gives the golden output (post_execute drops reference) plus the reference records."""
        out = ZTFSurvey().output_messages([self.ao], [self.prepared], self.candids)  # survey under test
        self.assertEqual(len(out), 1)  # one message per object
        self.assertEqual(  # every key but reference equals the golden output
            normalize({k: v for k, v in out[0].items() if k != "reference"}),  # what post_execute keeps
            self.golden["output"][0],  # golden first output message
        )  # end of the output assertion
        ref = self.golden["astro_objects"][str(self.oid)]["reference"]  # golden reference frame, split dict
        self.assertEqual(  # reference records equal the golden reference rows
            normalize(out[0]["reference"]),  # records as parse_output builds them
            [dict(zip(ref["columns"], row)) for row in ref["data"]],  # split dict -> records
        )  # end of the reference assertion
        self.assertEqual(  # exact output keys
            sorted(out[0]),  # keys produced
            ["candid", "detections", "features", "non_detections", "oid", "reference", "xmatches"],  # keys today
        )  # end of the keys assertion


class ReexportTestCase(unittest.TestCase):
    """features.utils.parsers keeps exporting the moved names."""

    def test_parsers_reexports_moved_names(self):
        """Every moved ztf name is importable from parsers and is the moved object itself."""
        from features.surveys import common  # DETECTION_KEYS_MAP home
        from features.surveys.ztf import astro_object, output, scribe  # new homes
        from features.utils import parsers  # old home, still imported by step.py and tests

        moved = {  # name -> new module
            "DETECTION_KEYS_MAP": common,  # shared by both surveys
            "add_mag_and_flux_columns": astro_object,  # ztf magnitude and flux rows
            "get_bogus_flags_for_each_detection": astro_object,  # rb and procstatus per epoch
            "get_reference_for_each_detection": astro_object,  # distnr and rfid per epoch
            "get_new_references_from_message": astro_object,  # references carried in the message
            "detections_to_astro_object": astro_object,  # the ztf parser
            "fid_mapper_for_db": scribe,  # band -> db code
            "parse_scribe_payload": scribe,  # scribe commands
            "parse_output": output,  # output messages
        }  # end of the moved names
        for name, module in moved.items():  # each moved name
            self.assertIs(getattr(parsers, name), getattr(module, name), name)  # same object in both places

    def test_fid_mapper_for_db_is_band_table(self):
        """fid_mapper_for_db keeps today's codes: g 1, r 2, g,r 12, anything else 0."""
        from features.surveys.ztf.scribe import fid_mapper_for_db  # thin alias under test

        got = [fid_mapper_for_db(b) for b in ("g", "r", "g,r", "i", None)]  # known, letter-only and missing bands
        self.assertEqual(got, [1, 2, 12, 0, 0])  # today's fid_map
