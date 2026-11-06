"""Golden capture of FeatureStep behaviour for ztf and lsst before the survey refactor.

Feature computation is replaced by a stub that writes fixed values, so the
fixtures freeze everything around it: pre_execute filtering, AstroObject
building, scribe commands and the output messages. Regenerate the fixtures
with FEATURE_STEP_WRITE_GOLDEN=1 (the test then fails on purpose).
"""
import json  # decode scribe values and read/write fixtures
import math  # detect NaN and infinities
import os  # read FEATURE_STEP_WRITE_GOLDEN
import pathlib  # locate the fixtures dir
import unittest  # test case base class
from unittest import mock  # mock db_sql, scribe producer and the references query

import numpy as np  # numpy scalar and array types in normalize
import pandas as pd  # build feature/reference frames, detect pandas objects

from features.step import FeatureStep  # the step under capture
from lc_classifier.features.core.base import query_ao_table  # read oid from AstroObject metadata

from ..message_factory_lsst_ms import generate_message_lsst  # synthetic lsst messages
from ..message_factory_ztf_ms import generate_message  # synthetic ztf multisurvey messages

FIXTURES = pathlib.Path(__file__).parent / "fixtures"  # where the golden json files live

CONSUMER_CONFIG = {  # consumer config copied from test_step_ztf_multisurvey
    "CLASS": "unittest.mock.MagicMock",  # no Kafka consumer
    "PARAMS": {"bootstrap.servers": "server", "group.id": "group_id"},  # dummy params
    "TOPICS": ["topic"],  # dummy topic
}
PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"}  # no Kafka producer
SCRIBE_PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test-scribe"}  # no scribe producer

STUB_FEATURES = {  # (name, fid, value) rows the stub writes into every AstroObject
    "ztf": [("Amplitude", "g", 1.5), ("Amplitude", "r", 2.5), ("g-r_max", "g,r", 0.3),  # band and color features
            ("g-r_mean", "g,r", 0.2), ("g-r_max_corr", "g,r", 0.25), ("g-r_mean_corr", "g,r", 0.15),  # colors read by update-ztf-object-features
            ("Power_rate_1_4", None, 0.7), ("Mean", "g", float("nan")), ("Unknown_feat", "g", 1.0),  # renamed, NaN and unmapped features
            ("Skew", "r", float("inf"))],  # infinite value
    "lsst": [("Amplitude", "u", 1.1), ("Amplitude", "g", 1.2), ("u-g_mean", "u,g", 0.4),  # band and color features
             ("Power_rate_1_2", None, 0.6), ("Mean", "i", float("nan")), ("Unknown_feat", "r", 1.0)],  # renamed, NaN and unmapped features
}
LUT = {"ztf": {1: "Amplitude", 2: "g-r_max", 3: "g-r_mean", 4: "g-r_max_corr", 5: "g-r_mean_corr",  # ztf feature id -> name
               6: "Power_rate_1/4", 7: "Mean", 8: "Skew"},  # Unknown_feat deliberately absent
       "lsst": {1: "Amplitude", 2: "u-g_mean", 3: "Power_rate_1/2", 4: "Mean"}}  # lsst feature id -> name

FRAME_NAMES = ("detections", "forced_photometry", "metadata", "reference")  # AstroObject frames recorded before preprocessing


class RecordingPreprocessor:
    """Wraps the step's real lightcurve preprocessor and, before delegating, records each
    AstroObject's frames (detections, forced_photometry, metadata, reference) as
    `to_dict("split")` under self.frames, keyed by oid: the frames as the parser returned
    them, before preprocessing, so criterion 6 has an oracle."""

    def __init__(self, inner):
        """Keep the real preprocessor to delegate to and start with no recorded frames."""
        self.inner = inner  # the step's own ZTF or LSST preprocessor
        self.frames = {}  # str(oid) -> {frame name: split dict or None}

    def preprocess_batch(self, astro_objects, progress_bar=False):
        """Record each AstroObject's frames as the parser returned them, then run the real preprocessor."""
        for ao in astro_objects:  # one AstroObject per prepared message
            oid = query_ao_table(ao.metadata, "oid")  # oid as the parser stored it
            self.frames[str(oid)] = {  # snapshot the four frames before preprocessing mutates them
                name: (None if getattr(ao, name) is None else getattr(ao, name).to_dict("split"))  # None stays null
                for name in FRAME_NAMES  # detections, forced_photometry, metadata, reference
            }
        self.inner.preprocess_batch(astro_objects, progress_bar=progress_bar)  # today's preprocessing, unchanged


class StubExtractor:
    """Writes STUB_FEATURES[survey] into every AstroObject instead of computing features."""

    def __init__(self, survey):
        """Keep the survey whose stub rows are written."""
        self.survey = survey  # selects STUB_FEATURES[survey]

    def compute_features_batch(self, astro_objects, progress_bar=False):
        """Replace each AstroObject's features with the stub rows."""
        for ao in astro_objects:  # one AstroObject per prepared message
            sid = ao.detections["sid"].values[0]  # same sid the scribe payload reads
            ao.features = pd.DataFrame(  # stub features in the extractor frame layout
                [(name, value, fid, sid, "stub") for name, fid, value in STUB_FEATURES[self.survey]],  # reorder to name, value, fid
                columns=["name", "value", "fid", "sid", "version"],  # columns of empty_normal_dataframe
            )


def messages_for(survey):
    """Return the input batch for `survey`: three ztf messages (the last all-bogus) or two lsst ones."""
    if survey == "ztf":  # ztf multisurvey batch
        bogus = generate_message(oid=36028941624528299, seed=3)  # message whose every epoch is bogus
        for key in ("detections", "previous_detections", "forced_photometries"):  # the three epoch arrays
            for epoch in bogus[key]:  # every epoch in the array
                epoch["rb"] = 0.1  # below the 0.55 real-bogus threshold, at the top level where discard_bogus_detections looks
        return [  # three messages: with xmatch, without forced, all bogus
            generate_message(seed=1, with_xmatch=True),  # default oid, allwise match attached
            generate_message(oid=36028941624528298, seed=2, n_forced=0),  # no forced photometry
            bogus,  # expected to be dropped by pre_execute
        ]
    return [  # two lsst messages
        generate_message_lsst(seed=7, with_xmatch=True),  # default oid, allwise match attached
        generate_message_lsst(oid=1152921504606847002, seed=8),  # second object, no xmatch
    ]


def references_df():
    """Return the one-row reference table the mocked `_get_sql_references` hands to the ztf path."""
    return pd.DataFrame(  # one reference row for the default ztf oid
        [{"oid": 36028941624528297, "rfid": 1, "sharpnr": 0.1, "chinr": 0.5}],  # values from the brief
        columns=["oid", "rfid", "sharpnr", "chinr"],  # column order of get_sql_references
    )


def build_step(survey):
    """Build a FeatureStep for `survey` with Kafka and the database mocked, the LUT pinned to LUT[survey]."""
    config = {  # same config shape as test_step_ztf_multisurvey.build_step
        "PRODUCER_CONFIG": PRODUCER_CONFIG,  # mocked producer
        "CONSUMER_CONFIG": CONSUMER_CONFIG,  # mocked consumer
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,  # mocked scribe producer
        "SURVEY": survey,  # ztf or lsst
    }
    step = FeatureStep(config=config, db_sql=mock.MagicMock())  # no database
    step.scribe_producer = mock.MagicMock()  # records producer.produce calls
    step.extractor_version = 1  # the mocked db would make this a MagicMock
    step.feature_name_lut = LUT[survey]  # the mocked db would make this a MagicMock
    return step  # ready to run


def capture(survey) -> dict:
    """Run pre_execute, execute and post_execute on messages_for(survey) and return everything observable, normalized."""
    step = build_step(survey)  # mocked step
    step.feature_extractor = StubExtractor(survey)  # replaces feature computation
    recorder = RecordingPreprocessor(step.lightcurve_preprocessor)  # records parser output, then preprocesses as today
    step.lightcurve_preprocessor = recorder  # inject the recording wrapper
    with mock.patch.object(FeatureStep, "_get_sql_references", return_value=references_df()):  # no db query for references
        prepared = step.pre_execute(messages_for(survey))  # filtering under capture
        out = step.post_execute(step.execute(prepared))  # AstroObjects, scribe commands and output
    return normalize({  # plain json-safe structure
        "prepared_oids": [m["oid"] for m in prepared],  # which messages survived pre_execute
        "astro_objects": recorder.frames,  # oid -> frames as the parser returned them, before preprocessing
        "scribe": [  # every scribe command in produce order
            {"topic": kw["topic"], "key": kw["key"].decode(), "value": json.loads(kw["value"]),  # decode key and value
             "raw": kw["value"].decode("utf-8")}  # wire bytes as text: pins key order, separators and NaN spelling
            for _, kw in step.scribe_producer.producer.produce.call_args_list  # keyword calls only
        ],
        "output": out,  # messages that would go to the output topic
    })


def normalize(obj, path="$"):
    """Turn `obj` into plain dicts/lists/str/int/float/bool/None that json.dumps accepts and that compare equal.

    NaN becomes "NaN", +/-inf becomes "Inf"/"-Inf", pandas NA/NaT become "NaN",
    numpy scalars and arrays become Python values, DataFrames become split dicts,
    bytes are decoded, tuples become lists and dict keys become str. Anything
    else raises TypeError naming its type and path.
    """
    if obj is None or isinstance(obj, (bool, str)):  # already json-safe
        return obj  # unchanged
    if obj is pd.NA or obj is pd.NaT:  # pandas missing markers
        return "NaN"  # same marker as float NaN
    if isinstance(obj, np.generic):  # numpy scalar (float64, int64, bool_, str_)
        return normalize(obj.item(), path)  # convert to the Python scalar and recheck
    if isinstance(obj, int):  # plain int
        return obj  # unchanged
    if isinstance(obj, float):  # plain float, may be NaN or inf
        if math.isnan(obj):  # NaN != NaN, so store a marker
            return "NaN"  # comparable marker
        if math.isinf(obj):  # json has no infinity
            return "Inf" if obj > 0 else "-Inf"  # signed marker
        return obj  # finite float unchanged
    if isinstance(obj, bytes):  # raw bytes
        return obj.decode()  # utf-8 text
    if isinstance(obj, dict):  # mapping
        return {str(k): normalize(v, f"{path}.{k}") for k, v in obj.items()}  # str keys, normalized values
    if isinstance(obj, (list, tuple)):  # sequence
        return [normalize(v, f"{path}[{i}]") for i, v in enumerate(obj)]  # tuples become lists
    if isinstance(obj, np.ndarray):  # numpy array
        return normalize(obj.tolist(), path)  # nested Python lists
    if isinstance(obj, pd.DataFrame):  # pandas frame
        return normalize(obj.to_dict("split"), path)  # index, columns, data
    if isinstance(obj, pd.Series):  # pandas series
        return normalize(obj.tolist(), path)  # values only
    raise TypeError(f"cannot normalize {type(obj)!r} at {path}")  # surface the exact type and path


def golden_text(got) -> str:
    """Serialize a capture exactly as the fixture file stores it; the writer and the comparer both use this."""
    return json.dumps(got, indent=1, sort_keys=True)  # text form keeps 1 vs 1.0 and true vs 1 distinct


class GoldenTestCase(unittest.TestCase):
    """Compare today's captured behaviour with the committed golden fixtures."""

    def test_ztf_matches_golden(self):
        """The ztf capture equals golden_ztf.json."""
        self._check("ztf")  # run the comparison for ztf

    def test_lsst_matches_golden(self):
        """The lsst capture equals golden_lsst.json."""
        self._check("lsst")  # run the comparison for lsst

    def _check(self, survey):
        """Capture `survey` and compare it with its fixture, or write the fixture when FEATURE_STEP_WRITE_GOLDEN is set."""
        path = FIXTURES / f"golden_{survey}.json"  # fixture for this survey
        got = capture(survey)  # today's behaviour
        if os.environ.get("FEATURE_STEP_WRITE_GOLDEN"):  # regeneration mode
            path.parent.mkdir(exist_ok=True)  # create fixtures/ on first write
            path.write_text(golden_text(got))  # same serialization the comparison uses
            self.fail(f"wrote {path}")  # never pass while regenerating
        self.maxDiff = None  # show the whole text diff on mismatch
        self.assertMultiLineEqual(path.read_text(), golden_text(got))  # text compare, FileNotFoundError when the fixture is missing
