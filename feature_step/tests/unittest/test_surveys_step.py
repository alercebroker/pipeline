"""FeatureStep dispatches through the Survey contract and utils/parsers is a re-export shim.

Spec criteria 1 (no survey branches in step.py), 2 (parsers.py has no bodies and
keeps its names) and the FeatureStep half of 3 (unknown or missing SURVEY raises
at construction). The behaviour around them (scribe commands, output, frames) is
pinned by test_surveys_golden; the tests here that already pass before the
rewrite pin what the rewrite must keep.
"""
import ast  # parse step.py and parsers.py without importing them
import importlib  # resolve the home module of each re-exported name
import json  # decode the scribe xmatch command
import pathlib  # locate the two source files
import re  # the survey-branch pattern of spec criterion 1
import unittest  # test case base class
from importlib.metadata import version  # package version the step registers
from unittest import mock  # mock the database, the scribe producer and the xmatch client

from pandas.testing import assert_frame_equal  # compare feature row frames

import features.step as step_module  # module under test, for its path and patch targets
from features.step import FeatureStep  # the step under test
from features.surveys.common import prepare_features_for_db  # shared body the two prepare names delegate to
from features.surveys.lsst import LSSTSurvey  # lsst survey class
from features.surveys.registry import SURVEYS  # registered survey names
from features.surveys.ztf import ZTFSurvey  # ztf survey class
from features.utils import parsers  # the shim under test
from lc_classifier.features.composites.lsst import LSSTFeatureExtractor  # extractor an lsst step holds
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor  # extractor a ztf step holds
from lc_classifier.features.preprocess.lsst import LSSTLightcurvePreprocessor  # preprocessor an lsst step holds
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor  # preprocessor a ztf step holds

from ..message_factory_ztf_ms import allwise_match  # allwise xmatch in the shape the service returns
from .test_surveys_common import astro_object_with  # minimal AstroObject with given feature rows
from .test_surveys_golden import (  # Task 1 step builder, configs, stub and inputs
    CONSUMER_CONFIG,  # mocked consumer config
    LUT,  # feature id -> name per survey
    PRODUCER_CONFIG,  # mocked producer config
    SCRIBE_PRODUCER_CONFIG,  # mocked scribe producer config
    STUB_FEATURES,  # stub feature rows per survey
    StubExtractor,  # writes the stub rows instead of computing
    build_step,  # FeatureStep with Kafka and the database mocked
    messages_for,  # golden input batch per survey
    references_df,  # one-row references table of the ztf path
)  # end of the golden imports

STEP_PATH = pathlib.Path(step_module.__file__)  # feature_step/features/step.py
PARSERS_PATH = pathlib.Path(parsers.__file__)  # feature_step/features/utils/parsers.py

CRITERION_2_NAMES = {  # the 12 names of spec criterion 2 -> module that must define each
    "detections_to_astro_object": "features.surveys.ztf.astro_object",  # ztf parser
    "detections_to_astro_object_lsst": "features.surveys.lsst.astro_object",  # lsst parser
    "get_bogus_flags_for_each_detection": "features.surveys.ztf.astro_object",  # rb and procstatus per epoch
    "prepare_ao_features_for_db": "features.surveys.ztf.scribe",  # ztf feature rows for the db
    "prepare_ao_features_for_db_lsst": "features.surveys.lsst.scribe",  # lsst feature rows for the db
    "parse_scribe_payload": "features.surveys.ztf.scribe",  # ztf scribe commands
    "parse_scribe_payload_lsst": "features.surveys.lsst.scribe",  # lsst scribe commands
    "parse_output": "features.surveys.ztf.output",  # ztf output messages
    "parse_output_lsst": "features.surveys.lsst.output",  # lsst output messages
    "fid_mapper_for_db": "features.surveys.ztf.scribe",  # ztf band -> db code
    "fid_mapper_for_db_lsst": "features.surveys.lsst.scribe",  # lsst band -> db code
    "DETECTION_KEYS_MAP": "features.surveys.common",  # input key -> AstroObject column
}  # end of the criterion 2 names

NAMES_BEFORE_THE_SHIM = [  # every public name bound in parsers.py before this task, in file order
    "logging", "np", "pd", "AstroObject", "query_ao_table", "mag2flux", "mag_err_2_flux_err",  # incidental imports
    "List", "Dict", "Optional", "os", "json",  # incidental imports
    "last_mjd_of", "prepare_features_for_db", "wise_magnitudes", "LSSTSurvey", "ZTFSurvey",  # shared helpers and classes
    "DETECTION_KEYS_MAP", "add_mag_and_flux_columns", "detections_to_astro_object",  # ztf parser names
    "get_bogus_flags_for_each_detection", "get_new_references_from_message",  # ztf parser names
    "get_reference_for_each_detection", "parse_output", "fid_mapper_for_db", "parse_scribe_payload",  # ztf names
    "add_mag_and_flux_lsst", "detections_to_astro_object_lsst", "flux_err_2_mag_err", "fluxnjy2mag",  # lsst parser names
    "parse_output_lsst", "fid_mapper_for_db_lsst", "parse_scribe_payload_lsst",  # lsst names
    "prepare_ao_features_for_db", "prepare_ao_features_for_db_lsst",  # the two bodies this task moves
]  # end of the names


def config_for(**extra):
    """Return the mocked step config of the golden test with `extra` keys added and no SURVEY of its own."""
    return {  # same shape as test_surveys_golden.build_step
        "PRODUCER_CONFIG": PRODUCER_CONFIG,  # mocked producer
        "CONSUMER_CONFIG": CONSUMER_CONFIG,  # mocked consumer
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,  # mocked scribe producer
        **extra,  # SURVEY and anything else the test sets
    }  # end of the config


def xmatch_step(survey):
    """Return a mocked step for `survey` with USE_XMATCH on and a service that answers one allwise match per oid."""
    config = config_for(SURVEY=survey, USE_XMATCH=True, XMATCH_CONFIG={"base_url": "http://xmatch.test"})  # xmatch on
    step = FeatureStep(config=config, db_sql=mock.MagicMock())  # no database
    step.scribe_producer = mock.MagicMock()  # records producer.produce calls
    step.xmatch_client = mock.MagicMock()  # no http
    step.xmatch_client.conesearch_with_metadata.side_effect = lambda **kw: (  # one answer per catalog request
        [{**allwise_match(int(oid)), "oid": oid} for oid in kw["oids"]]  # an allwise match per oid
        if kw["catalog"] == "allwise"  # only the allwise request matches
        else []  # other catalogs: nothing
    )  # end of the side effect
    return step  # ready for pre_execute


class ConstructionTestCase(unittest.TestCase):
    """FeatureStep resolves SURVEY through the registry when it is built (criterion 3, Review Focus 5)."""

    def test_unknown_survey_raises_at_construction(self):
        """SURVEY "x" raises ValueError naming the value and both known surveys."""
        with self.assertRaises(ValueError) as raised:  # no silent step without extractor
            FeatureStep(config=config_for(SURVEY="x"), db_sql=mock.MagicMock())  # unknown name
        message = str(raised.exception)  # text of the error
        self.assertIn("'x'", message)  # the offending value
        self.assertIn("ztf", message)  # first known survey
        self.assertIn("lsst", message)  # second known survey

    def test_missing_survey_raises_at_construction(self):
        """A config without SURVEY raises ValueError naming both known surveys."""
        with self.assertRaises(ValueError) as raised:  # no silent step without extractor
            FeatureStep(config=config_for(), db_sql=mock.MagicMock())  # no SURVEY key
        message = str(raised.exception)  # text of the error
        self.assertIn("None", message)  # the missing value
        self.assertIn("ztf", message)  # first known survey
        self.assertIn("lsst", message)  # second known survey

    def test_step_uses_survey_objects(self):
        """The step holds the Survey object, its name, and the preprocessor and extractor the survey makes."""
        ztf = build_step("ztf")  # mocked ztf step
        self.assertIsInstance(ztf.survey, ZTFSurvey)  # the contract object, not a string
        self.assertEqual(ztf.survey_name, "ztf")  # the string the xmatch scribe command carries
        self.assertIsInstance(ztf.lightcurve_preprocessor, ZTFLightcurvePreprocessor)  # from make_preprocessor
        self.assertIsInstance(ztf.feature_extractor, ZTFFeatureExtractor)  # from make_extractor
        lsst = build_step("lsst")  # mocked lsst step
        self.assertIsInstance(lsst.survey, LSSTSurvey)  # the contract object
        self.assertEqual(lsst.survey_name, "lsst")  # the string the xmatch scribe command carries
        self.assertIsInstance(lsst.lightcurve_preprocessor, LSSTLightcurvePreprocessor)  # from make_preprocessor
        self.assertIsInstance(lsst.feature_extractor, LSSTFeatureExtractor)  # from make_extractor

    def test_step_keeps_id_column_and_extractor_group(self):
        """id_column and extractor_group keep the values the survey branches set."""
        ztf, lsst = build_step("ztf"), build_step("lsst")  # one step per survey
        self.assertEqual((ztf.id_column, lsst.id_column), ("measurement_id", "measurement_id"))  # same column
        self.assertEqual(ztf.extractor_group, "ZTFFeatureExtractor")  # class name of the ztf extractor
        self.assertEqual(lsst.extractor_group, "LSSTFeatureExtractor")  # class name of the lsst extractor

    def test_version_and_lut_are_resolved_with_the_survey_sid_and_tid(self):
        """get_or_create_version_id and get_feature_name_lut get sid/tid 0/0 for ztf and 1/1 for lsst."""
        for survey, sid, tid in (("ztf", 0, 0), ("lsst", 1, 1)):  # ids of the Survey contract
            with mock.patch.object(step_module, "get_or_create_version_id") as version_id:  # no version lookup in the db
                with mock.patch.object(step_module, "get_feature_name_lut") as lut:  # no LUT read from the db
                    db_sql = mock.MagicMock()  # connection handed to both calls
                    step = FeatureStep(config=config_for(SURVEY=survey), db_sql=db_sql)  # step under test
            version_id.assert_called_once_with(  # one version lookup
                db_sql, "multisurvey", version("feature-step"), sid, tid, step.logger  # default schema, package version
            )  # end of the version assertion
            lut.assert_called_once_with(db_sql, "multisurvey", sid, tid, step.logger)  # one LUT read
            self.assertIs(step.extractor_version, version_id.return_value)  # stored as returned
            self.assertIs(step.feature_name_lut, lut.return_value)  # stored as returned


class NoSurveyBranchesTestCase(unittest.TestCase):
    """step.py takes no decision on the survey name (criterion 1)."""

    def setUp(self):
        """Read step.py once as text and as a syntax tree."""
        self.source = STEP_PATH.read_text()  # file content
        self.tree = ast.parse(self.source)  # syntax tree, comments and layout dropped

    def test_step_has_no_survey_branches(self):
        """No line matches the grep of criterion 1 (survey ==, == "ztf", == "lsst")."""
        pattern = re.compile(r'survey\s*==|==\s*"(ztf|lsst)"')  # criterion 1, spaces optional
        hits = [line for line in self.source.splitlines() if pattern.search(line)]  # offending lines
        self.assertEqual(hits, [])  # none

    def test_step_names_no_registered_survey(self):
        """No string constant in step.py is the name of a registered survey."""
        names = [  # string constants equal to a survey name
            node.value for node in ast.walk(self.tree)  # every node of the file
            if isinstance(node, ast.Constant) and node.value in set(SURVEYS)  # "ztf" or "lsst" as a literal
        ]  # end of the names
        self.assertEqual(names, [])  # none

    def test_step_compares_no_survey_attribute(self):
        """No comparison in step.py has `survey` or `survey_name` on either side."""
        hits = []  # line numbers of offending comparisons
        for node in ast.walk(self.tree):  # every node of the file
            if not isinstance(node, ast.Compare):  # only ==, !=, in, is ...
                continue  # skip the rest
            for side in [node.left, *node.comparators]:  # both sides of the comparison
                name = side.attr if isinstance(side, ast.Attribute) else getattr(side, "id", None)  # x.survey or survey
                if name in ("survey", "survey_name"):  # a decision on the survey
                    hits.append(node.lineno)  # remember where
        self.assertEqual(hits, [])  # none

    def test_debug_log_line_is_gone(self):
        """The "TIENE LENGTH MAYOR A CERO" info log of pre_execute is removed."""
        self.assertEqual(self.source.count("TIENE LENGTH"), 0)  # log noise dropped by this task


class ParsersShimTestCase(unittest.TestCase):
    """features/utils/parsers.py only re-exports names that live under features/surveys (criterion 2)."""

    def setUp(self):
        """Parse parsers.py once."""
        self.tree = ast.parse(PARSERS_PATH.read_text())  # syntax tree of the shim

    def test_parsers_is_a_shim(self):
        """parsers.py has no function, class or lambda, and the 12 names of criterion 2 are importable from it."""
        bodies = [  # nodes that carry code of their own
            type(node).__name__ for node in ast.walk(self.tree)  # every node of the file
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))  # def, class, lambda
        ]  # end of the bodies
        self.assertEqual(bodies, [])  # re-exports only
        for name in CRITERION_2_NAMES:  # each of the 12 names
            self.assertTrue(hasattr(parsers, name), name)  # importable from the old home

    def test_parsers_has_only_a_docstring_and_imports(self):
        """Every top-level statement of parsers.py is the module docstring or an import, and the docstring says shim."""
        docstring = ast.get_docstring(self.tree)  # module docstring, None when absent
        self.assertIsNotNone(docstring)  # the shim explains itself
        self.assertIn("re-export", docstring)  # and says what it is
        others = [  # statements that are neither the docstring nor an import
            type(node).__name__ for node in self.tree.body[1:]  # skip the docstring expression
            if not isinstance(node, (ast.Import, ast.ImportFrom))  # anything else is code
        ]  # end of the others
        self.assertEqual(others, [])  # nothing else

    def test_criterion_2_names_are_the_objects_under_surveys(self):
        """Each of the 12 names in parsers is the very object its features.surveys module defines."""
        for name, module_name in CRITERION_2_NAMES.items():  # name -> home module
            home = importlib.import_module(module_name)  # module under features/surveys
            self.assertIs(getattr(parsers, name), getattr(home, name), name)  # same object, not a copy
            if callable(getattr(home, name)):  # functions know where they were defined
                self.assertEqual(getattr(home, name).__module__, module_name, name)  # defined there, not re-imported

    def test_every_name_bound_before_the_shim_is_still_importable(self):
        """No name that parsers.py bound before this task disappears from it."""
        missing = [name for name in NAMES_BEFORE_THE_SHIM if not hasattr(parsers, name)]  # names lost
        self.assertEqual(missing, [])  # none

    def test_prepare_ao_features_for_db_keeps_its_rows(self):
        """prepare_ao_features_for_db(ao, lut) equals the shared body with the ztf bands and the name kept."""
        ao = astro_object_with(STUB_FEATURES["ztf"])  # stub rows with NaN, inf and an unknown name
        with self.assertLogs("alerce.FeatureStep", "WARNING"):  # Unknown_feat is dropped with a warning
            got = parsers.prepare_ao_features_for_db(ao, LUT["ztf"])  # name under test, two positional arguments
        with self.assertLogs("alerce.FeatureStep", "WARNING"):  # same warning from the shared body
            expected = prepare_features_for_db(ao, LUT["ztf"], ZTFSurvey.bands, keep_name=True)  # ztf bands, name kept
        assert_frame_equal(got, expected)  # same rows, columns, dtypes and index
        self.assertEqual(list(got.columns), ["name", "value", "band", "feature_id"])  # name survives for ztf

    def test_prepare_ao_features_for_db_lsst_keeps_its_rows(self):
        """prepare_ao_features_for_db_lsst(ao, lut) equals the shared body with the lsst bands and no name."""
        ao = astro_object_with(STUB_FEATURES["lsst"])  # stub rows with NaN and an unknown name
        with self.assertLogs("alerce.FeatureStep", "WARNING"):  # Unknown_feat is dropped with a warning
            got = parsers.prepare_ao_features_for_db_lsst(ao, LUT["lsst"])  # name under test, two positional arguments
        with self.assertLogs("alerce.FeatureStep", "WARNING"):  # same warning from the shared body
            expected = prepare_features_for_db(ao, LUT["lsst"], LSSTSurvey.bands, keep_name=False)  # lsst bands, no name
        assert_frame_equal(got, expected)  # same rows, columns, dtypes and index
        self.assertEqual(list(got.columns), ["value", "band", "feature_id"])  # name dropped for lsst


class StepCallsTheContractTestCase(unittest.TestCase):
    """What execute and pre_execute hand to the database and to scribe stays as it is per survey."""

    def run_batch(self, survey):
        """Run pre_execute and execute on the golden batch of `survey`; return the step and the references mock."""
        step = build_step(survey)  # mocked step
        step.feature_extractor = StubExtractor(survey)  # no feature computation
        with mock.patch.object(FeatureStep, "_get_sql_references", return_value=references_df()) as references:  # no db
            step.execute(step.pre_execute(messages_for(survey)))  # the two methods under test
        return step, references  # for the assertions

    def test_ztf_reads_references_once_for_the_prepared_oids(self):
        """ztf calls _get_sql_references once, with the oids that survived pre_execute."""
        _, references = self.run_batch("ztf")  # golden ztf batch, third message all bogus
        references.assert_called_once()  # one query per batch
        oids = references.call_args[0][0]  # the list handed to the query
        self.assertEqual(sorted(oids), [36028941624528297, 36028941624528298])  # the all-bogus oid is not asked for

    def test_lsst_never_reads_references(self):
        """lsst does not call _get_sql_references."""
        _, references = self.run_batch("lsst")  # golden lsst batch
        references.assert_not_called()  # needs_references is False

    def test_xmatch_scribe_command_carries_the_survey_name(self):
        """The xmatch scribe command has "survey" equal to the config string, for ztf and for lsst."""
        for survey in ("ztf", "lsst"):  # both registered surveys
            step = xmatch_step(survey)  # xmatch on, service mocked
            message = messages_for(survey)[0]  # first golden message
            step.pre_execute([message])  # asks the service and writes the xmatch command
            first = step.scribe_producer.producer.produce.call_args_list[0][1]  # keyword arguments of the first produce
            command = json.loads(first["value"])  # the xmatch command as sent
            self.assertEqual(command["step"], "xmatch")  # it is the xmatch command
            self.assertEqual(command["survey"], survey)  # a plain string, as before
            self.assertEqual(first["key"], str(message["oid"]).encode("utf-8"))  # keyed by oid
