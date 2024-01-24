import unittest
import pandas as pd
import os
import io
from unittest import mock
from astropy.table import Table
from astropy.io.votable import from_table
from xmatch_step.core.xmatch_client import XmatchClient


class MockResponse:
    def __init__(self, content_path, format="pandas", ok=True):
        if format == "pandas":
            with open(content_path) as f:
                expected_result = str.encode(f.read())

        else:
            t = Table.read(content_path, format="csv")
            votable = from_table(t)
            f = io.BytesIO()
            votable.to_xml(f)
            f.seek(0)
            expected_result = f.read()
        self.content = expected_result
        self.ok = ok


FILE_PATH = os.path.dirname(__file__)
DATA_FOLDER_RELATIVE_PATH = "../data/xmatch_client/"


class XmatchClientTest(unittest.TestCase):
    test_pandas_data = None
    selection = "best"
    radius = 1
    name = "allwise"
    columns = [
        "AllWISE",
        "RAJ2000",
        "DEJ2000",
        "W1mag",
        "W2mag",
        "W3mag",
        "W4mag",
        "e_W1mag",
        "e_W2mag",
        "e_W3mag",
        "e_W4mag",
        "Jmag",
        "e_Jmag",
        "Hmag",
        "e_Hmag",
        "Kmag",
        "e_Kmag",
    ]
    client = XmatchClient()

    def setUp(self):
        self.test_pandas_data = pd.read_csv(
            os.path.join(FILE_PATH, DATA_FOLDER_RELATIVE_PATH, "input.csv")
        )
        self.test_astropy_data = Table.read(
            os.path.join(
                FILE_PATH,
                DATA_FOLDER_RELATIVE_PATH,
                "astropy_response.votable",
            ),
            format="votable",
        )

    @mock.patch(
        "xmatch_step.core.xmatch_client.requests.post",
        return_value=MockResponse(
            content_path=os.path.join(
                FILE_PATH, DATA_FOLDER_RELATIVE_PATH, "response.csv"
            ),
            format="pandas",
        ),
    )
    def test_execute_pandas(self, requests_mock):
        input_type = "pandas"
        output_type = "pandas"
        output = self.client.execute(
            self.test_pandas_data,
            input_type,
            self.name,
            self.columns,
            self.selection,
            output_type,
            self.radius,
        )
        self.assertIsInstance(output, pd.DataFrame)

    @mock.patch(
        "xmatch_step.core.xmatch_client.requests.post",
        return_value=MockResponse(
            content_path=os.path.join(
                FILE_PATH, DATA_FOLDER_RELATIVE_PATH, "response.csv"
            ),
            format="votable",
        ),
    )
    def test_execute_astropy(self, requests_mock):
        input_type = "astropy"
        output_type = "astropy"
        output = self.client.execute(
            self.test_astropy_data,
            input_type,
            self.name,
            self.columns,
            self.selection,
            output_type,
            self.radius,
        )

    @mock.patch("xmatch_step.core.xmatch_client.requests.post")
    def test_execute_response_error(self, mock_post):
        mock_post.return_value.ok = False
        mock_post.return_value.status_code = 500
        with self.assertRaises(Exception):
            self.client.execute(
                self.test_pandas_data,
                "pandas",
                self.name,
                self.columns,
                self.selection,
                "pandas",
                self.radius,
            )

    def test_wrong_input_type(self):
        input_type = "wrong"
        output_type = "astropy"
        with self.assertRaises(Exception):
            self.client.execute(
                self.test_astropy_data,
                input_type,
                self.name,
                self.columns,
                self.selection,
                output_type,
                self.radius,
            )

    def test_wrong_output_type(self):
        input_type = "astropy"
        output_type = "wrong"
        with self.assertRaises(Exception):
            self.client.execute(
                self.test_astropy_data,
                input_type,
                self.name,
                self.columns,
                self.selection,
                output_type,
                self.radius,
            )

    def tearDown(self):
        del self.test_pandas_data
        del self.test_astropy_data
