import unittest
import pandas as pd
import os
import io
from unittest import mock
from xmatch_step.core.xwave_client import XwaveClient
from tests.data.xwave_client.data import set_data_test, client_parameters
import pytest

# mock requests en xwave_client

# testear que si la respuesta es correcta el dataframe queda como lo queremos
# por como se usa el cliente de xwave, vamos a hacer assert de que requests se llame
# n veces, con n = oids del input


# testear que pasa si el cliente no retorna 200
# para algun oid.


class MockResponse:
    status = 0
    json_data = {}

    def __init__(self, status_code, json_data, ok):
        self.status = status_code
        self.json_data = json_data
        self.ok = ok

    async def json(self):
        return self.json_data

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


class SessionMock:
    # Set the data for the tests
    def __init__(self):
        (
            _,
            self.conesearch_responses,
            self.metadata_responses,
            _,
        ) = set_data_test()
        self.get_call_count = 0

    def get(self, url):
        if "conesearch" in url:
            oid = url.split("/")[-1]
            self.get_call_count += 1
            return MockResponse(200, self.conesearch_responses[oid], True)

        elif "metadata" in url:
            source_id = url.split("/")[-1]
            self.get_call_count += 1
            return MockResponse(200, self.metadata_responses[source_id], True)

        else:
            self.get_call_count += 1
            return MockResponse(404, {}, False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class XwaveClientTest(unittest.TestCase):
    def get(self, url):
        if "conesearch" in url:
            oid = url.split("/")[-1]
            self.get_call_count += 1
            return MockResponse(
                200, {"results": self.conesearch_responses[oid]}, True
            )

        elif "metadata" in url:
            source_id = url.split("/")[-1]
            self.get_call_count += 1
            return MockResponse(
                200, {"results": self.metadata_responses[source_id]}, True
            )

        else:
            self.get_call_count += 1
            return MockResponse(404, {}, False)

    def setUp(self):
        self.client = XwaveClient("https://test_url_xwave:8081")
        self.get_call_count = 0
        (
            self.input_dataframe,
            self.conesearch_responses,
            self.metadata_responses,
            self.dataframe_output,
        ) = set_data_test()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @mock.patch("xmatch_step.core.xwave_client.aiohttp.ClientSession")
    def test_request(self, mock_get):
        session_mock = SessionMock()
        mock_get.return_value = session_mock
        result = self.client.execute(self.input_dataframe, **client_parameters)
        # Number of api calls = 2*oid (since we call the conesearch and the metadata)
        expected_calls = len(self.input_dataframe) * 2
        self.assertEqual(session_mock.get_call_count, expected_calls)
        pd.testing.assert_frame_equal(result, self.dataframe_output)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @mock.patch("xmatch_step.core.xwave_client.aiohttp.ClientSession")
    def error_request(self, mock_get):
        session_mock = mock.MagicMock()
        session_mock.get = self.get
        mock_get.return_value = session_mock
        with self.assertRaises(Exception):
            self.client.execute(self.input_dataframe, **client_parameters)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @mock.patch("xmatch_step.core.xwave_client.aiohttp.ClientSession")
    def test_projection_request(self, mock_get):
        # Change parameters to drop some metadata columns. This is done via projection, which selects the columns from metadata that will be kept
        # THis is added to the client parameters sent to the request
        projection_params = client_parameters.copy()
        # Projection uses the originak names before the rename of the columns for the final dataframe
        projection_params["ext_columns"] = ["W1mpro", "W2mpro", "J_m_2mass"]

        # Create expected output dataframe by selecting only the columns that will be kept from the original dataframe output
        columns_to_keep = [
            "angDist",
            "col1",
            "oid_in",
            "ra_in",
            "dec_in",
            "AllWISE",
            "RAJ2000",
            "DEJ2000",
            "W1mag",
            "W2mag",
            "Jmag",
        ]
        expected_output = self.dataframe_output[columns_to_keep].copy()

        session_mock = SessionMock()
        mock_get.return_value = session_mock

        # Execute with new parameters
        result = self.client.execute(self.input_dataframe, **projection_params)
        expected_calls = len(self.input_dataframe) * 2
        self.assertEqual(session_mock.get_call_count, expected_calls)

        # Verify result has the expected columns
        pd.testing.assert_frame_equal(result, expected_output)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @mock.patch("xmatch_step.core.xwave_client.aiohttp.ClientSession")
    def test_invalid_projection_request(self, mock_get):
        # Set up projection using a column not available in the metadata columns
        projection_params = client_parameters.copy()
        projection_params["ext_columns"] = ["invalid_column", "w1mpro"]

        session_mock = SessionMock()
        mock_get.return_value = session_mock
        # Check if the warning is raised
        with pytest.warns(Warning) as record:
            self.client.execute(self.input_dataframe, **projection_params)

        assert any(
            "The following columns in the projection are not valid"
            in str(w.message)
            for w in record
        )
