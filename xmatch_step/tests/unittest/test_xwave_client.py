
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

BASE_URL = "http://localhost:8080"


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
        _, self.conesearch_responses, self.metadata_responses, _  = set_data_test()
        self.get_call_count = 0

    def get(self, url):
        if "conesearch" in url:
            oid = url.split('/')[-1]
            self.get_call_count += 1
            print(self.conesearch_responses)
            return MockResponse(200, self.conesearch_responses[oid], True)
        
        elif "metadata" in url:
            source_id = url.split('/')[-1] 
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
            oid = url.split('/')[-1] 
            self.get_call_count += 1
            return MockResponse(200, {"results": self.conesearch_responses[oid]}, True)
        
        elif "metadata" in url:
            source_id = url.split('/')[-1] 
            self.get_call_count += 1
            return MockResponse(200, {"results": self.metadata_responses[source_id]}, True)
        
        else:
            self.get_call_count += 1
            return MockResponse(404, {}, False)

    def setUp(self):
        self.client = XwaveClient(BASE_URL)
        self.get_call_count = 0
        self.input_dataframe, self.conesearch_responses, self.metadata_responses, self.dataframe_output = set_data_test()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")    
    @mock.patch('xmatch_step.core.xwave_client.aiohttp.ClientSession')
    def test_request(self, mock_get):
        session_mock = SessionMock()
        mock_get.return_value = session_mock
        result = self.client.execute(self.input_dataframe, **client_parameters)
        # Number of api calls = 2*oid (since we call the conesearch and the metadata)
        expected_calls = len(self.input_dataframe) * 2
        self.assertEqual(session_mock.get_call_count, expected_calls)
        pd.testing.assert_frame_equal(result, self.dataframe_output)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @mock.patch('xmatch_step.core.xwave_client.aiohttp.ClientSession')
    def error_request(self, mock_get):
        session_mock = mock.MagicMock()
        session_mock.get = self.get
        mock_get.return_value = session_mock
        with self.assertRaises(Exception):
            self.client.execute(self.input_dataframe, **client_parameters)

