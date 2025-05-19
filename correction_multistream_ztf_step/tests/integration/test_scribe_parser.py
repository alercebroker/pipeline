
import json

import unittest
import logging
import pytest

from core.parsers.scribe_parser import dict_splitter

KEYS = [
    'step',
    'survey',
    'payload'
]

OIDS = [
    36028941591602594,
    61616161616161616,
    84848484848484848
]

with open('core/parsers/parsers_utils/test_scribe_parser.json') as f:
    data = json.load(f)

class TestScribeParser():

    def setup_method(self):
        self.input_data = data

    def test_input(self):
        
        assert type(self.input_data) == dict

        for k in list(self.input_data.keys()):
            assert k in KEYS
    
    def test_output(self):
        OIDS_AUX = OIDS.copy()
        output_data = dict_splitter(self.input_data)

        assert type(output_data) == list
        assert len(output_data) > 0

        for d in output_data:

            assert type(d) == dict
            assert len(list(d.keys()) ) > 0

            for k in list(d.keys()):
                assert k in KEYS

            assert d['payload']['oid'] in OIDS_AUX
            OIDS_AUX.remove(d['payload']['oid'])


