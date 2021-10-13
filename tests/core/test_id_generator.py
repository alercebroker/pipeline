import unittest

from survey_parser_plugins.core import id_generator


class TestIdGenerator(unittest.TestCase):
    def test_input_output_1(self):
        ra, dec = -100.9, -10.1
        response = id_generator(ra, dec)
        self.assertIsInstance(ra, float)
        self.assertIsInstance(dec, float)
        self.assertIsInstance(response, int)
        self.assertTrue(response >= 1000000000000000000)

    def test_input_output_2(self):
        ra, dec = 400.9, -10.1
        response = id_generator(ra, dec)
        self.assertIsInstance(ra, float)
        self.assertIsInstance(dec, float)
        self.assertIsInstance(response, int)
        self.assertTrue(response >= 1000000000000000000)

    def test_input_output_3(self):
        ra, dec = 100, 10
        response = id_generator(ra, dec)
        self.assertIsInstance(ra, int)
        self.assertIsInstance(dec, int)
        self.assertIsInstance(response, int)

    def test_input_output_4(self):
        ra, dec = "100", 10

        with self.assertRaises(Exception) as context:
            id_generator(ra, dec)
        self.assertIsInstance(ra, str)
        self.assertIsInstance(dec, int)
        self.assertIsInstance(context.exception, TypeError)
