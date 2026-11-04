import unittest
from unittest import mock

# Se importa delight_processing
from delight_step.clients.dummy_client import delight_processing

class DelightProcessingTest(unittest.TestCase):
    
    # Ejemplo de entrada JSON para probar
    sample_input_json = {
        "coordinates": [
            {"oid": 1, "ra": 10.0, "dec": 20.0},
            {"oid": 2, "ra": 30.0, "dec": 40.0}
        ],
        "filter": {},
        "calc_dispersion": True,
        "calc_galaxy_size": True
    }
    
    def test_delight_processing(self):
        # resultado de la fn con el ejemplo
        result = delight_processing(self.sample_input_json)
        
        # Verificación de formato
        self.assertIsInstance(result, list)
        
        # Verificación de estructura
        for coord in result:
            self.assertIsInstance(coord, dict)
            self.assertIn("oid", coord)
            self.assertIn("ra", coord)
            self.assertIn("dec", coord)
            self.assertIn("ra_gal", coord)
            self.assertIn("dec_gal", coord)
            self.assertIn("dispersion", coord)
            self.assertIn("galaxy_size", coord)
        
        # Verificación si la dispersión y tamaño son no nulos
        for coord in result:
            self.assertIsNotNone(coord["dispersion"])
            self.assertIsNotNone(coord["galaxy_size"])

