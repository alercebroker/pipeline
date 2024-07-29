import unittest
import numpy as np
from lc_classifier.utils import mag2flux, flux2mag
from lc_classifier.utils import mag_err_2_flux_err
from lc_classifier.utils import flux_err_2_mag_err


class TestMagFluxConversion(unittest.TestCase):
    def test_back_and_forth(self):
        a = np.logspace(-5, 10, 20)
        output = mag2flux(flux2mag(a))
        assert np.all((np.abs(a - output) / a) < 0.01)

        a = np.linspace(-5, 50, 20)
        output = flux2mag(mag2flux(a))
        assert np.all((np.abs(a - output) / a) < 0.01)

    def test_flux_err_2_mag_err(self):
        fluxes = np.logspace(-5, 10, 20)
        flux_errors = fluxes * 0.01

        mag1 = flux2mag(fluxes + flux_errors)
        mag2 = flux2mag(fluxes)

        mag_errors = flux_err_2_mag_err(flux_errors, fluxes)
        assert np.all(np.abs(((mag2 - mag1) - mag_errors) / mag_errors) < 0.01)

    def test_mag_err_2_flux_err(self):
        magnitudes = np.linspace(10, 40, 20)
        mag_error = 0.01

        flux1 = mag2flux(magnitudes)
        flux2 = mag2flux(magnitudes + mag_error)

        flux_errors = mag_err_2_flux_err(mag_error, magnitudes)
        assert np.all(np.abs(((flux1 - flux2) - flux_errors) / flux_errors) < 0.01)

    def test_err_back_and_forth(self):
        magnitudes = np.linspace(10, 40, 20)
        mag_error = 0.01

        fluxes = mag2flux(magnitudes)
        flux_errors = mag_err_2_flux_err(mag_error, magnitudes)

        mag_err_back = flux_err_2_mag_err(flux_errors, fluxes)
        assert np.all(np.abs((mag_err_back - mag_error) / mag_error) < 0.01)
