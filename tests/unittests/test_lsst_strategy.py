import numpy as np
import pandas as pd

from correction.core.strategy import lsst


def test_lsst_strategy_corrected_always_true():
    detections = pd.DataFrame({"dummy": np.zeros(10)})
    corrected = lsst.is_corrected(detections)
    assert corrected.all()


def test_lsst_strategy_dubious_is_always_false():
    detections = pd.DataFrame({"dummy": np.zeros(10)})
    dubious = lsst.is_dubious(detections)
    assert not dubious.any()


def test_lsst_strategy_correction_applies_expected_factor():
    detections = pd.DataFrame.from_records({"mag": [1.0], "e_mag": [0.1]})
    corrected = lsst.correct(detections)

    assert np.isclose(corrected["mag_corr"], 1.0 * 10 ** -(3.9 / 2.5))
    assert np.isclose(corrected["e_mag_corr"], 0.1 * 10 ** -(3.9 / 2.5))
    assert np.isclose(corrected["e_mag_corr_ext"], 0.1 * 10 ** -(3.9 / 2.5))
