import numpy as np
import pandas as pd

from unittest.mock import MagicMock

from magstats_step.strategies.magstats_computer import MagstatsComputer
from magstats_step.strategies.ztf_strategy import ZTFMagstatsStrategy
from magstats_step.strategies.atlas_strategy import ATLASMagstatsStrategy

from data.messages import data

def test_context_set_strategy():
    context = MagstatsComputer(ZTFMagstatsStrategy())
    context.strategy = ATLASMagstatsStrategy()
    assert isinstance(context.strategy, ATLASMagstatsStrategy)


def test_context_compute_magstasts():
    strategy = MagicMock(ZTFMagstatsStrategy)
    context = MagstatsComputer(strategy)
    context.compute_magstats([], [])
    strategy.compute_magstats.assert_called_with([], [])


def test_ztf_strategy():
    pass

def test_compute_stellar():
    df_min = pd.Series({'extra_fields' : {'distnr': np.nan,
                                          'distpsnr1': np.nan,
                                          'sgscore1': np.nan,
                                          'chinr': np.nan,
                                          'sharpnr': np.nan
                                          }}
                       )
    strategy = ZTFMagstatsStrategy()
    response = strategy.compute_stellar(df_min)
    assert not response["nearZTF"]
    assert not response["nearPS1"]
    assert not response["stellarZTF"]
    assert not response["stellarPS1"]
    assert not response["stellar"]

def test_ndet():
    df = pd.DataFrame([1, 2, 3])
    strategy = ZTFMagstatsStrategy()
    ndet = strategy.compute_ndet(df)
    assert ndet == 3

def test_ndubious():
    df = pd.DataFrame([True, False, True], columns=['dubious'])
    strategy = ZTFMagstatsStrategy()
    ndubious = strategy.compute_ndubious(df)
    assert ndubious== 2

def test_nrfid():
    df = pd.DataFrame([0, 1, 1, np.nan], columns=['rfid'])
    strategy = ZTFMagstatsStrategy()
    nrfid = strategy.compute_nrfid(df)
    assert nrfid == 2

def test_compute_magnitude_statistics_corr():
    df = pd.DataFrame([[2, 0],[1, 1]], columns=['mag', 'e_mag'])
    strategy = ZTFMagstatsStrategy()
    magnitude_stats = strategy.compute_magnitude_statistics(df, corr=True)
    assert 'mag_mean_corr' in magnitude_stats
    assert 'mag_median_corr' in magnitude_stats
    assert 'mag_max_corr' in magnitude_stats
    assert 'mag_min_corr' in magnitude_stats
    assert 'mag_sigma_corr' in magnitude_stats
    assert 'mag_first_corr' in magnitude_stats
    assert 'e_mag_first_corr' in magnitude_stats
    assert 'mag_last_corr' in magnitude_stats

def test_compute_magnitude_statistics():
    df = pd.DataFrame([[2, 0],[1, 1]], columns=['mag', 'e_mag'])
    strategy = ZTFMagstatsStrategy()
    magnitude_stats = strategy.compute_magnitude_statistics(df, corr=False)
    assert 'mag_mean' in magnitude_stats
    assert 'mag_median' in magnitude_stats
    assert 'mag_max' in magnitude_stats
    assert 'mag_min' in magnitude_stats
    assert 'mag_sigma' in magnitude_stats
    assert 'mag_first' in magnitude_stats
    assert 'e_mag_first' in magnitude_stats
    assert 'mag_last' in magnitude_stats

def test_compute_magnitude_statistics_ap():
    df = pd.DataFrame([[2, 0],[1, 1]], columns=['mag', 'e_mag'])
    strategy = ZTFMagstatsStrategy()
    magnitude_stats = strategy.compute_magnitude_statistics(df, corr=False, magtype='magap')
    assert 'magap_mean' in magnitude_stats
    assert 'magap_median' in magnitude_stats
    assert 'magap_max' in magnitude_stats
    assert 'magap_min' in magnitude_stats
    assert 'magap_sigma' in magnitude_stats
    assert 'magap_first' in magnitude_stats
    assert 'e_magap_first' in magnitude_stats
    assert 'magap_last' in magnitude_stats

def test_compute_time_statistics():
    df = pd.DataFrame([0, 1, 2, 3], columns=['mjd'])
    strategy = ZTFMagstatsStrategy()
    time_stats = strategy.compute_time_statistics(df)
    assert time_stats['first_mjd'] == 0
    assert time_stats['last_mjd'] == 3

def test_compute_saturation_rate():
    df = pd.DataFrame([10, 11, 14, 15], columns=['mag'])
    strategy = ZTFMagstatsStrategy()
    sat_rate = strategy.compute_saturation_rate(df)
    assert sat_rate == 0.5

def test_dmdt():
    sample = data[0]

    strategy = ZTFMagstatsStrategy()
    detections = pd.DataFrame(sample['detections'])
    non_detections = pd.DataFrame(sample['non_detections'])
    time_stats = strategy.compute_time_statistics(detections)
    mag_stats = strategy.compute_magnitude_statistics(detections, corr=False)

    dmdt = strategy.calculate_dmdt(non_detections, dict(**time_stats, **mag_stats), dt_min=0.05)
    assert "dmdt_first" in dmdt
    assert "dm_first" in dmdt
    assert "sigmadm_first" in dmdt
    assert "dt_first" in dmdt


