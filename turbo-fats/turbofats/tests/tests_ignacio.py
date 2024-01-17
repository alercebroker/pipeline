import unittest
from turbofats.FeatureFunctionLib import Mean
from turbofats import FeatureSpace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.lc_data = pd.read_pickle('ZTF18aaiopei_detections.pkl')
        self.lc_g = self.lc_data[self.lc_data.fid == 1]
        self.lc_r = self.lc_data[self.lc_data.fid == 2]
        self.lc_g_np = self.lc_g[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values.T

    def testMean(self):
        mean_computer = Mean(shared_data=None)
        mean = mean_computer.fit(self.lc_g_np)
        expected_mean = 17.371986
        self.assertTrue(np.abs(mean - expected_mean) < 0.01)


class TestNewFeatureSpace(unittest.TestCase):
    def setUp(self):
        self.lc_data = pd.read_pickle('ZTF18aaiopei_detections.pkl')
        self.lc_g = self.lc_data[self.lc_data.fid == 1]
        self.lc_r = self.lc_data[self.lc_data.fid == 2]
        self.lc_g_np = self.lc_g[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values.T

    def test1(self):
        feature_list = [
            'Amplitude', 'AndersonDarling', 'Autocor_length',
            'Beyond1Std',
            'Con', 'Eta_e',
            'Gskew',
            'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
            'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude', 'Q31',
            'PeriodLS_v2',
            'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs',
            'Skew', 'SmallKurtosis', 'Std',
            'StetsonK', 'Harmonics',
            'Pvar', 'ExcessVar',
            'GP_DRW_sigma', 'GP_DRW_tau', 'SF_ML_amplitude', 'SF_ML_gamma',
            'IAR_phi',
            'LinearTrend'
        ]
        new_feature_space = FeatureSpace(
            feature_list,
            data_column_names=['magpsf_corr', 'mjd', 'sigmapsf_corr']
        )
        features = new_feature_space.calculate_features(self.lc_g)
        print(features.iloc[0])
        expected_mean = 17.371986
        mean = features['Mean'][0]
        self.assertTrue(np.abs(mean - expected_mean) < 0.01)
        period = features['PeriodLS_v2'][0]
        do_plots = False
        if do_plots:
            plt.subplot(2, 1, 1)
            plt.errorbar(
                self.lc_g['mjd'] % period,
                self.lc_g['magpsf_corr'],
                yerr=self.lc_g['sigmapsf_corr'],
                fmt='*g')

            t = np.linspace(0, period, 100)
            y = features['Harmonics_mag_1'][0]*np.cos(2*np.pi*t/period)
            for harmonic in range(2, 8):
                y += features['Harmonics_mag_%d' % harmonic][0]*np.cos(harmonic*2*np.pi*t/period - features['Harmonics_phase_%d' % harmonic][0])
            plt.subplot(2, 1, 2)
            plt.plot(t, y)
            plt.show()


if __name__ == '__main__':
    unittest.main()
