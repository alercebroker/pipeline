import math

from numba import jit
import warnings

import numpy as np

from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from statsmodels.tsa import stattools
from scipy.interpolate import interp1d
from scipy.stats import chi2
import GPy

from .Base import Base
from . import lomb


class Amplitude(Base):
    """Half the difference between the maximum and the minimum magnitude"""

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        return (np.median(sorted_mag[int(-math.ceil(0.05 * N)):]) -
                np.median(sorted_mag[0:int(math.ceil(0.05 * N))])) / 2.0
        # return sorted_mag[10]


class Rcs(Base):
    """Range of cumulative sum"""

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R


class StetsonK(Base):
    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]
        N = len(magnitude)

        mean_mag = (np.sum(magnitude/(error*error))/np.sum(1.0 /(error * error)))

        #mean_mag = (np.sum(magnitude/(error/np.sqrt(N)))/np.sum(1.0 /(error/np.sqrt(N))))#Check if errors are stds or variance
        #weights = (1/error**2)/np.sum(1/error**2)
        #mean_mag = np.sum(weights*magnitude)

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude - mean_mag) / error)

        K = (1 / np.sqrt(N * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
        #K = (1 / N) * np.sum(np.abs(sigmap)) / (np.sqrt((1 / N) * np.sum(sigmap ** 2)))


        return K


class Meanvariance(Base):
    """variability index"""
    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return np.std(magnitude) / np.mean(magnitude)


class Autocor_length(Base):

    def __init__(self, lags=100):
        self.Data = ['magnitude']
        self.nlags = lags

    def fit(self, data):

        magnitude = data[0]
        AC = stattools.acf(magnitude, nlags=self.nlags, fft=False)

        k = next((index for index, value in
                 enumerate(AC) if value < np.exp(-1)), None)

        while k is None:
            if self.nlags > len(magnitude):
                warnings.warn('Setting autocorrelation length as light curve length')
                return len(magnitude)
            self.nlags = self.nlags + 100
            AC = stattools.acf(magnitude, nlags=self.nlags, fft=False)
            k = next((index for index, value in
                      enumerate(AC) if value < np.exp(-1)), None)

        return k


class SlottedA_length(Base):

    def __init__(self, T=-99):
        """
        lc: MACHO lightcurve in a pandas DataFrame
        k: lag (default: 1)
        T: tau (slot size in days. default: 4)
        """
        self.Data = ['magnitude', 'time']

        SlottedA_length.SAC = []

        self.T = T

    def slotted_autocorrelation(self, data, time, T, K,
                                second_round=False, K1=100):

        slots = np.zeros((K, 1))
        i = 1

        # make time start from 0
        time = time - np.min(time)

        # subtract mean from mag values
        m = np.mean(data)
        data = data - m

        prod = np.zeros((K, 1))
        pairs = np.subtract.outer(time, time)
        pairs[np.tril_indices_from(pairs)] = 10000000

        ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

        # We calculate the slotted autocorrelation for k=0 separately
        idx = np.where(ks == 0)
        prod[0] = ((sum(data ** 2) + sum(data[idx[0]] *
                   data[idx[1]])) / (len(idx[0]) + len(data)))
        slots[0] = 0

        # We calculate it for the rest of the ks
        if second_round is False:
            for k in np.arange(1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
        else:
            for k in np.arange(K1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i - 1] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
            np.trim_zeros(prod, trim='b')

        slots = np.trim_zeros(slots, trim='b')
        return prod / prod[0], np.int64(slots).flatten()

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        N = len(time)

        if self.T == -99:
            deltaT = time[1:] - time[:-1]
            sorted_deltaT = np.sort(deltaT)
            self.T = sorted_deltaT[int(N * 0.05)+1]

        K = 100

        [SAC, slots] = self.slotted_autocorrelation(magnitude, time, self.T, K)
        # SlottedA_length.SAC = SAC
        # SlottedA_length.slots = slots

        SAC2 = SAC[slots]
        SlottedA_length.autocor_vector = SAC2

        k = next((index for index, value in
                 enumerate(SAC2) if value < np.exp(-1)), None)

        while k is None:
            K = K+K

            if K > (np.max(time) - np.min(time)) / self.T:
                break
            else:
                [SAC, slots] = self.slotted_autocorrelation(magnitude,
                                                            time, self.T, K,
                                                            second_round=True,
                                                            K1=K/2)
                SAC2 = SAC[slots]
                k = next((index for index, value in
                         enumerate(SAC2) if value < np.exp(-1)), None)

        return slots[k] * self.T

    def getAtt(self):
        # return SlottedA_length.SAC, SlottedA_length.slots
        return SlottedA_length.autocor_vector


class StetsonK_AC(SlottedA_length):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:

            a = StetsonK_AC()
            # [autocor_vector, slots] = a.getAtt()
            autocor_vector = a.getAtt()

            # autocor_vector = autocor_vector[slots]
            N_autocor = len(autocor_vector)
            sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) *
                      (autocor_vector - np.mean(autocor_vector)) /
                      np.std(autocor_vector))

            K = (1 / np.sqrt(N_autocor * 1.0) *
                 np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

            return K

        except:

            print("error: please run SlottedA_length first to generate values for StetsonK_AC ")


class StetsonL(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time', 'error', 'magnitude2', 'error2']

    def fit(self, data):

        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_error = data[7]
        aligned_error2 = data[8]

        N = len(aligned_magnitude)

        mean_mag = (np.sum(aligned_magnitude/(aligned_error*aligned_error)) /
                    np.sum(1.0 / (aligned_error * aligned_error)))
        mean_mag2 = (np.sum(aligned_magnitude2/(aligned_error2*aligned_error2)) /
                     np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude[:N] - mean_mag) /
                  aligned_error)

        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude2[:N] - mean_mag2) /
                  aligned_error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) *
             np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i))))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i ** 2)))

        return J * K / 0.798


class Con(Base):
    """Index introduced for selection of variable starts from OGLE database.


    To calculate Con, we counted the number of three consecutive measurements
    that are out of 2sigma range, and normalized by N-2
    Pavlos not happy
    """
    def __init__(self, consecutiveStar=3):
        self.Data = ['magnitude']

        self.consecutiveStar = consecutiveStar

    def fit(self, data):

        magnitude = data[0]
        N = len(magnitude)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - self.consecutiveStar + 1):
            flag = 0
            for j in range(self.consecutiveStar):
                if(magnitude[i + j] > m + 2 * sigma or magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return count * 1.0 / (N - self.consecutiveStar + 1)


# class VariabilityIndex(Base):

#     # Eta. Removed, it is not invariant to time sampling
#     '''
#     The index is the ratio of mean of the square of successive difference to
#     the variance of data points
#     '''
#     def __init__(self):
#         self.category='timeSeries'


#     def fit(self, data):

#         N = len(data)
#         sigma2 = np.var(data)

#         return 1.0/((N-1)*sigma2) * np.sum(np.power(data[1:] - data[:-1] , 2)
#      )


class Color(Base):
    """Average color for each MACHO lightcurve
    mean(B1) - mean(B2)
    """
    def __init__(self):
        self.Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        magnitude = data[0]
        magnitude2 = data[3]
        return np.mean(magnitude) - np.mean(magnitude2)

# The categories of the following featurs should be revised


class Beyond1Std(Base):
    """Percentage of points beyond one st. dev. from the weighted
    (by photometric errors) mean
    """

    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):

        magnitude = data[0]
        error = data[2]
        n = len(magnitude)

        weighted_mean = np.average(magnitude, weights=1 / error ** 2)

        # Standard deviation with respect to the weighted mean

        var = sum((magnitude - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                     magnitude < weighted_mean - std))

        return float(count) / n


class SmallKurtosis(Base):
    """Small sample kurtosis of the magnitudes.

    See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    """

    def __init__(self):
        self.category = 'basic'
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        n = len(magnitude)
        mean = np.mean(magnitude)
        std = np.std(magnitude)

        S = sum(((magnitude - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return c1 * S - c2


class Std(Base):
    """Standard deviation of the magnitudes"""

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return np.std(magnitude)


class Skew(Base):
    """Skewness of the magnitudes"""

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return stats.skew(magnitude)


class StetsonJ(Base):
    """Stetson (1996) variability index, a robust standard deviation"""

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error', 'magnitude2', 'error2']

    #lc fields are [data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd]
    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_error = data[7]
        aligned_error2 = data[8]
        N = len(aligned_magnitude)

        mean_mag = (np.sum(aligned_magnitude/(aligned_error*aligned_error)) /
                    np.sum(1.0 / (aligned_error * aligned_error)))

        mean_mag2 = (np.sum(aligned_magnitude2 / (aligned_error2*aligned_error2)) /
                     np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude[:N] - mean_mag) /
                  aligned_error)
        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude2[:N] - mean_mag2) /
                  aligned_error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) * np.sum(np.sign(sigma_i) *
             np.sqrt(np.abs(sigma_i))))

        return J


class MaxSlope(Base):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)
    """

    def __init__(self):
        self.Data = ['magnitude', 'time']

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
        np.max(slope)

        return np.max(slope)


class MedianAbsDev(Base):

    def __init__(self):
        self.category = 'basic'
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)

        devs = (abs(magnitude - median))

        return np.median(devs)


class MedianBRP(Base):
    """Median buffer range percentage

    Fraction (<= 1) of photometric points within amplitude/10
    of the median magnitude
    """

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)
        amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
        n = len(magnitude)

        count = np.sum(np.logical_and(magnitude < median + amplitude,
                                      magnitude > median - amplitude))

        return float(count) / n


class PairSlopeTrend(Base):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    """

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        data_last = magnitude[-30:]

        return (float(len(np.where(np.diff(data_last) > 0)[0]) -
                len(np.where(np.diff(data_last) <= 0)[0])) / 30)


class FluxPercentileRatioMid20(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_60_index = math.ceil(0.60 * lc_length)
        F_40_index = math.ceil(0.40 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return F_mid20


class FluxPercentileRatioMid35(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_325_index = math.ceil(0.325 * lc_length)
        F_675_index = math.ceil(0.675 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return F_mid35


class FluxPercentileRatioMid50(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_25_index = math.ceil(0.25 * lc_length)
        F_75_index = math.ceil(0.75 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return F_mid50


class FluxPercentileRatioMid65(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_175_index = math.ceil(0.175 * lc_length)
        F_825_index = math.ceil(0.825 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return F_mid65


class FluxPercentileRatioMid80(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_10_index = math.ceil(0.10 * lc_length)
        F_90_index = math.ceil(0.90 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.floor(0.95 * lc_length)

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return F_mid80


class PercentDifferenceFluxPercentile(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)

        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        return percent_difference


class PercentAmplitude(Base):

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)
        distance_median = np.abs(magnitude - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        return percent_amplitude


class LinearTrend(Base):

    def __init__(self):
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        regression_slope = stats.linregress(time, magnitude)[0]

        return regression_slope


class Eta_color(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_time = data[6]
        N = len(aligned_magnitude)
        B_Rdata = aligned_magnitude - aligned_magnitude2

        w = 1.0 / np.power(aligned_time[1:] - aligned_time[:-1], 2)
        w_mean = np.mean(w)

        N = len(aligned_time)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (1 / sigma2) * (S1 / S2)


        return eta_B_R


class Eta_e(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        w_mean = np.mean(w)

        N = len(time)
        sigma2 = np.var(magnitude)

        S1 = np.sum(w * np.power(np.subtract(magnitude[1:], magnitude[:-1]), 2))
        S2 = np.sum(w)

        eta_e = (1 / sigma2) * (S1 / S2)
        #(sigma2 * S2 * N ** 2)

        return eta_e


class Mean(Base):

    def __init__(self):

        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        B_mean = np.mean(magnitude)

        return B_mean


class Q31(Base):

    def __init__(self):

        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = data[0]
        return np.percentile(magnitude, 75) - np.percentile(magnitude, 25)


class Q31_color(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        N = len(aligned_magnitude)
        b_r = aligned_magnitude[:N] - aligned_magnitude2[:N]

        return np.percentile(b_r, 75) - np.percentile(b_r, 25)


class AndersonDarling(Base):

    def __init__(self):

        self.Data = ['magnitude']

    def fit(self, data):

        magnitude = data[0]
        ander = stats.anderson(magnitude)[0]
        return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))


class PeriodLS(Base):

    def __init__(self, ofac=6.):

        self.Data = ['magnitude', 'time', 'error']
        self.ofac = ofac

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        error = data[2]

        global new_time
        global prob
        global period

        fx, fy, jmax, prob = lomb.fasper(time, magnitude, error, self.ofac, 100.)
        period = fx[jmax]
        T = 1.0 / period
        new_time = np.mod(time, 2 * T) / (2 * T)

        return T


class PeriodLS_v2(Base):
    def __init__(self, ofac=6.0):
        self.Data = ['magnitude', 'time', 'error']
        self.ofac = ofac

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        global new_time_v2
        global prob_v2
        global period_v2

        fx_v2, fy_v2, jmax_v2, prob_v2 = lomb.fasper(time, magnitude, error, self.ofac, 100.0,
                                         fmin=0.0,
                                         fmax=20.0)
        period_v2 = fx_v2[jmax_v2]
        T_v2 = 1.0 / period_v2
        new_time_v2 = np.mod(time, 2 * T_v2) / (2 * T_v2)

        return T_v2

class Period_fit(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        try:
            return prob
        except:
            print("error: please run PeriodLS first to generate values for Period_fit")


class Period_fit_v2(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time']


    def fit(self, data):
        try:
            return prob_v2
        except:
            print("error: please run PeriodLS_v2 first to generate values for Period_fit_v2")


class Psi_CS(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        try:
            magnitude = data[0]
            time = data[1]
            folded_data = magnitude[np.argsort(new_time)]
            sigma = np.std(folded_data)
            N = len(folded_data)
            m = np.mean(folded_data)
            s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
            R = np.max(s) - np.min(s)

            return R
        except:
            print("error: please run PeriodLS first to generate values for Psi_CS")


class Psi_eta(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        # folded_time = np.sort(new_time)
        try:
            magnitude = data[0]
            folded_data = magnitude[np.argsort(new_time)]

            # w = 1.0 / np.power(folded_time[1:]-folded_time[:-1] ,2)
            # w_mean = np.mean(w)

            # N = len(folded_time)
            # sigma2=np.var(folded_data)

            # S1 = sum(w*(folded_data[1:]-folded_data[:-1])**2)
            # S2 = sum(w)

            # Psi_eta = w_mean * np.power(folded_time[N-1]-folded_time[0],2) * S1 /
            # (sigma2 * S2 * N**2)

            N = len(folded_data)
            sigma2 = np.var(folded_data)

            Psi_eta = (1.0 / ((N - 1) * sigma2) *
                       np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))

            return Psi_eta
        except:
            print("error: please run PeriodLS first to generate values for Psi_eta")


class Psi_CS_v2(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            magnitude = data[0]
            time = data[1]
            folded_data = magnitude[np.argsort(new_time_v2)]
            sigma = np.std(folded_data)
            N = len(folded_data)
            m = np.mean(folded_data)
            s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
            R = np.max(s) - np.min(s)

            return R
        except:
            print("error: please run PeriodLS_v2 first to generate values for Psi_CS_v2")


class Psi_eta_v2(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            magnitude = data[0]
            folded_data = magnitude[np.argsort(new_time_v2)]

            N = len(folded_data)
            sigma2 = np.var(folded_data)

            Psi_eta = (1.0 / ((N - 1) * sigma2) *
                       np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))

            return Psi_eta
        except:
            print("error: please run PeriodLS_v2 first to generate values for Psi_eta_v2")


@jit(nopython=True)
def car_likelihood(parameters, t, x, error_vars):
    sigma = parameters[0]
    tau = parameters[1]
    # b = parameters[1] #comment it to do 2 pars estimation
    # tau = params(1,1);
    # sigma = sqrt(2*var(x)/tau);

    b = np.mean(x) / tau
    epsilon = 1e-300
    cte_neg = -np.infty
    num_datos = len(x)

    Omega = []
    x_hat = []
    a = []
    x_ast = []

    # Omega = np.zeros((num_datos,1))
    # x_hat = np.zeros((num_datos,1))
    # a = np.zeros((num_datos,1))
    # x_ast = np.zeros((num_datos,1))

    # Omega[0]=(tau*(sigma**2))/2.
    # x_hat[0]=0.
    # a[0]=0.
    # x_ast[0]=x[0] - b*tau

    Omega.append(np.array([tau * (sigma ** 2) / 2.0]))
    x_hat.append(np.array([0.0]))
    a.append(np.array([0.0]))
    x_ast.append(x[0] - b * tau)

    loglik = np.array([0.0])

    for i in range(1, num_datos):
        a_new = np.exp(-(t[i] - t[i - 1]) / tau)
        x_ast.append(x[i] - b * tau)
        x_hat_val = (
            a_new * x_hat[i - 1] +
            (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
            (x_ast[i - 1] - x_hat[i - 1]))
        x_hat.append(x_hat_val)

        Omega.append(
            Omega[0] * (1 - (a_new ** 2)) + ((a_new ** 2)) * Omega[i - 1] *
            (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

        # x_ast[i]=x[i] - b*tau
        # x_hat[i]=a_new*x_hat[i-1] + (a_new*Omega[i-1]/(Omega[i-1] +
        # error_vars[i-1]))*(x_ast[i-1]-x_hat[i-1])
        # Omega[i]=Omega[0]*(1-(a_new**2)) + ((a_new**2))*Omega[i-1]*
        # ( 1 - (Omega[i-1]/(Omega[i-1]+ error_vars[i-1])))

        loglik_inter = np.log(
            ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
            (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
            (Omega[i] + error_vars[i]))) + epsilon))

        loglik = loglik + loglik_inter

        if(loglik[0] <= cte_neg):
            print('CAR lik --> inf')
            return None

    # the minus one is to perfor maximization using the minimize function
    return -loglik

class CAR_sigma(Base):

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def CAR_Lik(self, parameters, t, x, error_vars):
        return car_likelihood(parameters, t, x, error_vars)

    def calculateCAR(self, time, data, error):

        x0 = [0.01, 100]
        bnds = ((0, 20), (0.000001, 2000))

        global tau
        try:
            res = minimize(self.CAR_Lik, x0, args=(time, data, error),
                           method='L-BFGS-B', bounds=bnds)
            sigma = res.x[0]
            tau = res.x[1]
        except TypeError:
            warnings.warn('CAR optimizer raised an exception')
            # options={'disp': True}
            sigma = np.nan
            tau = np.nan
        return sigma

    # def getAtt(self):
    #     return CAR_sigma.tau

    def fit(self, data):
        # LC = np.hstack((self.time , data.reshape((self.N,1)), self.error))

        N = len(data[0])
        magnitude = data[0].reshape((N, 1))
        time = data[1].reshape((N, 1))
        error = data[2].reshape((N, 1)) ** 2

        a = self.calculateCAR(time, magnitude, error)

        return a


class CAR_tau(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:
            return tau
        except:
            print("error: please run CAR_sigma first to generate values for CAR_tau")


class CAR_mean(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        magnitude = data[0]

        try:
            return np.mean(magnitude) / tau
        except:
            print("error: please run CAR_sigma first to generate values for CAR_mean")


class Harmonics(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']
        self.n_harmonics = 7 # HARD-CODED
        self.penalization = 1 # HARD-CODED

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]+10**-2

        try:
            best_freq = period_v2
        except:
            print("error: please run PeriodLS_v2 first to generate values for Harmonics")

        Omega = [np.array([[1.]*len(time)])]
        timefreq = (2.0*np.pi*best_freq*np.arange(1, self.n_harmonics+1)).reshape(1, -1).T*time
        Omega.append(np.cos(timefreq))
        Omega.append(np.sin(timefreq))
        Omega = np.concatenate(Omega).T
        inverr2 = (1.0/error**2)
        #coeffs = np.linalg.lstsq(inverr2.reshape(-1, 1)*Omega, magnitude*inverr2, rcond=None)[0]

        # weighted regularized linear regression
        reg = np.array([0.]*(1+self.n_harmonics*2))
        reg[1:(self.n_harmonics+1)] = np.linspace(0, 1, self.n_harmonics)*self.penalization*len(magnitude)**2*inverr2.mean()
        reg[(self.n_harmonics+1):] = reg[1:(self.n_harmonics+1)]
        dreg = np.diag(reg)
        wA = inverr2.reshape(-1, 1)*Omega
        wB = (magnitude*inverr2).reshape(-1, 1)
        coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(wA.T, wA)+dreg), wA.T), wB).flatten()
        fitted_magnitude = np.dot(Omega, coeffs)
        coef_cos = coeffs[1:self.n_harmonics+1]
        coef_sin = coeffs[self.n_harmonics+1:]
        coef_mag = np.sqrt(coef_cos**2 + coef_sin**2)
        coef_phi = np.arctan2(coef_sin, coef_cos)

        # Relative phase
        coef_phi = coef_phi - coef_phi[0]
        coef_phi = coef_phi[1:]

        nmse = np.mean((fitted_magnitude - magnitude)**2)/np.var(error)
        return np.concatenate([coef_mag, coef_phi, np.array([nmse])]).tolist()

    def is1d(self):
        return False

    def get_feature_names(self):
        feature_names = ['Harmonics_mag_%d' % (i+1) for i in range(self.n_harmonics)]
        feature_names += ['Harmonics_phase_%d' % (i+1) for i in range(1, self.n_harmonics)]
        feature_names.append('Harmonics_mse')
        return feature_names

class Freq1_harmonics_amplitude_0(Base):
    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]


        time = time - np.min(time)

        global A
        global PH
        global scaledPH
        A = []
        PH = []
        scaledPH = []

        def model(x, a, b, c, Freq):
            return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c

        for i in range(3):
            wk1, wk2, jmax, prob = lomb.fasper(time, magnitude, error, 6., 100.)
            fundamental_Freq = wk1[jmax]

            # fit to a_i sin(2pi f_i t) + b_i cos(2 pi f_i t) + b_i,o

            # a, b are the parameters we care about
            # c is a constant offset
            # f is the fundamental Frequency
            def yfunc(Freq):
                def func(x, a, b, c):
                    return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c
                return func

            Atemp = []
            PHtemp = []
            popts = []

            for j in range(4):
                popt, pcov = curve_fit(yfunc((j+1)*fundamental_Freq), time, magnitude)
                Atemp.append(np.sqrt(popt[0]**2+popt[1]**2))
                PHtemp.append(np.arctan(popt[1] / popt[0]))
                popts.append(popt)

            A.append(Atemp)
            PH.append(PHtemp)

            for j in range(4):
                magnitude = np.array(magnitude) - model(time, popts[j][0], popts[j][1], popts[j][2], (j+1)*fundamental_Freq)

        for ph in PH:
            scaledPH.append(np.array(ph) - ph[0])

        return A[0][0]


class Freq1_harmonics_amplitude_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        try:
            return A[0][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_amplitude_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):

        try:
            return A[0][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_amplitude_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[0][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_amplitude_0(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[1][0]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_amplitude_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[1][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_amplitude_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[1][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_amplitude_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[1][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_amplitude_0(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[2][0]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_amplitude_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[2][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_amplitude_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[2][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_amplitude_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return A[2][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_rel_phase_0(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[0][0]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_rel_phase_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[0][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_rel_phase_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[0][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq1_harmonics_rel_phase_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[0][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_rel_phase_0(Base):
    def __init__(self):
        self.category = 'timeSeries'

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[1][0]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_rel_phase_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[1][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_rel_phase_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[1][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq2_harmonics_rel_phase_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[1][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_rel_phase_0(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[2][0]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_rel_phase_1(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[2][1]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_rel_phase_2(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[2][2]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Freq3_harmonics_rel_phase_3(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return scaledPH[2][3]
        except:
            print("error: please run Freq1_harmonics_amplitude_0 first to generate values for all harmonics")


class Gskew(Base):
    """Median-based measure of the skew"""

    def __init__(self):
        self.Data = ['magnitude']

    def fit(self, data):
        magnitude = np.array(data[0])
        median_mag = np.median(magnitude)
        F_3_value = np.percentile(magnitude, 3)
        F_97_value = np.percentile(magnitude, 97)

        return (np.median(magnitude[magnitude <= F_3_value]) +
                np.median(magnitude[magnitude >= F_97_value])
                - 2*median_mag)


class StructureFunction_index_21(Base):

    def __init__(self):
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]

        global m_21
        global m_31
        global m_32

        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, magnitude)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
            sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 1.0))
            sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 2.0)))
            sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

        return m_21


class StructureFunction_index_31(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return m_31
        except:
            print("error: please run StructureFunction_index_21 first to generate values for all Structure Function")


class StructureFunction_index_32(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            return m_32
        except:
            print("error: please run StructureFunction_index_21 first to generate values for all Structure Function")



class Pvar(Base):
    """
    Calculate the probability of a light curve to be variable.
    """
    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]


        mean_mag = np.mean(magnitude)
        nepochs = float(len(magnitude))

        chi = np.sum( (magnitude - mean_mag)**2. / error**2. )
        p_chi = chi2.cdf(chi,(nepochs-1))

        return p_chi




class ExcessVar(Base):
    """
    Calculate the excess variance,which is a measure of the intrinsic variability amplitude.
    """
    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]

        mean_mag=np.mean(magnitude)
        nepochs=float(len(magnitude))

        a = (magnitude-mean_mag)**2
        ex_var = (np.sum(a-error**2)/((nepochs*(mean_mag**2))))

        return ex_var



class GP_DRW_sigma(Base):
    """
    Based on Matthew Graham's method to model DRW with gaussian process.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        magnitude = data[0]
        t = data[1]
        err = data[2]


        mag = magnitude-magnitude.mean()
        kern = GPy.kern.OU(1)
        m = GPy.models.GPHeteroscedasticRegression(t[:, None], mag[:, None], kern)

        # DeprecationWarning:Assigning the 'data' attribute is an inherently
        # unsafe operation and will be removed in the future.

        m['.*het_Gauss.variance'] = abs(err ** 2.)[:, None] # Set the noise parameters to the error in Y
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
        m.optimize()
        pars = [m.OU.variance.values[0], m.OU.lengthscale.values[0]] # sigma^2, tau

        sigmaDRW = pars[0]
        global tauDRW
        tauDRW = pars[1]
        return sigmaDRW


class GP_DRW_tau(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:
            return tauDRW
        except:
            print("error: please run GP_DRW_sigma first to generate values for GP_DRW_tau")


@jit(nopython=True)
def SFarray(jd, mag, err):
    """
    calculate an array with (m(ti)-m(tj)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt

    inputs:
    jd: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    outputs:
    tauarray: array with the difference in time (ti-tj)
    sfarray: array with |m(ti)-m(tj)|
    errarray: array with err(ti)^2+err(tj)^2
    """

    sfarray = []
    tauarray = []
    errarray = []
    err_squared = err**2
    len_mag = len(mag)
    for i in range(len_mag):
        for j in range(i+1, len_mag):
            dm = mag[i] - mag[j]
            sigma = err_squared[i] + err_squared[j]
            dt = jd[j] - jd[i]
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray = np.array(sfarray)
    tauarray = np.array(tauarray)
    errarray = np.array(errarray)
    return (tauarray, sfarray, errarray)


class SF_ML_amplitude(Base):
    """
    Fit the model A*tau^gamma to the SF, finding the maximum value of the likelihood.
    Based on Schmidt et al. 2010.
    """
    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def bincalc(self,nbin=0.1,bmin=5,bmax=2000):
        """
        calculate the bin range, in logscale

        inputs:
        nbin: size of the bin in log scale
        bmin: minimum value of the bins
        bmax: maximum value of the bins

        output: bins array
        """

        logbmin=np.log10(bmin)
        logbmax=np.log10(bmax)

        logbins=np.arange(logbmin,logbmax,nbin)

        bins=10**logbins

        #bins=np.linspace(bmin,bmax,60)
        return (bins)


    def SF_formula(self,jd,mag,errmag,nbin=0.1,bmin=5,bmax=2000):


        dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)
        ndt=np.where((dtarray<=365) & (dtarray>=5))
        dtarray=dtarray[ndt]
        dmagarray=dmagarray[ndt]
        sigmaarray=sigmaarray[ndt]

        bins=self.bincalc(nbin,bmin,bmax)

        sf_list=[]
        tau_list=[]
        numobj_list=[]

        for i in range(0,len(bins)-1):
            n=np.where((dtarray>=bins[i]) & (dtarray<bins[i+1]))
            nobjbin=len(n[0])
            if nobjbin>=1:
                dmag1=(dmagarray[n])**2
                derr1=(sigmaarray[n])
                sf=(dmag1-derr1)
                sff=np.sqrt(np.mean(sf))
                sf_list.append(sff)
                numobj_list.append(nobjbin)
                #central tau for the bin
                tau_list.append((bins[i]+bins[i+1])*0.5)


        SF=np.array(sf_list)
        nob=np.array(numobj_list)
        tau=np.array(tau_list)
        nn=np.where((nob>0) & (SF>-99))
        tau=tau[nn]
        SF=SF[nn]

        if (len(SF)<2):
            tau = np.array([-99])
            SF = np.array([-99])


        return (tau/365.,SF)


    def fit(self,data):
        
        mag = data[0]
        t = data[1]
        err = data[2]

        tau,sf = self.SF_formula(t,mag,err)

        if tau[0] == -99:
            A = -0.5
            gamma = -0.5

        else:

            y=np.log10(sf)
            x=np.log10(tau)
            x=x[np.where((tau<=0.5) & (tau>0.01))]
            y=y[np.where((tau<=0.5) & (tau>0.01))]

            try :
                coefficients = np.polyfit(x, y, 1)
                A=10**(coefficients[1])
                gamma = coefficients[0]

                if A<0.005:
                    A = 0.0
                    gamma = 0.0
                elif A>15: A = 15
                if gamma>3: gamma = 3
                elif gamma<-0.5 : gamma =-0.5


            except:
                A = -0.5
                gamma = -0.5

        A_sf = A
        global g_sf
        g_sf = gamma

        return(A_sf)

class SF_ML_gamma(Base):
    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:
            return g_sf
        except:
            print("error: please run SF_amplitude first to generate values for SF_gamma")


class IAR_phi(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']



    def IAR_phi_kalman(self,x,t,y,yerr,standarized=True,c=0.5):
        n=len(y)
        Sighat=np.zeros(shape=(1,1))
        Sighat[0,0]=1
        if standarized == False:
             Sighat=np.var(y)*Sighat
        xhat=np.zeros(shape=(1,n))
        delta=np.diff(t)
        Q=Sighat
        phi=x
        F=np.zeros(shape=(1,1))
        G=np.zeros(shape=(1,1))
        G[0,0]=1
        sum_Lambda=0
        sum_error=0
        if np.isnan(phi) == True:
            phi=1.1
        if abs(phi) < 1:
            for i in range(n-1):
                Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2
                if (Lambda <= 0) or (np.isnan(Lambda) == True):
                    sum_Lambda=n*1e10
                    break
                phi2=phi**delta[i]
                F[0,0]=phi2
                phi2=1-phi**(delta[i]*2)
                Qt=phi2*Q
                sum_Lambda=sum_Lambda+np.log(Lambda)
                Theta=np.dot(np.dot(F,Sighat),G.transpose())
                sum_error= sum_error + (y[i]-np.dot(G,xhat[0:1,i]))**2/Lambda
                xhat[0:1,i+1]=np.dot(F,xhat[0:1,i])+np.dot(np.dot(Theta,np.linalg.inv(Lambda)),(y[i]-np.dot(G,xhat[0:1,i])))
                Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,np.linalg.inv(Lambda)),Theta.transpose())
            yhat=np.dot(G,xhat)
            out=(sum_Lambda + sum_error)/n
            if np.isnan(sum_Lambda) == True:
                out=1e10
        else:
            out=1e10
        return out

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        if np.sum(error)==0:
            error=np.zeros(len(magnitude))

        ynorm = (magnitude-np.mean(magnitude))/np.sqrt(np.var(magnitude,ddof=1))
        deltanorm = error/np.sqrt(np.var(magnitude,ddof=1))

        out=minimize_scalar(self.IAR_phi_kalman,args=(time,ynorm,deltanorm),bounds=(0,1),method="bounded",options={'xatol': 1e-12, 'maxiter': 50000})

        phi = out.x
        try: phi = phi[0][0]
        except: phi = phi

        return phi


class CIAR_phiR_beta(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    (beta version)
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def CIAR_phi_kalman(self,x,t,y,yerr,mean_zero=True,standarized=True,c=0.5):
        n=len(y)
        Sighat=np.zeros(shape=(2,2))
        Sighat[0,0]=1
        Sighat[1,1]=c
        if standarized == False:
             Sighat=np.var(y)*Sighat
        if mean_zero == False:
             y=y-np.mean(y)
        xhat=np.zeros(shape=(2,n))
        delta=np.diff(t)
        Q=Sighat
        phi_R=x[0]
        phi_I=x[1]
        F=np.zeros(shape=(2,2))
        G=np.zeros(shape=(1,2))
        G[0,0]=1
        phi=complex(phi_R, phi_I)
        Phi=abs(phi)
        psi=np.arccos(phi_R/Phi)
        sum_Lambda=0
        sum_error=0
        if np.isnan(phi) == True:
            phi=1.1
        if abs(phi) < 1:
            for i in range(n-1):
                Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2
                if (Lambda <= 0) or (np.isnan(Lambda) == True):
                    sum_Lambda=n*1e10
                    break
                phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
                phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
                phi2=1-abs(phi**delta[i])**2
                F[0,0]=phi2_R
                F[0,1]=-phi2_I
                F[1,0]=phi2_I
                F[1,1]=phi2_R
                Qt=phi2*Q
                sum_Lambda=sum_Lambda+np.log(Lambda)
                Theta=np.dot(np.dot(F,Sighat),G.transpose())
                sum_error= sum_error + (y[i]-np.dot(G,xhat[0:2,i]))**2/Lambda
                xhat[0:2,i+1]=np.dot(F,xhat[0:2,i])+np.dot(np.dot(Theta,np.linalg.inv(Lambda)),(y[i]-np.dot(G,xhat[0:2,i])))
                Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,np.linalg.inv(Lambda)),Theta.transpose())
            yhat=np.dot(G,xhat)
            out=(sum_Lambda + sum_error)/n
            if np.isnan(sum_Lambda) == True:
                out=1e10
        else:
            out=1e10
        return out

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        niter=4
        seed=1234

        ynorm = (magnitude-np.mean(magnitude))/np.sqrt(np.var(magnitude,ddof=1))
        deltanorm = error/np.sqrt(np.var(magnitude,ddof=1))

        np.random.seed(seed)
        aux=1e10
        value=1e10
        br=0
        if np.sum(error)==0:
            deltanorm=np.zeros(len(y))
        for i in range(niter):
            phi_R=2*np.random.uniform(0,1,1)-1
            phi_I=2*np.random.uniform(0,1,1)-1
            bnds = ((-0.9999, 0.9999), (-0.9999, 0.9999))
            out=minimize(self.CIAR_phi_kalman,np.array([phi_R, phi_I]),args=(time,ynorm,deltanorm),bounds=bnds,method='L-BFGS-B')
            value=out.fun
            if aux > value:
                par=out.x
                aux=value
                br=br+1
            if aux <= value and br>1 and i>math.trunc(niter/2):
                break
            #print br
        if aux == 1e10:
           par=np.zeros(2)
        return par[0]
