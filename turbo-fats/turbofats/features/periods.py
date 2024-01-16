from turbofats.Base import Base
from turbofats import lomb
import numpy as np


class PeriodLS_v2(Base):
    def __init__(self, shared_data, ofac=6.0):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']
        self.ofac = ofac

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        fx_v2, fy_v2, period_v2, prob_v2 = lomb.fasper(
            time,
            magnitude,
            error,
            self.ofac,
            100.0,
            fmin=0.0,
            fmax=20.0)

        new_time_v2 = np.mod(time, 2 * period_v2) / (2 * period_v2)
        self.shared_data['new_time_v2'] = new_time_v2
        self.shared_data['prob_v2'] = prob_v2
        self.shared_data['period_v2'] = period_v2

        jmax_v2 = np.nanargmax(fy_v2)
        frac_big = fy_v2[jmax_v2]
        frac_small = fy_v2[int(jmax_v2/2)]
        power_rate = frac_small/frac_big
        self.shared_data['power_rate'] = power_rate

        return period_v2


class PeriodPowerRate(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        try:
            power_rate = self.shared_data['power_rate']
            return power_rate
        except:
            print("error: please run PeriodLS_v2 first to generate values for Period_power_rate")


class Period_fit_v2(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time']

    def fit(self, data):
        try:
            prob_v2 = self.shared_data['prob_v2']
            return prob_v2
        except:
            print("error: please run PeriodLS_v2 first to generate values for Period_fit_v2")
