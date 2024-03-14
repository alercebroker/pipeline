from turbofats.Base import Base
import numpy as np


class Harmonics(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']
        self.n_harmonics = 7  # HARD-CODED

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]+10**-2

        try:
            period_v2 = self.shared_data['period_v2']
            best_freq = 1/period_v2
        except:
            raise Exception("error: please run PeriodLS_v2 first to generate values for Harmonics")

        Omega = [np.array([[1.]*len(time)])]
        timefreq = (2.0*np.pi*best_freq*np.arange(1, self.n_harmonics+1)).reshape(1, -1).T*time
        Omega.append(np.cos(timefreq))
        Omega.append(np.sin(timefreq))
        Omega = np.concatenate(Omega, axis=0).T  # Omega.shape == (lc_length, 1+2*self.n_harmonics)
        inverr = 1.0/error

        # weighted regularized linear regression
        wA = inverr.reshape(-1, 1)*Omega
        wB = (magnitude*inverr).reshape(-1, 1)
        coeffs = np.matmul(np.linalg.pinv(wA), wB).flatten()
        fitted_magnitude = np.dot(Omega, coeffs)
        coef_cos = coeffs[1:self.n_harmonics+1]
        coef_sin = coeffs[self.n_harmonics+1:]
        coef_mag = np.sqrt(coef_cos**2 + coef_sin**2)
        coef_phi = np.arctan2(coef_sin, coef_cos)

        # Relative phase
        coef_phi = coef_phi - coef_phi[0]*np.arange(1, self.n_harmonics+1)
        coef_phi = coef_phi[1:] % (2*np.pi)

        mse = np.mean((fitted_magnitude - magnitude)**2)
        return np.concatenate([coef_mag, coef_phi, np.array([mse])]).tolist()

    def is1d(self):
        return False

    def get_feature_names(self):
        feature_names = ['Harmonics_mag_%d' % (i+1) for i in range(self.n_harmonics)]
        feature_names += ['Harmonics_phase_%d' % (i+1) for i in range(1, self.n_harmonics)]
        feature_names.append('Harmonics_mse')
        return feature_names
