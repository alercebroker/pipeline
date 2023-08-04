from turbofats.Base import Base
import GPy


class GP_DRW_sigma(Base):
    """
    Based on Matthew Graham's method to model DRW with gaussian process.
    """
    def __init__(self, shared_data):
        super().__init__(shared_data)
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

        m['.*het_Gauss.variance'] = abs(err ** 2.)[:, None]  # Set the noise parameters to the error in Y
        m.het_Gauss.variance.fix()  # We can fix the noise term, since we already know it
        m.optimize()
        pars = [m.OU.variance.values[0], m.OU.lengthscale.values[0]]  # sigma^2, tau

        sigmaDRW = pars[0]
        tauDRW = pars[1]
        self.shared_data['tauDRW'] = tauDRW
        return sigmaDRW


class GP_DRW_tau(Base):
    def __init__(self, shared_data):
        super().__init__(shared_data)
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        try:
            tauDRW = self.shared_data['tauDRW']
            return tauDRW
        except:
            print("error: please run GP_DRW_sigma first to generate values for GP_DRW_tau")
