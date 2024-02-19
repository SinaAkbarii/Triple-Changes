import numpy as np
import warnings
from scipy.stats import norm, expon
from scipy.optimize import minimize
from utils import log_likelihood_normal, log_likelihood_exponential

""" A class for the estimation of 
    empirical cdfs and their inverses:
    """


class Empirical:
    def __init__(self, data, estimate_cdf=False, samples=None):
        """
        :param data: a dataset of scalar real numbers
        :param samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        if samples is None:
            samples = len(data)
        elif samples > len(data):
            warnings.warn("Not enough samples provided. Reducing the number of samples to the amount provided.")
            samples = len(data)
        self.data = data[:samples]
        self.n = len(self.data)
        self.yl = np.min(self.data)
        self.ym = np.max(self.data)
        self.F = dict()
        self.invF = dict()
        if estimate_cdf:
            self.init_estim()

    def init_estim(self):
        """ Estimate the cdf and its inverse.
        :return: None
        """
        sorted_data = np.sort(self.data)
        unique_values, counts = np.unique(sorted_data, return_counts=True)
        cumu_probab = np.cumsum(counts) / self.n
        self.F = dict(zip(unique_values, cumu_probab))
        self.invF = dict(zip(cumu_probab, unique_values))
        return None

    def get_cdf(self, y):
        """
        Return the cdf of y, F(y)
        :param y: value for which the cdf is returned
        :return: cdf evaluated at y.
        """
        if y < self.yl:
            return 0
        keys = np.array(list(self.F.keys()))
        return self.F[np.max(keys[keys <= y])]

    def get_cdf_all(self):
        return self.F.values()

    def get_inv_cdf(self, u):
        """
        returns the inverse of cdf evaluated at u.
        """
        assert 0 <= u <= 1
        keys = np.array(list(self.invF.keys()))
        return self.invF[np.min(keys[keys >= u])]

    def get_inv_cdf_all(self, m):
        """
        Divides (0,1) into m segments, returns the inverse cdf evaluated at these points
        """
        U = np.linspace(0, 1, m)
        inv_cdf = np.zeros(len(U))
        for i in range(len(U)):
            inv_cdf[i] = self.get_inv_cdf(U[i])
        return inv_cdf

    def get_mean(self):
        """
        :return: the estimated expectation (average of the data)
        """
        return np.mean(self.data)


class MLE:
    def __init__(self, data, distribution_class='Gaussian', samples=None):
        """
        :param data: a dataset of scalar real numbers
        :param samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        if samples is None:
            samples = len(data)
        elif samples > len(data):
            warnings.warn("Not enough samples provided. Reducing the number of samples to the amount provided.")
            samples = len(data)
        self.data = data[:samples]
        self.n = len(self.data)
        self.mean = 0
        self.dist = distribution_class
        if distribution_class in ['Gaussian', 'lognormal']:
            self.sigma = 1  # standard deviation
        elif distribution_class == 'Exponential':
            self.mle_lambda = 1
        self.estimate_mle()

    def estimate_mle(self):
        """
        :param dist_class: class of distributions. Only Gaussian, lognormal, and exponential distributions are
        supported for now.
        :return: None
        """
        # Minimize the negative log-likelihood to obtain MLE
        if self.dist == 'Gaussian':
            result = minimize(log_likelihood_normal, np.array([self.mean, self.sigma]),
                              args=(self.data,), method='L-BFGS-B')
            # Extract MLE
            self.mean, self.sigma = result.x
            self.sigma = np.abs(self.sigma)
        elif self.dist == 'Exponential':
            result = minimize(log_likelihood_exponential, np.array([self.mle_lambda]),
                              args=(self.data,), method='L-BFGS-B')
            # Extract MLE
            self.mle_lambda = result.x[0]
            self.mean = 1 / self.mle_lambda
        elif self.dist == 'lognormal':
            result = minimize(log_likelihood_normal, np.array([self.mean, self.sigma]),
                              args=(self.data, True,), method='L-BFGS-B')
            # Extract MLE
            self.mean, self.sigma = result.x
            self.sigma = np.abs(self.sigma)
        else:
            raise Exception("Maximum Likelihood estimation is supported for Gaussian and "
                            "Exponential distribution only!")

    def get_mean(self):
        return self.mean

    def get_cdf(self, y):
        if self.dist == 'Gaussian':
            cdf = norm.cdf(y, loc=self.mean, scale=self.sigma)
            if cdf >= .99999999:
                return .99999999
            elif cdf <= .00000001:
                return .00000001
            return cdf
        elif self.dist == 'lognormal':
            cdf = norm.cdf(np.log(y*10), loc=self.mean, scale=self.sigma)
            if cdf >= .99999999:
                return .99999999
            elif cdf <= .00000001:
                return .00000001
            return cdf
        elif self.dist == 'Exponential':
            return expon.cdf(y, scale=1 / self.mle_lambda)

    def get_inv_cdf(self, u):
        assert 0 <= u <= 1
        if self.dist == 'Gaussian':
            return norm.ppf(u, loc=self.mean, scale=self.sigma)
        elif self.dist == 'lognormal':
            return np.exp(norm.ppf(u, loc=self.mean, scale=self.sigma))/10
        elif self.dist == 'Exponential':
            return expon.ppf(u, scale=1 / self.mle_lambda)


class DiD:
    def __init__(self):
        self.d_treat = None
        self.d_control = None

    def set_data(self, data_treat, data_control):
        self.d_treat = data_treat
        self.d_control = data_control

    def estimate(self, samples=None, estimation_method='Empirical'):
        """
        DiD estimator for the average effect of the treatment on the treated
        :param: samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        if estimation_method != 'Empirical':
            raise Exception('Use DiD with empirical estimation only.')
        emp0_control = Empirical(self.d_control['Y(t0)'], False, samples)
        emp1_control = Empirical(self.d_control['Y(t1)'], False, samples)
        diff0 = emp1_control.get_mean() - emp0_control.get_mean()
        emp0_treat = Empirical(self.d_treat['Y(t0)'], False, samples)
        emp1_treat = Empirical(self.d_treat['Y(t1)'], False, samples)
        diff1 = emp1_treat.get_mean() - emp0_treat.get_mean()
        return diff1 - diff0


class DDD:
    def __init__(self):
        self.did_est0 = DiD()
        self.did_est1 = DiD()

    def set_data(self, data_treat_s1, data_control_s1, data_treat_s0, data_control_s0):
        self.did_est0.set_data(data_treat_s0, data_control_s0)
        self.did_est1.set_data(data_treat_s1, data_control_s1)

    def estimate(self, samples=None, estimation_method='Empirical'):
        """
        triple difference estimator for the average effect of the treatment on the treated
        :param: samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        return self.did_est1.estimate(samples, estimation_method=estimation_method) - \
               self.did_est0.estimate(samples, estimation_method=estimation_method)


class CiC:
    def __init__(self):
        self.d_treat = None
        self.d_control = None

    def set_data(self, data_treat, data_control):
        self.d_treat = data_treat
        self.d_control = data_control

    def estimate(self, samples=None, estimation_method='Empirical'):
        """
        CiC estimator for the average effect of the treatment on the treated
        :param: samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        if estimation_method == 'Empirical':
            est_control_t0 = Empirical(self.d_control['Y(t0)'], True, samples)
            est_control_t1 = Empirical(self.d_control['Y(t1)'], True, samples)
        elif estimation_method == 'Gaussian' or estimation_method == 'Exponential':
            est_control_t0 = MLE(self.d_control['Y(t0)'], estimation_method, samples)
            est_control_t1 = MLE(self.d_control['Y(t1)'], estimation_method, samples)
        else:
            raise Exception("estimator type not supported!")
        est_treat_t1 = Empirical(self.d_treat['Y(t1)'], False, samples)
        if samples is None:
            samples = len(self.d_treat['Y(t0)'])
        counterfactual_estimate = 0
        for y in self.d_treat['Y(t0)'][:samples]:
            u_1 = est_control_t0.get_cdf(y)
            y_2 = est_control_t1.get_inv_cdf(u_1)
            counterfactual_estimate += y_2
        counterfactual_estimate /= samples
        return est_treat_t1.get_mean() - counterfactual_estimate


class CCC:
    def __init__(self):
        self.data_treat_s1 = None
        self.data_control_s1 = None
        self.data_treat_s0 = None
        self.data_control_s0 = None

    def set_data(self, data_treat_s1, data_control_s1, data_treat_s0, data_control_s0):
        self.data_treat_s1 = data_treat_s1
        self.data_control_s1 = data_control_s1
        self.data_treat_s0 = data_treat_s0
        self.data_control_s0 = data_control_s0

    def estimate(self, samples=None, estimation_method='Empirical'):
        """
        triple changes estimator for the average effect of the treatment on the treated
        :param: samples: number of samples to use. if unspecified, the whole dataset is used.
        """
        if estimation_method == 'Empirical':
            est_control_s0_t0 = Empirical(self.data_control_s0['Y(t0)'], True, samples)
            est_control_s0_t1 = Empirical(self.data_control_s0['Y(t1)'], True, samples)
            est_treat_s0_t0 = Empirical(self.data_treat_s0['Y(t0)'], True, samples)
            est_treat_s0_t1 = Empirical(self.data_treat_s0['Y(t1)'], True, samples)
            est_control_s1_t0 = Empirical(self.data_control_s1['Y(t0)'], True, samples)
            est_control_s1_t1 = Empirical(self.data_control_s1['Y(t1)'], True, samples)
        elif estimation_method == 'Gaussian' or estimation_method == 'Exponential':
            est_control_s0_t0 = MLE(self.data_control_s0['Y(t0)'], estimation_method, samples)
            est_control_s0_t1 = MLE(self.data_control_s0['Y(t1)'], estimation_method, samples)
            est_treat_s0_t0 = MLE(self.data_treat_s0['Y(t0)'], estimation_method, samples)
            est_treat_s0_t1 = MLE(self.data_treat_s0['Y(t1)'], estimation_method, samples)
            est_control_s1_t0 = MLE(self.data_control_s1['Y(t0)'], estimation_method, samples)
            est_control_s1_t1 = MLE(self.data_control_s1['Y(t1)'], estimation_method, samples)
        elif estimation_method == 'normal-lognormal':
            est_control_s0_t0 = MLE(self.data_control_s0['Y(t0)'], 'Gaussian', samples)
            est_control_s0_t1 = MLE(self.data_control_s0['Y(t1)'], 'Gaussian', samples)
            est_treat_s0_t0 = MLE(self.data_treat_s0['Y(t0)'], 'Gaussian', samples)
            est_treat_s0_t1 = MLE(self.data_treat_s0['Y(t1)'], 'lognormal', samples)
            est_control_s1_t0 = MLE(self.data_control_s1['Y(t0)'], 'Gaussian', samples)
            est_control_s1_t1 = MLE(self.data_control_s1['Y(t1)'], 'Gaussian', samples)
        else:
            raise Exception("estimator type not supported!")
        est_treat_s1_t1 = Empirical(self.data_treat_s1['Y(t1)'], False, samples)
        if samples is None:
            samples = len(self.data_treat_s1['Y(t0)'])
        counterfactual_estimate = 0
        for y in self.data_treat_s1['Y(t0)'][:samples]:
            u1 = est_control_s1_t0.get_cdf(y)
            y2 = est_control_s1_t1.get_inv_cdf(u1)
            u3 = est_control_s0_t1.get_cdf(y2)
            y4 = est_control_s0_t0.get_inv_cdf(u3)
            u5 = est_treat_s0_t0.get_cdf(y4)
            y6 = est_treat_s0_t1.get_inv_cdf(u5)
            counterfactual_estimate += y6
        counterfactual_estimate /= samples
        return est_treat_s1_t1.get_mean() - counterfactual_estimate
