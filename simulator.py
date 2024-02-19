from estimators import DiD, CiC
import numpy as np


class RunEstimators:
    def __init__(self, datasets, taus):
        self.datasets = datasets
        self.runs = len(datasets)
        self.taus = taus

    def run(self, estimator, samples, estimation_method='Empirical'):
        abs_biases = np.zeros(self.runs)  # absolute relative bias w.r.t the true ATT
        for j in range(self.runs):
            print(f'--------{j}/{self.runs}')
            if isinstance(estimator, DiD) or isinstance(estimator, CiC):
                estimator.set_data(*self.datasets[j][:2])
            else:
                estimator.set_data(*self.datasets[j])
            estimate = estimator.estimate(samples=samples, estimation_method=estimation_method)
            abs_biases[j] = np.abs(estimate-self.taus[j]) / self.taus[j]
        mean_bias = np.mean(abs_biases)
        std_bias = np.std(abs_biases, ddof=1)  # ddof=1 for sample standard deviation
        confidence_interval = (mean_bias - 1.96 * std_bias / np.sqrt(self.runs),
                               mean_bias + 1.96 * std_bias / np.sqrt(self.runs))
        return mean_bias, *confidence_interval
