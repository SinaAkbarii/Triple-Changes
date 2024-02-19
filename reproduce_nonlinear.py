from estimators import DiD, DDD, CiC, CCC
from gen_data import gen_data
import matplotlib
import matplotlib.pyplot as plt
from plots import init_plot, plot_data
import numpy as np
from simulator import RunEstimators


# Fix the number of samples to use for estimations
dataset_size = 8100      # max number of samples
min_samples = 100        # min number of samples
num_datasets = 20       # number of runs per data point on the plots
num_points = 17         # number of data points on the plots
sample_choices = np.linspace(min_samples, dataset_size, num_points)

# Generate datasets
np.random.seed(132)
datasets = []
taus = []  # true value of the ATT
for i in range(num_datasets):
    data = [[]] * 4
    data[0], data[1], data[2], data[3], tau = gen_data(dataset_size, linear=False)
    datasets.append(data)
    taus.append(tau)
simulator = RunEstimators(datasets, taus)

# Instantiating DiD, CiC, DDD, and CCC with the dataset during instantiation
did = DiD()
cic = CiC()
ddd = DDD()
ccc = CCC()

did_estimates = np.zeros((len(sample_choices), 3))
ddd_estimates = np.zeros((len(sample_choices), 3))
cic_estimates_emp = np.zeros((len(sample_choices), 3))
ccc_estimates_emp = np.zeros((len(sample_choices), 3))
cic_estimates_gaus = np.zeros((len(sample_choices), 3))
ccc_estimates_gaus = np.zeros((len(sample_choices), 3))

for i in range(len(sample_choices)):
    print(f'Running simulations with {num_datasets} runs per data point.')
    num_sample = int(sample_choices[i])
    print(f'Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample}')

    print(f'--DiD estimator running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    did_estimates[i, :] = simulator.run(estimator=did, samples=num_sample)
    print(f'--DDD estimator running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    ddd_estimates[i, :] = simulator.run(estimator=ddd, samples=num_sample)

    print(f'--CiC estimator (Empirical cdf) running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    cic_estimates_emp[i, :] = simulator.run(estimator=cic, samples=num_sample)
    print(f'--CCC estimator (Empirical cdf) running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    ccc_estimates_emp[i, :] = simulator.run(estimator=ccc, samples=num_sample)

    print(f'--CiC estimator (MLE cdf) running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    cic_estimates_gaus[i, :] = simulator.run(estimator=cic, samples=num_sample, estimation_method='Gaussian')
    print(f'--CCC estimator (MLE cdf) running... (Running point {i + 1}/{len(sample_choices)}. # samples:{num_sample})')
    ccc_estimates_gaus[i, :] = simulator.run(estimator=ccc, samples=num_sample, estimation_method='normal-lognormal')

matplotlib.use('TkAgg')
plt.style.use('ggplot')
ax = init_plot(dataset_size)


estimate_all = [did_estimates, cic_estimates_emp, cic_estimates_gaus, ddd_estimates,
                ccc_estimates_emp, ccc_estimates_gaus]
colors = ['dodgerblue', 'forestgreen', 'purple', 'darkorange', 'red', 'darkcyan']
markers = ['o', '^', 'D', 's', 'v', 'P']
labels = ['DiD Estimates', 'CiC Estimates (Empirical)', 'CiC Estimates (MLE)',
          'DDD Estimates', 'CCC Estimates (Empirical)', 'CCC Estimates (MLE)']
# Plotting mean estimates
for i in range(len(estimate_all)):
    plot_data(ax, sample_choices, estimate_all[i][:, 0], labels[i], colors[i], markers[i])

# Adding confidence intervals
for i in range(len(estimate_all)):
    ax.fill_between(sample_choices, estimate_all[i][:, 1], estimate_all[i][:, 2], color=colors[i], alpha=0.2)

# Add legend
ax.legend()


plt.ylim([0, 1])
plt.xlim([0, 8100])
plt.savefig('nonlinear.pdf', bbox_inches='tight')
plt.show()
plt.xlim([0, 5600])
plt.ylim([0, .9])
np.save('nonlinear_results.npy', np.array(estimate_all))