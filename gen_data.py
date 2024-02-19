import numpy as np
import pandas as pd

# rate_lambda = .5


def h_lin(u, s, d, t):
    return 2*u + (s+1)*t/4 + (d-.5)*t/2


# data_000 = np.random.exponential(scale=1/rate_lambda, size=num_samples)
def gen_data(num_samples=100, mean_vec=None, linear=True, exponential=False):
    # Choose the production function:
    h = h_lin
    if mean_vec is None:
        mean_vec = [0, .25, -.25, .5, 2.75]
    lambda_vec = [1, 2, 3, 1, 1/3.75]
    std_vec = [1, 1, 1, 1]
    if not linear:
        std_vec[-1] = 1.25
        mean_vec[3] = -.5
        mean_vec[-1] = 10
    # generate samples from U in group s, d and time t (U_sdt)
    # NOTE: U_sd0 and U_sd1 must have the same distribution
    # Also generate outcomes based on the production function
    Us = dict()
    Ys = dict()
    for s in range(2):
        for d in range(2):
            for t in range(2):
                if not exponential:
                    Us[f'U_{s}{d}{t}'] = np.random.normal(mean_vec[2*s+d], std_vec[2*s+d], num_samples)
                else:
                    Us[f'U_{s}{d}{t}'] = np.random.exponential(scale=1./lambda_vec[2*s+d], size=num_samples)
                Ys[f'Y_{s}{d}{t}'] = h(Us[f'U_{s}{d}{t}'], s, d, t)
    if not exponential:
        Y_treated = np.random.normal(mean_vec[-1], 1, num_samples)  # the outcome of the treated group is not Y^0, but Y^1
    else:
        Y_treated = np.random.exponential(scale=1./lambda_vec[-1], size=num_samples)
    if not linear:  # add some non-linearity
        Ys['Y_011'] = np.exp(Ys['Y_011'])/10
        Ys['Y_111'] = np.exp(Ys['Y_111'])/10
    if linear:
        if not exponential:
            tau = mean_vec[-1] - h(mean_vec[3], 1, 1, 1)  # true ATT
        else:
            tau = 1./lambda_vec[-1] - h(1./lambda_vec[3], 1, 1, 1)
    else:
        tau = mean_vec[-1] - np.exp(h(mean_vec[3], 1, 1, 1) + (2*std_vec[-1])**2/2)/10  # mean of the log linear distribution
    cont_data_s0 = pd.DataFrame({'Y(t0)': Ys['Y_000'], 'Y(t1)': Ys['Y_001']})
    treat_data_s0 = pd.DataFrame({'Y(t0)': Ys['Y_010'], 'Y(t1)': Ys['Y_011']})
    cont_data_s1 = pd.DataFrame({'Y(t0)': Ys['Y_100'], 'Y(t1)': Ys['Y_101']})
    treat_data_s1 = pd.DataFrame({'Y(t0)': Ys['Y_110'], 'Y(t1)': Y_treated})

    return treat_data_s1, cont_data_s1, treat_data_s0, cont_data_s0, tau
