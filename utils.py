import numpy as np
import pandas as pd


# Likelihood function for the normal distribution
def log_likelihood_normal(params, data, log=False):
    """
    :param params: parameters to estimate
    :param data: given data
    :param log: if True, the density is assumed to be log-normal rather than normal
    :return:
    """
    if log:
        data_loc = np.log(10*data)
    else:
        data_loc = data
    mu, sigma = params
    n = len(data_loc)
    log_likelihood = -n / 2 * np.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * np.sum((data_loc - mu)**2)
    return -log_likelihood  # Negative because we will minimize the negative log-likelihood


# Likelihood function for the exponential distribution
def log_likelihood_exponential(params, data):
    rate_lambda = params[0]
    n = len(data)
    log_likelihood = -n * np.log(rate_lambda) + rate_lambda * np.sum(data)
    return -log_likelihood  # Negative because we will minimize the negative log-likelihood


def prep_nsch():
    # suppress warnings
    pd.options.mode.chained_assignment = None

    # Read data
    data2016 = pd.read_stata('nsch_2016_topical.dta')
    data2017 = pd.read_stata('nsch_2017_topical.dta')

    # Keep only entries from Louisiana, Misissipi, and Texas:
    data2016 = data2016[data2016['fipsst'].isin([22, 48, 28])]
    data2017 = data2017[data2017['fipsst'].isin([22, 48, 28])]

    # Replace the FIPS codes with state names
    state_map = {22: 'Louisiana', 48: 'Texas', 28: 'Mississippi'}
    data2016['fipsst'] = data2016['fipsst'].map(state_map)
    data2017['fipsst'] = data2017['fipsst'].map(state_map)

    # partition groups into d=0 and d=1
    data2016_d0 = data2016[data2016['fpl'] >= 140]
    data2017_d0 = data2017[data2017['fpl_i2'] >= 140]
    data2016_d1 = data2016[data2016['fpl'] <= 100]
    data2017_d1 = data2017[data2017['fpl_i2'] <= 100]

    # partition data based on states:
    # s0: texas, mississippi (no expansion) s1: louisiana (expansion adopted)
    datas0d0t0 = data2016_d0[data2016_d0['fipsst'].isin(['Mississippi', 'Texas'])]['k4q20r']
    datas0d0t1 = data2017_d0[data2017_d0['fipsst'].isin(['Mississippi', 'Texas'])]['k4q20r']
    datas0d1t0 = data2016_d1[data2016_d1['fipsst'].isin(['Mississippi', 'Texas'])]['k4q20r']
    datas0d1t1 = data2017_d1[data2017_d1['fipsst'].isin(['Mississippi', 'Texas'])]['k4q20r']
    datas1d0t0 = data2016_d0[data2016_d0['fipsst'].isin(['Louisiana'])]['k4q20r']
    datas1d0t1 = data2017_d0[data2017_d0['fipsst'].isin(['Louisiana'])]['k4q20r']
    datas1d1t0 = data2016_d1[data2016_d1['fipsst'].isin(['Louisiana'])]['k4q20r']
    datas1d1t1 = data2017_d1[data2017_d1['fipsst'].isin(['Louisiana'])]['k4q20r']

    # Drop NaN entries and adjust entries based on true values:
    visit_map = {1: 0, 2: 2, 3: 2}
    datas0d0t0.dropna(inplace=True)
    datas0d0t0 = datas0d0t0.map(visit_map)
    datas0d0t1.dropna(inplace=True)
    datas0d0t1 = datas0d0t1.map(visit_map)
    datas0d1t0.dropna(inplace=True)
    datas0d1t0 = datas0d1t0.map(visit_map)
    datas0d1t1.dropna(inplace=True)
    datas0d1t1 = datas0d1t1.map(visit_map)
    datas1d0t0.dropna(inplace=True)
    datas1d0t0 = datas1d0t0.map(visit_map)
    datas1d0t1.dropna(inplace=True)
    datas1d0t1 = datas1d0t1.map(visit_map)
    datas1d1t0.dropna(inplace=True)
    datas1d1t0 = datas1d1t0.map(visit_map)
    datas1d1t1.dropna(inplace=True)
    datas1d1t1 = datas1d1t1.map(visit_map)

    # prepare the input:
    np.random.seed(132)
    cont_data_s0 = {'Y(t0)': np.array(datas0d0t0) + np.random.normal(loc=0, scale=0.2, size=len(datas0d0t0)),
                    'Y(t1)': np.array(datas0d0t1) + np.random.normal(loc=0, scale=0.2, size=len(datas0d0t1))}
    treat_data_s0 = {'Y(t0)': np.array(datas0d1t0) + np.random.normal(loc=0, scale=0.2, size=len(datas0d1t0)),
                     'Y(t1)': np.array(datas0d1t1) + np.random.normal(loc=0, scale=0.2, size=len(datas0d1t1))}
    cont_data_s1 = {'Y(t0)': np.array(datas1d0t0) + np.random.normal(loc=0, scale=0.2, size=len(datas1d0t0)),
                    'Y(t1)': np.array(datas1d0t1) + np.random.normal(loc=0, scale=0.2, size=len(datas1d0t1))}
    treat_data_s1 = {'Y(t0)': np.array(datas1d1t0) + np.random.normal(loc=0, scale=0.2, size=len(datas1d1t0)),
                     'Y(t1)': np.array(datas1d1t1) + np.random.normal(loc=0, scale=0.2, size=len(datas1d1t1))}

    return [treat_data_s1, cont_data_s1, treat_data_s0, cont_data_s0]