import os
import shutil

import numpy as np
import pandas as pd
import scipy.integrate, scipy.stats, scipy.optimize, scipy.signal
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import pystan


def clean_folder(folder):
    """Create a new folder, or if the folder already exists,
    delete all containing files
    
    Args:
        folder (string): Path to folder
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_data_for_stan(y):
    """Convenience function for
    collecting data for STAN estimation
    
    Args:
        y (np vector): Data series for Bayesian filtering
    
    Returns:
        dict: Data for Stan estimation
    """
    assert y.ndim == 1, \
    "y must be a vector"

    assert len(y) > 0, \
    "y must have positive length"

    assert isinstance(y, np.ndarray), \
    "y must be a numpy array"

    N_obs  = len(pd.Series(y).dropna())
    N_mis  = np.sum(np.isnan(y))
    ii_obs = list(range(1, N_obs + N_mis + 1))
    ii_mis = []
    if N_mis > 0:
        for ii in np.argwhere(np.isnan(y)):
            ii_mis.append(ii[0] + 1)
            ii_obs.remove(ii[0] + 1)
    return {'N_obs': N_obs,
            'N_mis': N_mis,
            'ii_obs': ii_obs,
            'ii_mis': ii_mis,
            'y_obs': pd.Series(y).dropna()}


def estimate_R(y, gamma, stm_missing, stm_no_missing, num_iter, num_chains, num_warmup, rng, sig_levels, full_output = False):
    """Estimate R using Bayesian Kalman
    smoothing
    
    Args:
        y (np array): Data series for the growth rate of infected individuals
        gamma (double): Inverse of average infectiousness duration
        stm_missing (pickle): Stan model (for case with missing data)
        stm_no_missing (pickle): Stan model (for case without missing data)
        num_iter (int): Number of iterations
        num_chains (int): Number of MCMC chains
        num_warmup (int): Number of warmup periods
        rng (obj): Numpy random state
        sig_levels (list): List of significance levels for credible bounds
        full_output (bool, optional): If True, return full output from Stan
    
    Returns:
        TYPE: Description
    """
    assert y.ndim == 1, \
        "y must be a vector"

    assert len(y) > 0, \
        "y must have positive length"

    assert isinstance(y, np.ndarray), \
        "y must be a numpy array"

    assert isinstance(num_chains, int) and isinstance(num_iter, int) and isinstance(num_warmup, int), \
        "num_chains, num_iter, and num_warmup must be integers"

    assert num_chains > 0 and num_iter > 0 and num_warmup > 0, \
        "num_chains, num_iter, and num_warmup must be positive"

    assert len(sig_levels) >= 1 and all(isinstance(x, int) for x in sig_levels), \
        "sig_levels must be a list with only integers"

    # Get data in Stan format
    s_data = get_data_for_stan(y)

    # Estimate model
    if np.sum(np.isnan(y)) > 0:
        fit = stm_missing.sampling(data = s_data, 
           iter = num_iter, 
           chains = num_chains, 
           warmup = num_warmup, 
           verbose = False,
           seed = rng)
    else:
        fit = stm_no_missing.sampling(data = s_data, 
           iter = num_iter, 
           chains = num_chains, 
           warmup = num_warmup, 
           verbose = False,
           seed = rng)
    fit_res = fit.extract(permuted = True) 

    # Collect results
    res = {}
    res['R']  = 1 + 1 / gamma * fit_res['mu'].mean(axis = 0)
    for aa in sig_levels:
        ub = 1 + 1 / gamma * np.percentile(fit_res['mu'], axis = 0, q = 100 - aa / 2.0)
        lb = np.maximum(1 + 1 / gamma * np.percentile(fit_res['mu'], axis = 0, q = aa / 2.0), 0.0)
        res['ub_{}'.format(100 - aa)] = ub
        res['lb_{}'.format(100 - aa)] = lb
    res['signal_to_noise'] = fit_res['signal_to_noise'].mean()
    res['var_irregular'] = (1 / fit_res['precision_irregular']).mean()

    # Extract convergence statistics
    fit_summary = fit.summary()
    df_conv_stats = pd.DataFrame(fit_summary['summary'])
    df_conv_stats.columns = fit_summary['summary_colnames']
    df_conv_stats['var_name'] = fit_summary['summary_rownames']
    mask = df_conv_stats['var_name'].apply(lambda x: 'mu' in x)
    df_conv_stats = df_conv_stats.loc[mask, ]
    res['n_eff_pct'] = df_conv_stats['n_eff'].min() / float(num_chains * (num_iter - num_warmup))
    res['Rhat_diff'] = (df_conv_stats['Rhat'] - 1).abs().max()

    # If requested, extract full Stan fit
    if full_output:
        res['stan_fit'] = fit
    
    return res


def mean_se(x, robust = True):
    """Aggregation function for
    pandas to calculate standard errors
    for the mean
    
    Args:
        x (series): pandas Series
        robust (bool, optional): if True, calculate 
            heteroskedasticity-robust standard errors
    
    Returns:
        float: standard error
    """
    x = pd.DataFrame(x)
    x.columns = ['x']
    if robust:
        mod = smf.ols('x ~ 1', data = x).fit(cov_type = 'HC2')
    else:
        mod = smf.ols('x ~ 1', data = x).fit()
    return mod.bse['Intercept']


def simulate_AR1(rho, sigma, T, shocks = None):
    """Simulate a time series for 
    an AR(1) process with
    
    x_{t + 1} = rho x_t + eps_{t+1}
    
    where
    
    eps_{t + 1} ~ N(0, sigma ^ 2).
    
    Initial condition is 
    
    x_0 ~ N(0, sigma ^ 2 / (1 - rho ^ 2))

    Persistence parameter must lie in (-1, 1)
    for an AR(1) to be simulated.
    
    Args:
        rho (float): AR(1) persistence parameter
        sigma (float): Standard deviation of shocks
        T (int): Length of simulated time series
        shocks (array, optional): If provided,
            use the time series in shocks for the disturbances (eps)
    
    Returns:
        dict: Dictionary, contains:
            shocks (float): Simulated shocks (eps)
            x (float): Simulated time series
    """
    assert rho > - 1 and rho < 1, \
        'Persistence parameter should be in (-1, 1).'

    if shocks is None:
        shocks = np.random.randn(1, T).flatten() * sigma
        shocks[0] = np.random.randn(1, 1) * sigma / np.sqrt(1 - rho ** 2)
    return {'shocks': shocks,
            'x': scipy.signal.lfilter([1] ,[1, -rho], shocks)}