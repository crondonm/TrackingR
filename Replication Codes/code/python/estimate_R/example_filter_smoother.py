import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import matplotlib as mpl

mpl.use('pgf')
pgf_with_latex = {                     
    "pgf.texsystem": "xelatex",        
    "pgf.rcfonts": False,
    "text.usetex": True,                
    "font.family": "Times New Roman",
    "pgf.preamble": [
        r"\usepackage{fontspec}",    
        r"\setmainfont{Times New Roman}",        
        r"\usepackage{unicode-math}",
        r"\setmathfont{xits-math.otf}"
        ]
    }    
mpl.rcParams.update(pgf_with_latex)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pystan
import scipy

from python.tools import (
    clean_folder,
    estimate_R
)

################
## Parameters ##
################

output_folder = './estimate_R/output/example_filter_smoother'
input_folder = './estimate_R/input/example_filter_smoother'
days_infectious = 7
num_iter = 10000 # MCMC iterations
num_warmup = 2000 # MCMC warmup period
num_chains = 3 # MCMC chains
mean_mu0    = 0.35 # Prior for initial state
std_dev_mu0 = 0.50 
sig_levels = [5, 35] # Significance levels for credible intervals
rng = np.random.RandomState(20200504) # For setting seed in Stan's estimation
np.random.seed(19801980) # For reproducible Bayesian filtering

# Implied parameters
gamma = 1 / float(days_infectious)

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data for China
df = pd.read_csv('./estimate_R/input/estimate_R_STAN/dataset.csv')
mask = df['Country/Region'] == 'China'
df = df.loc[mask, ]
df.reset_index(inplace = True)
del df['index']
df['Date'] = pd.to_datetime(df['Date'])

###################
## Get estimates ##
###################

# Frequentist estimates
mod = sm.tsa.UnobservedComponents(df['gr_infected_{}'.format(days_infectious)].values, 'local level')
res_freq = mod.fit(disp = False)
df_freq = pd.DataFrame()
df_freq['Date'] = df['Date']
df_freq['R_filtered'] = 1 + gamma ** (-1) * res_freq.filtered_state[0]
df_freq['R_smoothed'] = 1 + gamma ** (-1) * res_freq.smoothed_state[0]
df_freq.to_csv('{}/frequentist_estimates.csv'.format(output_folder), index = False)

# Bayesian estimates

# Load STAN models
stm_missing = pickle.load(open('./estimate_R/input/estimate_R_STAN/model_missing.pkl', 'rb'))
stm_no_missing = pickle.load(open('./estimate_R/input/estimate_R_STAN/model_no_missing.pkl', 'rb'))
res_Bayes = estimate_R(y = df['gr_infected_{}'.format(days_infectious)].values, 
                       gamma = 1 / days_infectious,
                       stm_missing = stm_missing, 
                       stm_no_missing = stm_no_missing,
                       num_iter = num_iter, 
                       num_chains = num_chains, 
                       num_warmup = num_warmup, 
                       rng = rng, 
                       sig_levels = sig_levels,
                       full_output = True)

# Collect results
df_smoother_Bayesian = pd.DataFrame()
df_smoother_Bayesian['Date'] = df['Date']
df_smoother_Bayesian['R'] = res_Bayes['R']
df_smoother_Bayesian['lb_65'] = res_Bayes['lb_65']
df_smoother_Bayesian['ub_65'] = res_Bayes['ub_65']
df_smoother_Bayesian.to_csv('{}/bayesian_smoother.csv'.format(output_folder), index = False)

# Get Bayesian filtered states
fit_res = res_Bayes['stan_fit'].extract(permuted = True) 
dist_var_irregular = fit_res['precision_irregular'] ** (-1) # Draws from the estimated distribution from Stan
dist_signal_to_noise = fit_res['signal_to_noise'] # Draws from the estimated distribution from Stan

# Monte Carlo loop -- take random draws from the distributions
# of variances for the irregular and level components
# and run the Kalman filter
res_filter_Bayesian = []
for mm in range(num_iter):
    # Pick a random draw for the variances of the 
    # irregular and level components from the Bayesian estimates
    var_irregular = np.random.choice(dist_var_irregular, 1)[0]
    var_level = var_irregular * np.random.choice(dist_signal_to_noise, 1)[0]

    # Run Kalman filter using these variances
    mod = sm.tsa.UnobservedComponents(df['gr_infected_{}'.format(days_infectious)].values, 'local level')
    mod.initialize_known(np.array([mean_mu0]), np.array([[std_dev_mu0]]))
    res_temp = mod.smooth(params = np.array([var_irregular, var_level]))

    df_temp = pd.DataFrame()
    df_temp['R'] = 1 + (1 / gamma) * res_temp.filtered_state[0]
    df_temp['R_se'] = (1 / gamma) * res_temp.filtered_state_cov[0][0] ** 0.5

    alpha = [0.05, 0.35]
    names = ['95', '65']
    for aa, name in zip(alpha, names):
        t_crit = scipy.stats.norm.ppf(1 - aa / 2)
        df_temp['ub_{}'.format(name)] = df_temp['R'] + t_crit * df_temp['R_se']
        df_temp['lb_{}'.format(name)] = np.maximum(df_temp['R'] - t_crit * df_temp['R_se'], 0.0)
    df_temp['MC_id'] = mm
    df_temp['Date'] = df['Date']
    res_filter_Bayesian.append(df_temp)
df_filter_Bayesian = pd.concat(res_filter_Bayesian)
df_filter_Bayesian = df_filter_Bayesian.groupby('Date').mean()[['R', 'lb_65', 'ub_65']].reset_index()
df_filter_Bayesian.to_csv('{}/bayesian_filter.csv'.format(output_folder), index = False)

###############
## Get graph ##
###############

fig, ax = plt.subplots(figsize = (5.0, 4.0))
plt.plot(df_smoother_Bayesian['Date'], df_smoother_Bayesian['R'], '-r', linewidth = 1.5, label = 'Bayesian Smoother')
plt.plot(df_freq['Date'], df_freq['R_smoothed'], '--k', label = 'Classical Smoother')
plt.fill_between(df_smoother_Bayesian['Date'], 
                 df_smoother_Bayesian['lb_65'], 
                 df_smoother_Bayesian['ub_65'],
                 color = 'r', alpha = 0.15)
plt.plot(df_filter_Bayesian['Date'], df_filter_Bayesian['R'], '-b', 
         linewidth = 1.5, label = 'Bayesian Filter')
plt.fill_between(df_filter_Bayesian['Date'], 
                 df_filter_Bayesian['lb_65'], 
                 df_filter_Bayesian['ub_65'],
                 color = 'b', alpha = 0.15)
plt.plot(df_freq['Date'], df_freq['R_filtered'], ':k', label = 'Classical Filter')
plt.legend(frameon = False, fontsize = 12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
plt.axhline(1, linestyle = '--', color = 'k', alpha = 0.25)
plt.axhline(0, linestyle = '--', color = 'k', alpha = 0.25)
plt.ylabel('Effective Repr. Number ($\\mathcal{R}$)', fontsize = 12)
fig.savefig("{}/filter_smoother.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/filter_smoother.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)