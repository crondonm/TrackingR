import os
import sys
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

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python.tools import (
    clean_folder,
    simulate_AR1,
    estimate_R
)

################
## Parameters ##
################

# 6.5 minutes for M = 10

output_folder = './mismeasurement/output'
input_folder = './mismeasurement/input'
rng = np.random.RandomState(20200504) # For setting seed in Stan's estimation
np.random.seed(19801980) # For simulating data

# DGP and estimation parameters
days_infectious = 7.0
num_iter = 10000 # MCMC iterations
num_warmup = 2000 # MCMC warmup period
num_chains = 3 # MCMC chains
ramp_up_increase = 0.5 # Percentage increase in fraction detected
                       # in the testing ramp-up scenario
phi = 0.75 # Persistence of the AR(1) process
           # for the detection rate

# Simulation parameters
T = 50 # Sample size
M = 1000 # Number of Monte Carlo replications
gamma = 1 / days_infectious

################
## Simulation ##
################

clean_folder(output_folder)

# Read in data to calibrate simulation parameters
# from the empirical estimates
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
mask = (df['Country/Region'] == 'World') & (df['days_infectious'] == days_infectious)
df = df.loc[mask, ].reset_index()
df['obs'] = range(1, df.shape[0] + 1)
mask = df['obs'] <= T
df = df.loc[mask, ]

# Get "true" values of R and mu from
# the empirical estimates
R_true = df['R'].values
var_eps = df['var_irregular'][0]
var_mu = df['R'].var()
mu_true = gamma * (R_true - 1)

# Load STAN models
stm_missing = pickle.load(open('{}/model_missing.pkl'.format(input_folder), 'rb'))
stm_no_missing = pickle.load(open('{}/model_no_missing.pkl'.format(input_folder), 'rb'))

# Monte Carlo loop
for case in ['constant', 'ramp_up', 'stochastic']:
  df_res = pd.DataFrame()
  for mm in range(M):
    # Save true values of R
    df_temp = pd.DataFrame()
    df_temp['time'] = range(1, T + 1)
    df_temp['var_name'] = 'R_true'
    df_temp['value'] = R_true
    df_temp['MC_id'] = mm
    df_res = pd.concat([df_res, df_temp])
    # Simulate the growth in detection rate
    # according to simulation scenario
    if case == 'constant':
      # Case 1: Constant underdetection
      gr_alpha = np.zeros(len(mu_true))
      var_irregular = var_eps
    if case == 'ramp_up':
      # Case 2: Ramp-up in testing
      gr_alpha = np.zeros(len(mu_true))
      gr_alpha[0:14] = (1 + ramp_up_increase) ** (1 / 14) - 1
      var_irregular = var_eps
    if case == 'stochastic':
      # Case 3: Stochastic underdetection
      var_nu = 0.5 * (1 - phi ** 2) * var_eps / (1 + var_mu)
      gr_alpha = simulate_AR1(rho = phi, 
                              sigma = var_nu ** 0.5, 
                              T = len(mu_true))['x']
      var_irregular = 0.5 * var_eps

    # Simulate observed growth rate of infected
    # individuals
    eps = np.random.normal(loc = 0.0, 
                           scale = var_irregular ** 0.5, 
                           size = len(mu_true))
    gr_I_obs = gr_alpha * (1 + mu_true) + mu_true + eps

    # Estimate R
    res_est = estimate_R(y = gr_I_obs, 
                         gamma = 1 / days_infectious,
                         stm_missing = stm_missing, 
                         stm_no_missing = stm_no_missing,
                         num_iter = num_iter, 
                         num_chains = num_chains, 
                         num_warmup = num_warmup, 
                         rng = rng, 
                         sig_levels = [5]) 
    for var_name in ['R', 'ub_95', 'lb_95']:
      df_temp = pd.DataFrame()
      df_temp['time'] = range(1, T + 1)
      df_temp['var_name'] = var_name
      df_temp['value'] = res_est[var_name]
      df_temp['MC_id'] = mm
      df_res = pd.concat([df_res, df_temp])

  # Save results
  df_res.to_csv('{}/MC_results_{}.csv'.format(output_folder, case), index = False)

# Save results into wide format
for case in ['constant', 'ramp_up', 'stochastic']:
  df_res = pd.read_csv('{}/MC_results_{}.csv'.format(output_folder, case))
  df_res = df_res.set_index(['MC_id', 'time', 'var_name']).unstack('var_name')
  df_res.columns = df_res.columns.droplevel(0)
  df_res = df_res.reset_index(level = ['MC_id', 'time'])
  df_res.to_csv('{}/MC_results_wide_format_{}.csv'.format(output_folder, case), index = False)

################
## Get graphs ##
################

# Plot average estimates across different scenarios
fig, ax = plt.subplots(figsize = (5.0, 4.0))
for case, style, label in zip(['constant', 'ramp_up', 'stochastic'],
                              ['--r', '-.b', ':g'],
                              ['Constant Underdetection', 'Testing Ramp-Up', 'Stochastic Underdetection']):
    df_res = pd.read_csv('{}/MC_results_{}.csv'.format(output_folder, case))
    if case == 'constant':
      mask = (df_res['MC_id'] == 0) & (df_res['var_name'] == 'R_true') # The true is the same across all Monte Carlos
      plt.plot(df_res.loc[mask, 'time'], 
               df_res.loc[mask, 'value'], 
               '-k', linewidth = 2.0, label = 'True $\mathcal{R}$')
    # Plot average MC estimates
    mask = df_res['var_name'] == 'R'
    MC_mean = df_res.loc[mask, ['time', 'value']].groupby('time').mean().reset_index()
    plt.plot(MC_mean['time'], MC_mean['value'], style, label = label)
plt.legend(frameon = False, fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Effective Repr. Number ($\mathcal{R}$)', fontsize = 12)
fig.savefig("{}/MC_mean_estimates.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/MC_mean_estimates.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)

# Plot RMSE over time
fig, ax = plt.subplots(figsize = (5.0, 4.0))
for case, style, label in zip(['constant', 'ramp_up', 'stochastic'],
                              ['-r', '--b', ':g'],
                              ['Constant Underdetection', 'Testing Ramp-Up', 'Stochastic Underdetection']):
    df_res = pd.read_csv('{}/MC_results_wide_format_{}.csv'.format(output_folder, case))
    df_res['abs_error'] = np.abs(df_res['R_true'] - df_res['R'])    
    MC_mean = df_res.loc[:, ['time', 'abs_error']].groupby('time').mean().reset_index()
    plt.plot(MC_mean['time'], MC_mean['abs_error'], style, label = label)
plt.legend(frameon = False, fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Absolute Error', fontsize = 12)
fig.savefig("{}/MC_abs_error.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/MC_abs_error.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)
plt.show()

# Plot coverage frequency over time
fig, ax = plt.subplots(figsize = (5.0, 4.0))
for case, style, label in zip(['constant', 'ramp_up', 'stochastic'],
                              ['-r', '--b', ':g'],
                              ['Constant Underdetection', 'Testing Ramp-Up', 'Stochastic Underdetection']):
    df_res = pd.read_csv('{}/MC_results_wide_format_{}.csv'.format(output_folder, case))
    df_res['contained'] = ((df_res['lb_95'] <= df_res['R_true']) & (df_res['R_true'] <= df_res['ub_95'])) * 1.0
    MC_mean = df_res.loc[:, ['time', 'contained']].groupby('time').mean().reset_index()
    plt.plot(MC_mean['time'], MC_mean['contained'], style, label = label)
plt.legend(frameon = False, fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Coverage Frequency (95\% Cred. Int.)', fontsize = 12)
fig.savefig("{}/MC_coverage_prob.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/MC_coverage_prob.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)
plt.show()