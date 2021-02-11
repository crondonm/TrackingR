import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from datetime import datetime
import numpy as np
import pandas as pd
import pystan
import pickle

from python.tools import (
    clean_folder,
    estimate_R
)

################
## Parameters ##
################

output_folder = './estimate_R/output/estimate_R_STAN'
input_folder  = './estimate_R/input/estimate_R_STAN'
min_T = 20
days_infectious_list = [5, 6, 7, 8, 9, 10] # Values of (1 / gamma) used in constructing
                                           # time series of infected individuals
num_iter = 10000 # MCMC iterations
num_warmup = 2000 # MCMC warmup period
num_chains = 3 # MCMC chains
sig_levels = [5, 35] # Significance levels for credible intervals
rng = np.random.RandomState(20200418) # Set seed

##############
## Clean up ##
##############

# clean_folder(output_folder)

for days_infectious in days_infectious_list:

  ###############
  ## Load data ##
  ###############
  
  df = pd.read_csv('{}/dataset_IJ.csv'.format(input_folder))
  df['Date'] = pd.to_datetime(df['Date'])

  # Impose minimum time-series observations
  df_temp = df.groupby('Country/Region').count()['gr_infected_{}'.format(days_infectious)].reset_index()
  df_temp.rename(columns = {'gr_infected_{}'.format(days_infectious): 'no_obs'},
                 inplace = True)
  df = pd.merge(df, df_temp, how = 'left')
  mask = df['no_obs'] >= min_T
  df = df.loc[mask, ]

  ################
  ## Estimate R ##
  ################

  df['R'] = np.nan
  df['n_eff_pct'] = np.nan
  df['Rhat_diff'] = np.nan
  for aa in sig_levels:
      df['ci_{}_u'.format(100 - aa)] = np.nan
      df['ci_{}_l'.format(100 - aa)] = np.nan

  # Load STAN models
  stm_missing = pickle.load(open('{}/model_missing.pkl'.format(input_folder), 'rb'))
  stm_no_missing = pickle.load(open('{}/model_no_missing.pkl'.format(input_folder), 'rb'))

  # Loop over countries
  for country in df['Country/Region'].unique():
    mask = df['Country/Region'] == country
    df_temp = df.loc[mask, ].copy()
    y = df_temp['gr_infected_{}'.format(days_infectious)].values
    res = estimate_R(y = y, 
                     gamma = 1 / days_infectious,
                     stm_missing = stm_missing, 
                     stm_no_missing = stm_no_missing,
                     num_iter = num_iter, 
                     num_chains = num_chains, 
                     num_warmup = num_warmup, 
                     rng = rng, 
                     sig_levels = sig_levels) 
    df.loc[mask, 'R'] = res['R']
    df.loc[mask, 'signal_to_noise'] = res['signal_to_noise']
    df.loc[mask, 'var_irregular']   = res['var_irregular']
    df.loc[mask, 'n_eff_pct'] = res['n_eff_pct']
    df.loc[mask, 'Rhat_diff'] = res['Rhat_diff']
    for aa in sig_levels:
      df.loc[mask, 'ci_{}_u'.format(100 - aa)] = res['ub_{}'.format(100 - aa)]
      df.loc[mask, 'ci_{}_l'.format(100 - aa)] = res['lb_{}'.format(100 - aa)]

  ####################
  ## Export results ##
  ####################

  df = df[['Country/Region', 'Date', 'R', 
           'ci_{}_u'.format(100 - sig_levels[0]), 'ci_{}_l'.format(100 - sig_levels[0]), 
           'ci_{}_u'.format(100 - sig_levels[1]), 'ci_{}_l'.format(100 - sig_levels[1]),
           'n_eff_pct', 'Rhat_diff', 
           'signal_to_noise', 'var_irregular']].copy()
  df['days_infectious'] = days_infectious
  df.to_csv('{}/estimated_R_IJ_{}.csv'.format(output_folder, days_infectious), index = False)

####################################
## Combine results into single df ##
####################################

res = []
for days_infectious in days_infectious_list:
    df_temp = pd.read_csv('{}/estimated_R_IJ_{}.csv'.format(output_folder, days_infectious))
    res.append(df_temp)

df = pd.concat(res)
df.reset_index(inplace = True)
del df['index']

# Check if any point estimates are negative
mask = df['R'] < 0
if mask.sum() > 0:
  print('Negative estimates found ({:} in total)'.format(mask.sum()))

# Save estimates
df['last_updated'] = datetime.today().strftime('%Y-%m-%d')
df.to_csv('{}/estimated_R_IJ.csv'.format(output_folder), index = False)  