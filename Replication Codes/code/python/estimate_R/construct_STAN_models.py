import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
import pandas as pd
import pystan
import pickle

from python.tools import (
    clean_folder
)

################
## Parameters ##
################

output_folder = './estimate_R/output/construct_STAN_models'
input_folder = './estimate_R/input/construct_STAN_models'
min_T = 20
inflation_factor = 1.0 # Factor by which variances are inflated for priors
mean_mu0    = 0.25 # Prior for initial state: mean
std_dev_mu0 = 0.15 # Prior for initial state: standard devitaion

#########################################
## Calibrate priors using KF estimates ##
#########################################

clean_folder(output_folder)

df = pd.read_csv('{}/optim_res.csv'.format(input_folder))

# The local-level model estimated by frequentist Kalman filtering
# does not include a seasonal component. For calibration
# of the priors for full estimation that includes a seasonal
# component, we allocate fraction of the variance of the irregular 
# component (estimated by frequentist Kalman filtering) to the "true" irregular
# component, and remainder to the seasonal component
df['precision_irregular'] = 0.50 * (1 / df['sigma2_irregular'])
df['precision_seasonal']  = 0.50 * (1 / df['sigma2_irregular'])

# Remove the World aggregate when calibrating
mask = df['Country/Region'] == 'World'
df = df.loc[~mask, ]

# Calibration step
priors = []
for var_name in ['precision_irregular', 'precision_seasonal', 'signal_to_noise']:
  # Inflate variance of variables keeping mean constant
  df[var_name] = ((inflation_factor ** 0.5) * df[var_name] 
                  - (inflation_factor ** 0.5 - 1) * df[var_name].mean())
  # Calculate implied parameters for
  # gamma distribution
  alpha = df[var_name].mean() ** 2 / df[var_name].var()
  beta = df[var_name].mean() / df[var_name].var()
  priors.append({'var_name': var_name,
                 'alpha': alpha,
                 'beta': beta})
priors = pd.DataFrame(priors)

##################################################
## STAN code for the model: WITH MISSING values ##
##################################################

s_code_missing = """
  data {{
    int<lower=0> N_obs;
    int<lower=0> N_mis;
    int<lower=1, upper=N_obs + N_mis> ii_obs[N_obs];
    int<lower=1, upper=N_obs + N_mis> ii_mis[N_mis];
    real y_obs[N_obs];
  }}
  
  transformed data {{
    int<lower=0> N = N_obs + N_mis;
  }}
  
  parameters {{
    real y_mis[N_mis]; // To deal with missing values;
    real mu_zero;      // see https://mc-stan.org/docs/2_18/stan-users-guide/sliced-missing-data.html
    real mu[N];
    real<lower=0> precision_irregular;  // Precision (inverse of variance) of irregular component
    real<lower=0> signal_to_noise;      // Signal-to-noise ratio
    real s[N];                          // Intraweek seasonality component
    real<lower=0> precision_seasonal;   // Precision (inverse of variance) of seasonal component
  }}
  
  transformed parameters {{
    real y[N]; // To deal with missing values
    real<lower=0> precision_level;      // Precision (inverse of variance) of level component
    precision_level = precision_irregular / signal_to_noise;
    y[ii_obs] = y_obs;
    y[ii_mis] = y_mis;
  }}
  
  model {{
    // Priors
    precision_irregular ~ gamma({alpha_irregular}, {beta_irregular});
    signal_to_noise     ~ gamma({alpha_signal_to_noise}, {beta_signal_to_noise});
    mu_zero             ~ normal({mean_mu0}, {std_dev_mu0});
    precision_seasonal  ~ gamma({alpha_seasonal}, {beta_seasonal});
  
    // initial state
    mu[1] ~ normal(mu_zero, precision_level ^ (-0.5)); 

    // intraweek seasonality component
    for(i in 7:N) {{
      s[i] ~ normal(-s[i-1]-s[i-2]-s[i-3]-s[i-4]-s[i-5]-s[i-6], precision_seasonal ^ (-0.5));
      // Ensures that E(s[i] + s_[i-1] + ... s[i- 6]) = 0
      // See https://tharte.github.io/mbt/mbt.html#sec-3-9
    }} 

    // state equation
    for(i in 2:N) {{
      mu[i] ~ normal(mu[i-1], precision_level ^ (-0.5));
    }}

    // observation equation
    for(i in 1:N) {{
      y[i] ~ normal(mu[i] + s[i], precision_irregular ^ (-0.5));
    }}
  }}
"""  

s_code_missing = s_code_missing.format(mean_mu0 = mean_mu0,
                       std_dev_mu0 = std_dev_mu0,
                       alpha_irregular = priors.loc[priors['var_name'] == 'precision_irregular', 'alpha'].values[0],
                       beta_irregular = priors.loc[priors['var_name'] == 'precision_irregular', 'beta'].values[0],
                       alpha_seasonal = priors.loc[priors['var_name'] == 'precision_seasonal', 'alpha'].values[0],
                       beta_seasonal = priors.loc[priors['var_name'] == 'precision_seasonal', 'beta'].values[0],
                       alpha_signal_to_noise = priors.loc[priors['var_name'] == 'signal_to_noise', 'alpha'].values[0],
                       beta_signal_to_noise = priors.loc[priors['var_name'] == 'signal_to_noise', 'beta'].values[0])
with open('{}/stan_code_missing.txt'.format(output_folder), 'w+') as text_file:
            text_file.write(s_code_missing)

stm_missing = pystan.StanModel(model_code = s_code_missing)
with open('{}/model_missing.pkl'.format(output_folder), 'wb') as f:
    pickle.dump(stm_missing, f)

#####################################################
## STAN code for the model: WITHOUT MISSING values ##
#####################################################

s_code_no_missing = """
  data {{
    int<lower=0> N_obs;
    real y_obs[N_obs];
  }}
  
  transformed data {{
    int<lower=0> N = N_obs;
  }}
  
  parameters {{
    real mu_zero;
    real mu[N];
    real<lower=0> precision_irregular;  // Precision (inverse of variance) of irregular component
    real<lower=0> signal_to_noise;      // Signal-to-noise ratio
    real s[N];                          // Intraweek seasonality component
    real<lower=0> precision_seasonal;   // Precision (inverse of variance) of seasonal component
  }}
  
  transformed parameters {{
    real y[N]; 
    real<lower=0> precision_level;      // Precision (inverse of variance) of level component
    precision_level = precision_irregular / signal_to_noise;
    y = y_obs;
  }}
  
  model {{
    // Priors
    precision_irregular ~ gamma({alpha_irregular}, {beta_irregular});
    signal_to_noise     ~ gamma({alpha_signal_to_noise}, {beta_signal_to_noise});
    mu_zero             ~ normal({mean_mu0}, {std_dev_mu0});
    precision_seasonal  ~ gamma({alpha_seasonal}, {beta_seasonal});
  
    // initial state
    mu[1] ~ normal(mu_zero, precision_level ^ (-0.5));

    // intraweek seasonality component
    for(i in 7:N) {{
      s[i] ~ normal(-s[i-1]-s[i-2]-s[i-3]-s[i-4]-s[i-5]-s[i-6], precision_seasonal ^ (-0.5));
      // Ensures that E(s[i] + s_[i-1] + ... s[i- 6]) = 0
      // See https://tharte.github.io/mbt/mbt.html#sec-3-9
    }}  

    // state equation
    for(i in 2:N) {{
      mu[i] ~ normal(mu[i-1], precision_level ^ (-0.5));
    }}

    // observation equation
    for(i in 1:N) {{
      y[i] ~ normal(mu[i] + s[i], precision_irregular ^ (-0.5));
    }}
  }}
"""  

s_code_no_missing = s_code_no_missing.format(mean_mu0 = mean_mu0,
                       std_dev_mu0 = std_dev_mu0,
                       alpha_irregular = priors.loc[priors['var_name'] == 'precision_irregular', 'alpha'].values[0],
                       beta_irregular = priors.loc[priors['var_name'] == 'precision_irregular', 'beta'].values[0],
                       alpha_seasonal = priors.loc[priors['var_name'] == 'precision_seasonal', 'alpha'].values[0],
                       beta_seasonal = priors.loc[priors['var_name'] == 'precision_seasonal', 'beta'].values[0],
                       alpha_signal_to_noise = priors.loc[priors['var_name'] == 'signal_to_noise', 'alpha'].values[0],
                       beta_signal_to_noise = priors.loc[priors['var_name'] == 'signal_to_noise', 'beta'].values[0])
with open('{}/stan_code_no_missing.txt'.format(output_folder), 'w+') as text_file:
            text_file.write(s_code_no_missing)

stm_no_missing = pystan.StanModel(model_code = s_code_no_missing)
with open('{}/model_no_missing.pkl'.format(output_folder), 'wb') as f:
    pickle.dump(stm_no_missing, f)
