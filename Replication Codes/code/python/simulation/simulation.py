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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import warnings

from python.tools import (
    clean_folder
)

################
## Parameters ##
################

output_folder = './simulation/output/'
input_folder = './simulation/input/'
np.random.seed(20145224)

# Epidemiological parameters
gamma   = 1 / 18.0
kappa   = 1 / 5.2
eps     = 2 / 3.0
R0_star = 2.6
beta    = R0_star * gamma * kappa / (gamma * eps + kappa)

# Initialization
S_init = 11 * 1e6
E_init = 0
I_init = 1
R_init = 0
N = S_init + E_init + I_init + R_init # Implied population size

# Simulation parameters
T = 300
sd_noise = 0.10
M = 10000
gamma_estimation = (gamma ** (-1) + kappa ** (-1)) ** (-1)

################
## Simulation ##
################

clean_folder(output_folder)

# Check that the implied R0_star is correct
R0_implied = beta / gamma + eps * beta / kappa
assert (R0_implied - R0_star) <= 1e-12, \
 'The model is not calibrated correctly'

 # Initialize simulation
S = np.zeros(T) * np.nan
E = np.zeros(T) * np.nan
I = np.zeros(T) * np.nan
R = np.zeros(T) * np.nan
total_cases = np.zeros(T) * np.nan

S[0] = S_init
E[0] = E_init
I[0] = I_init
R[0] = R_init
total_cases[0] = I_init

# Simulation loop
for tt in range(1, T):
    S[tt] = (S[tt - 1] - beta * I[tt - 1] * S[tt - 1] / N 
                       - beta * eps * E[tt - 1] * S[tt - 1] / N)
    E[tt] = (E[tt - 1] + beta * I[tt - 1] * S[tt - 1] / N 
                       + beta * eps * E[tt - 1] * S[tt - 1] / N 
                       - kappa * E[tt - 1])
    I[tt] = (I[tt - 1] + kappa * E[tt - 1]
                       - gamma * I[tt - 1])
    R[tt] = R[tt - 1]  + gamma * I[tt - 1]
    total_cases[tt] = total_cases[tt - 1] + kappa * E[tt - 1]

# Construct dataframe
df = pd.DataFrame(S)
df.rename(columns = {0: 'S'},
          inplace = True)
df['E'] = E
df['I'] = I
df['R'] = R
df['total_cases'] = total_cases
df['time'] = range(1, df.shape[0] + 1)
df['eff_repr_number'] = R0_star * df['S'].shift() / N

# Look at data after 100 cases has been reached
mask = df['total_cases'] >= 100
df = df.loc[mask, ]
df.reset_index()

# Calculate growth in cases
df['gr_I'] = df['I'] / df.shift(1)['I'] - 1
df.to_csv('{}/SEIR_data.csv'.format(output_folder), index = False)

print('Range of gr(I): {:.2f}'.format(df['gr_I'].max() - df['gr_I'].min()))

# Main loop: Monte Carlo
res = pd.DataFrame()
res['eff_repr_number'] = df['eff_repr_number'].values
with warnings.catch_warnings():
  # Ignore warnings from statsmodels
  warnings.filterwarnings("ignore", message = "Maximum Likelihood optimization failed to converge. Check mle_retvals")
  for mm in range(M):
    df['gr_I_observed'] = df['gr_I'] + np.random.normal(scale = sd_noise, size = df.shape[0])
    mod_ll = sm.tsa.UnobservedComponents(df['gr_I_observed'].iloc[1:].values, 'local level')
    res_ll = mod_ll.fit(disp = False)
    estimated_R_star = 1 + 1 / (gamma_estimation) * res_ll.smoothed_state[0]
    estimated_R_star = np.insert(estimated_R_star, 0, np.nan)
    res['estimated_R_star_{}'.format(mm)] = estimated_R_star
res.to_csv('{}/MC_results.csv'.format(output_folder), index = False)  

# Get graph
res_MC = res.iloc[:, 1:].apply(np.mean, axis = 1)
fig, ax = plt.subplots(figsize = (5.0, 4.0))
plt.plot(res['eff_repr_number'], '-k', linewidth = 2.0, label = 'True $\\mathcal{R}$')
plt.plot(res_MC, '--b', linewidth = 1.5, 
         label = '$\\hat{{\\mathcal{{R}}}}, \gamma_{{est.}} = {:.3f}$'.format(gamma_estimation))

gamma_alt = 1 / 10.0
plt.plot(1 + (gamma_estimation / gamma_alt) * (res_MC - 1),  '-.r',
         linewidth = 1.5,
         label = '$\\hat{{\\mathcal{{R}}}}, \gamma_{{est.}} = {:.3f}$'.format(gamma_alt))
plt.xlabel('Days Since 100 Total Cases', fontsize = 12)
plt.ylabel('Effective Repr. Number', fontsize = 12)
plt.legend(frameon = False)
fig.savefig("{}/MC_SEIR.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/MC_SEIR.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)
