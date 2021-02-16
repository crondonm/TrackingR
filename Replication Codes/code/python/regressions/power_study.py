import os
import sys
import copy
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
import matplotlib.patches as mpatches


from python.tools import (
    clean_folder
)

################
## Parameters ##
################

output_folder = './regressions/output/power_study/'
input_folder = './regressions/input/power_study/'
days_infectious = 7
M = 10000 # Number of Monte Carlo replications
T = 14 # Simulation period
R_0 = 2.0 # Initial R effective
R_1 = 1.5 # New lower R effective
np.random.seed(20200429)

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data on estimates of R
df = pd.read_csv('{}/regressions_dataset.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns = {'Country/Region': 'Country'},
          inplace = True)

# Select number of days infectious
mask = df['days_infectious'] == days_infectious
df = df.loc[mask, ]

# Calculate number of countries
num_countries = len(df['Country'].unique())

# Calibrate noise-to-signal ratio
# and variance of the irregular (noise) component
# by using the empirical estimates
q = df.groupby('Country').first()['signal_to_noise'].median() # Signal-to-noise ratio
var_eps = df.groupby('Country').first()['var_irregular'].median() # Variance of irregular component

# Calculate implied parameters
gamma = 1 / float(days_infectious)
mu_0 = gamma * (R_0 - 1)
mu_1 = gamma * (R_1 - 1)
var_eta = q * var_eps # Variance of the level component
omega = (q + np.sqrt(q ** 2 + 4 * q)) / (2 + q + np.sqrt(q ** 2 + 4 * q)) # Kalman gain
var_errors = (omega ** 2 * var_eps + (1 - omega) ** 2 * var_eta) / (1 - (1 - omega) ** 2) # Variance of nowcast errors

# Save parameter data
pars = {'$\gamma$': gamma,
        '$q$': q,
        '$\sigma^2_{\\varepsilon}$': var_eps,
        '$\omega$': omega}
pars = pd.DataFrame([pars])
f = lambda x: '{:.3f}'.format(x) 
latex = pars.to_latex(column_format = 'c' * 4,
                     formatters = [f, f, f, f],
                     escape = False,
                     index = False)
with open('{}/parameters_power_study.txt'.format(output_folder), 'w+') as text_file:
        text_file.write(latex)

############################
## Monte Carlo simulation ##
############################

res = []
for mm in range(M):
    res_temp = []
    # Simulate data for each country
    for nn in range(num_countries):
        # Initialize
        mu = np.zeros(1 + T) * np.nan
        y  = np.zeros(1 + T) * np.nan
        m  = np.zeros(1 + T) * np.nan

        # Simulate true growth rates
        mu[0] = mu_0
        mu[1] = mu_1
        for tt in range(2, T + 1):
            mu[tt] = mu[tt - 1] + np.random.normal(loc = 0.0, scale = var_eta ** 0.5)
        R_true = 1 + gamma ** (-1) * mu

        # Simulate observed growth rates
        y = mu + np.random.normal(loc = 0.0, scale = var_eps ** 0.5, size = len(mu))

        # Initialize forecasts
        m[0] = np.random.normal(loc = mu_0, scale = var_errors ** 0.5)

        # Simulate forecasts
        for tt in range(1, T + 1):
            m[tt] = omega * y[tt] + (1 - omega) * m[tt - 1]
        R_estim = 1 + gamma ** (-1) * m
        df_temp = pd.DataFrame()
        df_temp['R_true'] = R_true[1:]
        df_temp['R_estim'] = R_estim[1:]
        df_temp['time'] = range(len(R_true[1:]))
        df_temp['country_id'] = nn
        res_temp.append(df_temp)
    # Take averages across countries
    res_temp = pd.concat(res_temp)
    df_temp = res_temp.groupby('time').mean()[['R_true', 'R_estim']].reset_index()
    df_temp['MC_id'] = mm
    res.append(df_temp)
res = pd.concat(res)
res.to_csv('{}/MC_results.csv'.format(output_folder), index = False)

###############
## Get graph ##
###############

mean = res.groupby('time').mean()[['R_true', 'R_estim']].reset_index()
fig, ax = plt.subplots(figsize = (5.0, 4.0))
plt.plot(mean['time'], mean['R_estim'], 'r-o', linewidth = 2.0)

alpha = [0.05, 0.35]
colors = ['b', 'r']
for aa, cc in zip(alpha, colors):
    pct_lower = res.groupby('time').quantile(aa / 2.0)['R_estim'].reset_index()
    pct_upper = res.groupby('time').quantile(1 - aa / 2.0)['R_estim'].reset_index()
    plt.fill_between(mean['time'], 
                 pct_lower['R_estim'], 
                 pct_upper['R_estim'],
                 color = cc, alpha = 0.10)

conf_95 = mpatches.Patch(color = colors[0], alpha = 0.15,
                       label = '95$\%$ Simulations')
conf_65 = mpatches.Patch(color = colors[1], alpha = 0.15, 
                       label = '65$\%$ Simulations')
plt.legend(handles = [conf_65, conf_95],
         frameon = False,
         fontsize = 10,
         loc = 'lower left')

plt.axhline(R_0, linestyle = '--', color = 'k')
plt.axhline(R_1, linestyle = '--', color = 'k')
plt.text(8, 2.04, 'Initial $\mathcal{R}$', fontsize = 12)
plt.text(1.0, 1.42, 'New $\mathcal{R}$', fontsize = 12)
plt.xlabel('Days After Change in $\mathcal{R}$', fontsize = 12)
plt.ylabel('Effective Repr. Number', fontsize = 12)
fig.savefig("{}/power_study.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/power_study.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)