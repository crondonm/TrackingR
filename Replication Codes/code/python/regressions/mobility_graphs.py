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
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
from python.stargazer.stargazer import Stargazer
import re
from sklearn.decomposition import PCA

from python.tools import (
    clean_folder,
    mean_se
)

################
## Parameters ##
################

output_folder = './regressions/output/mobility_graphs/'
input_folder = './regressions/input/mobility_graphs/'
T_before = 7 # Number of days before intervention in event-study graphs
T_after = 3 * 7 # Number of days after intervention in event-study graphs
min_countries = 5 # Minimum number of countries per time in regressions

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data on estimates of R
df = pd.read_csv('{}/regressions_dataset.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns = {'Country/Region': 'Country'},
          inplace = True)

###############################
## Mobility around lockdowns ##
###############################

# Select countries for which we have data on
# R T_before days before the intervention and
# T_after days after the intervention (to make
# sure the sample of countries is the same)

# Calculate number of days before intervention
df_temp = df.groupby('Country').min()['days_lockdown'].abs().reset_index()
df_temp.rename(columns = {'days_lockdown': 'T_before'},
               inplace = True)
df = pd.merge(df, df_temp, how = 'left')

# Calculate number of days after intervention
df_temp = df.groupby('Country').max()['days_lockdown'].reset_index()
df_temp.rename(columns = {'days_lockdown': 'T_after'},
               inplace = True)
df = pd.merge(df, df_temp, how = 'left')

# Mask for selecting a sample of countries
# that remains fixed during the event-study time
mask = ((df['T_before'] >= T_before) & 
        (df['T_after'] >= T_after) &
        (df['days_lockdown'] >= -T_before) & 
        (df['days_lockdown'] <= T_after))
df = df.loc[mask, ]

assert np.isnan(df['mobility_PC1']).sum() == 0, \
  "Missing values in the mobility measure."

# Calculate means and s.e.'s
df_temp = df.loc[mask, ].groupby('days_lockdown').mean()['mobility_PC1'].reset_index()
df_se = df.loc[mask, ].groupby('days_lockdown')['mobility_PC1'].agg(mean_se).reset_index()
# df_temp.to_csv('{}/mean_mobility.csv'.format(output_folder), index = False)
# df_se.to_csv('{}/se_mobiliy.csv'.format(output_folder), index = False)
df_se.rename(columns = {'mobility_PC1': 'se'},
             inplace = True)
df_temp = pd.merge(df_temp, df_se, on = 'days_lockdown')

# Get plots
fig, ax = plt.subplots(figsize = (5.0, 4.0))
plt.plot(df_temp['days_lockdown'], df_temp['mobility_PC1'], '-b', linewidth = 2.5, label = '$R$')

alpha = [0.05, 0.35]
colors = ['b', 'r']
names = ['95conf', '65conf']
for aa, cc, nn in zip(alpha, colors, names):
    t_crit = scipy.stats.norm.ppf(1 - aa / 2)
    df_temp['lb_{}'.format(nn)] = df_temp['mobility_PC1'] - t_crit * df_temp['se']
    df_temp['ub_{}'.format(nn)] = df_temp['mobility_PC1'] + t_crit * df_temp['se']
    plt.fill_between(df_temp['days_lockdown'], 
                     df_temp['lb_{}'.format(nn)], 
                     df_temp['ub_{}'.format(nn)],
                     color = cc, alpha = 0.15)

# Add legend for confidence bounds
conf_95 = mpatches.Patch(color = colors[0], alpha = 0.15,
                         label = '95$\%$ Confidence Interval')
conf_65 = mpatches.Patch(color = colors[1], alpha = 0.15, 
                         label = '65$\%$ Confidence Interval')
plt.legend(handles = [conf_65, conf_95],
           frameon = False,
           fontsize = 10)

# Save graph data
df_temp.to_csv('{}/mobility_event_study.csv'.format(output_folder), index = False)

# Annotate graph
plt.text(6.0, 2.65, 
         '$N = {}$ countries'.format(int(df.loc[mask, ].groupby('days_lockdown').count()['mobility_PC1'].values[0])),
         fontsize = 12)
plt.axvline(0, linestyle = '--', alpha = 0.25)
plt.ylabel('Mobility Index', fontsize = 12)
plt.xlabel('Days Since Intervention', fontsize = 12)
fig.savefig("{}/event_study_mobility.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/event_study_mobility.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)

###############################
## Correlate mobility with R ##
###############################

# Load data on estimates of R
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
df.rename(columns = {'Country/Region': 'Country'},
          inplace = True)

# Remove aggregates for the world
mask = df['Country'] == 'World'
df = df.loc[~mask, ]

# Look at benchmark estimates
mask = df['days_infectious'] == 7.0
df = df.loc[mask, ]

# Aggregate to weekly frequency
df = df.groupby('Country').resample('w').mean().reset_index()[['Country', 'Date', 'R']]

# Load mobility data and merge in
df_temp = pd.read_csv('{}/mobility.csv'.format(input_folder))
df_temp['Date'] = pd.to_datetime(df_temp['Date'])
df_temp.sort_values(by = ['Country', 'Date'], ascending = True,
                    inplace = True)
df_temp.index = df_temp['Date']

# Aggregate to weekly frequency
df_temp = df_temp.groupby('Country').resample('w').mean().reset_index()[['Country', 'Date', 'mobility_PC1']]
df_temp['mobility_PC1_LAG2'] = df_temp.groupby('Country').shift(2)['mobility_PC1']
df = pd.merge(df, df_temp, how = 'left', on = ['Date', 'Country'])

# Perform within transformation
for var_name in ['R', 'mobility_PC1_LAG2']:
    df_temp = df.groupby('Country').mean()[var_name].reset_index()
    df_temp.rename(columns = {var_name: '{}_country_mean'.format(var_name)},
                   inplace = True)
    df = pd.merge(df, df_temp, how = 'left')
    df['{}_within'.format(var_name)] = df[var_name] - df['{}_country_mean'.format(var_name)] + df[var_name].mean()

# Calculate number of observations and countries
num_obs = {}
num_obs['N_total'] = df[['R_within', 'mobility_PC1_LAG2_within']].dropna().shape[0]
# num_obs['N_countries'] = len(df['Country'].unique())
num_obs['N_countries'] = len(df[['Country', 'R_within', 'mobility_PC1_LAG2_within']].dropna()['Country'].unique())
num_obs = pd.Series(num_obs)
num_obs.to_csv('{}/scatter_num_obs.csv'.format(output_folder), index = True)

# Get scatter plot
fig, ax = plt.subplots(figsize = (5.0, 4.0))
sns.regplot(df['mobility_PC1_LAG2_within'], df['R_within'],
            line_kws = {'color': 'k', 'alpha': 0.9},
            scatter_kws = {'alpha': 0.5, 'color': 'k'})
plt.xlabel('Mobility Index Two Weeks Ago', fontsize = 12)
plt.ylabel('Effective Repr. Number This Week', fontsize = 12)
plt.text(-75, 3.5, 
         'Corr. = {:.2f}'.format(df[['R_within', 'mobility_PC1_LAG2_within']].corr().values[0,1]),
         fontsize = 12)
df[['mobility_PC1_LAG2_within', 'R_within']].to_csv('{}/scatter_data.csv'.format(output_folder), index = False)
fig.savefig("{}/mobility_scatter.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/mobility_scatter.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)