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
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf

from python.tools import (
    clean_folder
)

################
## Parameters ##
################

output_folder = './regressions/output/predictive/'
input_folder = './regressions/input/predictive/'
min_deaths = 50 # Minimum number of deaths to be included in the sample
days_infectious = 7.0

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data on estimates of R
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
mask = df['days_infectious'] == days_infectious

# Remove aggregates for the world
mask = df['Country/Region'] == 'World'
df = df.loc[~mask, ]

# Merge in data on cases / deaths
df_temp = pd.read_csv('{}/dataset.csv'.format(input_folder))
df_temp = df_temp[['Country/Region', 'Date', 'new_deaths', 'total_deaths']]
df = pd.merge(df, df_temp, how = 'left')
df.rename(columns = {'Country/Region': 'Country'},
          inplace = True)

# Convert to weekly frequency
df.index = pd.to_datetime(df['Date'])
df_new = df.groupby('Country').resample('w').sum().reset_index()[['Country', 'Date', 'new_deaths']]
df_total = df.groupby('Country').resample('w').last()['total_deaths'].reset_index()
df_R = df.groupby('Country').resample('w').mean().reset_index()[['Country', 'Date', 'R']]
df = df_new
df = pd.merge(df, df_total, how = 'left')
df = pd.merge(df, df_R, how = 'left')

# Mask on total number of deaths
mask = df['total_deaths'] >= min_deaths
df = df.loc[mask, ]

# Remove the outlier observation for China
# (including many previously uncounted deaths 
# in the week of 2020-04-13 -- 2020-04-19)
mask = (df['Country'] == 'China') & (df['Date'] == '2020-04-19')
df.loc[mask, 'new_deaths'] = np.nan

# Calculate growth rate in new deaths
df['gr_new_deaths'] = df['new_deaths'] / df.groupby('Country').shift(1)['new_deaths'] - 1
mask = np.isinf(df['gr_new_deaths'])
df.loc[mask, 'gr_new_deaths'] = np.nan
df['gr_new_deaths_F1'] =  df.groupby('Country').shift(1)['gr_new_deaths']
df['gr_new_deaths_F2'] =  df.groupby('Country').shift(2)['gr_new_deaths']
df['net_R'] = df['R'] - 1

################
## Get graphs ##
################

# Perform within transformation by each country
for var_name in ['net_R', 'gr_new_deaths_F1', 'gr_new_deaths_F2']:
    df_temp = df.groupby('Country').mean()[var_name].reset_index()
    df_temp.rename(columns = {var_name: '{}_country_mean'.format(var_name)},
                   inplace = True)
    df = pd.merge(df, df_temp, how = 'left')
    df['{}_within'.format(var_name)] = df[var_name] - df['{}_country_mean'.format(var_name)] + df[var_name].mean()

# Calculate number of observations and countries
for var_name in ['gr_new_deaths_F1', 'gr_new_deaths_F2']:
  num_obs = {}
  num_obs['N_total'] = df[['net_R_within', '{}_within'.format(var_name)]].dropna().shape[0]
  num_obs['N_countries'] = len(df[['Country', 'net_R_within', '{}_within'.format(var_name)]].dropna()['Country'].unique())
  num_obs = pd.Series(num_obs)
  num_obs.to_csv('{}/scatter_num_obs_{}.csv'.format(output_folder, var_name), index = True)

for ff, ylabel in zip([1, 2],
                      ['Growth in Deaths in One Week', 'Growth in Deaths in Two Weeks']):
  fig, ax = plt.subplots(figsize = (5.0, 4.0))
  seaborn.regplot(df['net_R_within'], df['gr_new_deaths_F{}_within'.format(ff)],
                line_kws = {'color': 'k', 'alpha': 0.9},
                scatter_kws = {'alpha': 0.5, 'color': 'k'})
  plt.xlabel('Effective Repr. Number ($\\mathcal{R}-1$) This Week', fontsize = 12)
  plt.ylabel(ylabel, fontsize = 12)
  plt.text(-0.5, 4.5, 
           'Corr. = {:.2f}'.format(df[['net_R_within', 'gr_new_deaths_F{}_within'.format(ff)]].corr().values[0,1]),
           fontsize = 12)
  fig.savefig("{}/predictive_{}.png".format(output_folder, ff), bbox_inches = 'tight', dpi = 600)
  fig.savefig("{}/predictive_{}.pgf".format(output_folder, ff), bbox_inches = 'tight', dpi = 600)