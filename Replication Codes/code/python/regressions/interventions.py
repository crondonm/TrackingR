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

output_folder = './regressions/output/interventions/'
input_folder = './regressions/input/interventions/'
T_before = 7 # Number of days before intervention in event-study graphs
T_after = 3 * 7 # Number of days after intervention in event-study graphs
min_countries = 5 # Minimum number of countries per time in regressions
mobility_lag_list = [7, 14] # Daily lags of mobility measure to be included in regression
end_date             = '2020-05-06'        # End of sample
restrict_end_sample  = True

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data on estimates of R
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns = {'Country/Region': 'Country'},
          inplace = True)

# Look at benchmark estimates
mask = df['days_infectious'] == 7.0
df = df.loc[mask, ]

# Merge in data on interventions
df_temp = pd.read_csv('{}/interventions.csv'.format(input_folder))
df_temp.sort_values(by = ['Country', 'Date effective'], 
                    inplace = True)
mask = ((df_temp['Type'] == 'Self-isolating if ill') |
        (df_temp['Type'] == 'Public events') |
        (df_temp['Type'] == 'Lockdown') |
        (df_temp['Type'] == 'Social distancing encouraged') |
        (df_temp['Type'] == 'Schools + Universities'))
df_temp = df_temp.loc[mask, ['Country', 'Type', 'Date effective']]
df_temp.rename(columns = {'Date effective': 'Date'},
               inplace = True)
df_temp['Date'] = pd.to_datetime(df_temp['Date'], format = '%d.%m.%Y')

# Clean up variable names
df_temp['Type'] = df_temp['Type'].apply(lambda x: 'public_date' if x == 'Public events' else x)
df_temp['Type'] = df_temp['Type'].apply(lambda x: 'social_distancing_date' if x == 'Social distancing encouraged' else x)
df_temp['Type'] = df_temp['Type'].apply(lambda x: 'case_based_date' if x == 'Self-isolating if ill' else x)
df_temp['Type'] = df_temp['Type'].apply(lambda x: 'school_closure_date' if x == 'Schools + Universities' else x)
df_temp['Type'] = df_temp['Type'].apply(lambda x: 'lockdown_date' if x == 'Lockdown' else x)
df_temp['Country'] = df_temp['Country'].apply(lambda x: 'United Kingdom' if x == 'United_Kingdom' else x)

# Pivot table
df_temp.index = df_temp['Country']
del df_temp['Country']
df_temp = df_temp.pivot(columns='Type')['Date'].reset_index()

# Some countries without NPIs
# are coded as having NPIs at some faraway future
# date in the original data; replace with NaNs
for var_name in ['case_based_date', 
                 'lockdown_date', 
                 'public_date', 
                 'school_closure_date',
                 'social_distancing_date']:
    mask = df_temp[var_name] >= '2021-01-01'
    df_temp.loc[mask, var_name] = np.nan

# Merge in interventions
mask = df['Country'].apply(lambda x: x in df_temp['Country'].unique())
df = df.loc[mask, ]
assert len(df_temp['Country'].unique()) == len(df['Country'].unique()), \
  "Not all countries with data on interventions have data on R."
df = pd.merge(df, df_temp, how = 'left')

# Create dummy variables for regressions
# and calculate number of days since intervention
for var_name in ['public', 'social_distancing', 'case_based', 'lockdown', 'school_closure']:
    df[var_name] = (df['Date'] >= df['{}_date'.format(var_name)]) * 1.0
    df['days_{}'.format(var_name)] = (df['Date'] - df['{}_date'.format(var_name)]).dt.days

# Merge in data on tests
df_temp = pd.read_csv('{}/full-list-daily-covid-19-tests-per-thousand.csv'.format(input_folder))
del df_temp['Code']
df_temp['Date'] = pd.to_datetime(df_temp['Date'])
df_temp.rename(columns = {'New tests per thousand': 'change_tests_capita',
                          'Entity': 'Country'},
               inplace = True)
df = pd.merge(df, df_temp, how = 'left')

###############################
## Merge in data on mobility ##
###############################

df_temp = pd.read_csv('{}/Global_Mobility_Report.csv'.format(input_folder))

# Restrict end sample of the data
if restrict_end_sample:
  df_temp['date'] = pd.to_datetime(df_temp['date'])
  mask = df_temp['date'] <= end_date
  df_temp = df_temp.loc[mask, ]

# Only select data for countries (not subregions)
mask = df_temp['sub_region_1'].isnull() & df_temp['sub_region_2'].isnull() & df_temp['metro_area'].isnull()
df_temp = df_temp.loc[mask, ]
del df_temp['sub_region_1'], df_temp['sub_region_2'], df_temp['country_region_code']

# Tidy up
df_temp.rename(columns = {'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
                          'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
                          'parks_percent_change_from_baseline': 'parks',
                          'transit_stations_percent_change_from_baseline': 'transit',
                          'residential_percent_change_from_baseline': 'residential',
                          'workplaces_percent_change_from_baseline': 'work',
                          'date': 'Date',
                          'country_region': 'Country'},
              inplace = True)
df_temp['Date'] = pd.to_datetime(df_temp['Date'])

# Calculate first principal component
mobility_vars = ['retail_recreation',
                 'grocery_pharmacy',
                 'parks',
                 'transit',
                 'residential',
                 'work']
pca = PCA(n_components = 1)
mask = ~np.isnan(df_temp[mobility_vars]).any(axis = 1)
pca.fit(df_temp.loc[mask, mobility_vars])
print('   First principal component explains {:.2f}% of the variance in the mobility data'.format(pca.explained_variance_ratio_[0] * 100))
df_var_explained = pd.DataFrame()
df_var_explained['explained'] = pca.explained_variance_ratio_
df_var_explained.to_csv('{}/var_explained.csv'.format(output_folder), index = False)
df_temp.loc[mask, 'mobility_PC1'] = pca.fit_transform(df_temp.loc[mask, mobility_vars])

# Rotate for easier interpretability (without
# loss of generality) in case the first PC
# is negatively correlated with mobility measures
if df_temp[['mobility_PC1', 'transit']].corr().values[0, 1] < 0:
  df_temp['mobility_PC1'] = df_temp['mobility_PC1'] * (-1)
df_temp.to_csv('{}/mobility.csv'.format(output_folder), index = False)

# Calculate lags
for ll in mobility_lag_list:
  df_temp['mobility_PC1_LAG{}'.format(ll)] = df_temp.groupby('Country').shift(ll)['mobility_PC1']

# Select variables to merge in
var_list = ['mobility_PC1_LAG{}'.format(ll) for ll in mobility_lag_list]
var_list.append('mobility_PC1')
var_list.append('Date')
var_list.append('Country')
df = pd.merge(df, df_temp[var_list], how = 'left', on = ['Date', 'Country'])

# Save final dataset
df.to_csv('{}/regressions_dataset.csv'.format(output_folder), index = False)

########################
## Event-Study Graphs ##
########################

for var_name in ['public', 'social_distancing', 'case_based', 'lockdown', 'school_closure']:
  # Select countries for which we have data on
  # R T_before days before the intervention and
  # T_after days after the intervention (to make
  # sure the sample of countries is the same)

  # Calculate number of days before intervention
  df_temp = df.groupby('Country').min()['days_{}'.format(var_name)].reset_index()
  mask = (df_temp['days_{}'.format(var_name)] > 0.0)
  df_temp.loc[mask, 'days_{}'.format(var_name)] = np.nan # Intervention implemented before beginning of sample
  df_temp['days_{}'.format(var_name)] = np.abs(df_temp['days_{}'.format(var_name)])
  df_temp.dropna()
  df_temp.rename(columns = {'days_{}'.format(var_name): 'T_before'},
                 inplace = True)
  df = pd.merge(df, df_temp, how = 'left', on = 'Country')

  # Calculate number of days after intervention
  df_temp = df.groupby('Country').max()['days_{}'.format(var_name)].reset_index()
  mask = (df_temp['days_{}'.format(var_name)] < 0.0 )
  df_temp.loc[mask, 'days_{}'.format(var_name)] = np.nan # Intervention implemented after end of sample
  df_temp.rename(columns = {'days_{}'.format(var_name): 'T_after'},
                 inplace = True)
  df = pd.merge(df, df_temp, how = 'left', on = 'Country')

  # Mask for selecting a sample of countries
  # that remains fixed during the event-study time
  mask = ((df['T_before'] >= T_before) & 
          (df['T_after'] >= T_after) &
          (df['days_{}'.format(var_name)] >= -T_before) & 
          (df['days_{}'.format(var_name)] <= T_after))

  # Calculate means and s.e.'s
  df_temp = df.loc[mask, ].groupby('days_{}'.format(var_name)).mean()['R'].reset_index()
  df_se = df.loc[mask, ].groupby('days_{}'.format(var_name))['R'].agg(mean_se).reset_index()
  df_se.rename(columns = {'R': 'se'},
               inplace = True)
  df_temp = pd.merge(df_temp, df_se, on = 'days_{}'.format(var_name))
  del df['T_before'], df['T_after'] # Clean up

  # Get plots
  fig, ax = plt.subplots(figsize = (5.0, 4.0))
  plt.plot(df_temp['days_{}'.format(var_name)], df_temp['R'], '-b', linewidth = 2.5, label = '$R$')

  alpha = [0.05, 0.35]
  colors = ['b', 'r']
  names = ['95conf', '65conf']
  for aa, cc, nn in zip(alpha, colors, names):
      t_crit = scipy.stats.norm.ppf(1 - aa / 2)
      df_temp['lb_{}'.format(nn)] = df_temp['R'] - t_crit * df_temp['se']
      df_temp['ub_{}'.format(nn)] = df_temp['R'] + t_crit * df_temp['se']
      plt.fill_between(df_temp['days_{}'.format(var_name)], 
                       df_temp['lb_{}'.format(nn)], 
                       df_temp['ub_{}'.format(nn)],
                       color = cc, alpha = 0.15)

  # Save graph data
  df_temp.to_csv('{}/{}.csv'.format(output_folder, var_name), index = False)

  # Add legend for confidence bounds
  conf_95 = mpatches.Patch(color = colors[0], alpha = 0.15,
                           label = '95$\%$ Confidence Interval')
  conf_65 = mpatches.Patch(color = colors[1], alpha = 0.15, 
                           label = '65$\%$ Confidence Interval')
  plt.legend(handles = [conf_65, conf_95],
             frameon = False,
             fontsize = 10)

  # Annotate graph
  plt.text(6.0, 2.60, 
           '$N = {}$ countries'.format(int(df.loc[mask, ].groupby('days_{}'.format(var_name)).count()['R'].values[0])),
           fontsize = 12)
  plt.axvline(0, linestyle = '--', alpha = 0.25)
  plt.ylabel('Effective Repr. Number ($\\mathcal{R}$)', fontsize = 12)
  plt.xlabel('Days Since Intervention', fontsize = 12)
  fig.savefig("{}/event_study_{}.png".format(output_folder, var_name), bbox_inches = 'tight', dpi = 600)
  fig.savefig("{}/event_study_{}.pgf".format(output_folder, var_name), bbox_inches = 'tight', dpi = 600)

#################
## Regressions ##
#################

# COnstruct variable for time fixed effects
df['days_since_t0'] = np.nan
for country in df['Country'].unique():
  mask = (df['Country'] == country)
  df.loc[mask, 'days_since_t0'] = range(df.loc[mask, 'days_since_t0'].shape[0])

# Only consider stages of epidemic
# for which we have sufficiently many
# countries with data
df_temp = df.groupby('days_since_t0').count()['Country']
df_temp = df_temp.reset_index()
df_temp.rename(columns = {'Country': 'num_countries'},
               inplace = True)
df = pd.merge(df, df_temp, how = 'left')
mask = df['num_countries'] >= min_countries
df = df.loc[mask, ]

formula_baseline = 'np.log(R) ~ C(Country) + public \
                                           + social_distancing \
                                           + school_closure \
                                           + case_based \
                                           + lockdown'

mod_1 = smf.ols(formula_baseline, data = df).fit(cov_type = 'HC2')

# Add days-since-outbreak FE
formula_2 = formula_baseline + ' + C(days_since_t0)'
mod_2 = smf.ols(formula_2, data = df).fit(cov_type = 'HC2')

# Add mobility controls
for ll in mobility_lag_list:
  if ll == mobility_lag_list[0]:
    formula_3 = formula_2 + ' + mobility_PC1_LAG{}'.format(ll)
  else:
    formula_3 += ' + mobility_PC1_LAG{}'.format(ll)
mod_3 = smf.ols(formula_3, data = df).fit(cov_type = 'HC2')

# Add esting control
formula_4 = formula_3 + ' + change_tests_capita'
mod_4 = smf.ols(formula_4, data = df).fit(cov_type = 'HC2')

# Get LaTeX table
table = Stargazer([mod_1, mod_2, mod_3, mod_4])
table.rename_covariates({'public': 'Public Events',
                         'social_distancing': 'Social Distancing',
                         'school_closure': 'School Closure',
                         'case_based': 'Self Isolation',
                         'lockdown': 'Lockdown'})
table.covariate_order(['lockdown',
                       'public', 
                       'school_closure', 
                       'case_based', 
                       'social_distancing'])
checks = [{'label': 'Country FE',
           'values': [True, True, True, True]},
          {'label': 'Days-Since-Outbreak FE',
           'values': [False, True, True, True]},
          {'label': 'Mobility Controls',
           'values': [False, False, True, True]},
          {'label': 'Testing Controls',
           'values': [False, False, False, True]}]
table.add_checkmarks(checks)
table.custom_note_label('')
table.significant_digits(2)
table.dependent_variable_name('Dependent Variable: $\log(R_t)$')
latex = table.render_latex()
latex = re.search(r'\\begin{tabular}.*?\n\\end{tabular}', latex, re.DOTALL).group()
with open('{}/interventions.txt'.format(output_folder), 'w+') as text_file:
        text_file.write(latex)

# Calculate regressions where regressions are included
# one at a time
formula_baseline = 'np.log(R) ~ C(Country) + C(days_since_t0)'

formula_1 = formula_baseline + ' + lockdown'
mod_1 = smf.ols(formula_1, data = df).fit(cov_type = 'HC2')

formula_2 = formula_baseline + ' + public'
mod_2 = smf.ols(formula_2, data = df).fit(cov_type = 'HC2')

formula_3 = formula_baseline + ' + school_closure'
mod_3 = smf.ols(formula_3, data = df).fit(cov_type = 'HC2')

formula_4 = formula_baseline + ' + case_based'
mod_4 = smf.ols(formula_4, data = df).fit(cov_type = 'HC2')

formula_5 = formula_baseline + ' + social_distancing'
mod_5 = smf.ols(formula_5, data = df).fit(cov_type = 'HC2')

# Get LaTeX table
table = Stargazer([mod_1, mod_2, mod_3, mod_4, mod_5])
table.rename_covariates({'public': 'Public Events',
                         'social_distancing': 'Social Distancing',
                         'school_closure': 'School Closure',
                         'case_based': 'Self Isolation',
                         'lockdown': 'Lockdown'})
table.covariate_order(['lockdown',
                       'public', 
                       'school_closure', 
                       'case_based', 
                       'social_distancing'])
checks = [{'label': 'Country FE',
           'values': [True, True, True, True, True]},
          {'label': 'Days-Since-Outbreak FE',
           'values': [True, True, True, True, True]},
          {'label': 'Mobility Controls',
           'values': [False, False, False, False, False]},
          {'label': 'Testing Controls',
           'values': [False, False, False, False, False]}]
table.add_checkmarks(checks)
table.custom_note_label('')
table.significant_digits(2)
table.dependent_variable_name('Dependent Variable: $\log(R_t)$')
latex = table.render_latex()
latex = re.search(r'\\begin{tabular}.*?\n\\end{tabular}', latex, re.DOTALL).group()
with open('{}/interventions_one_at_a_time.txt'.format(output_folder), 'w+') as text_file:
        text_file.write(latex)