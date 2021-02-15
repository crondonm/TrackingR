import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
import pandas as pd

from python.tools import (
    clean_folder
)

# Formatters for LaTeX output
def f1(x):
    return '%1.0f' % x

def f2(x):
    return '%1.2f' % x

################
## Parameters ##
################

output_folder = './regressions/output/correlate_epiestim/'
input_folder = './regressions/input/correlate_epiestim/'
min_T = 20 # Minimum number of time-series observations

###############
## Load data ##
###############

clean_folder(output_folder)

# Load data on our estimates of R
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])

# Look at benchmark estimates
mask = df['days_infectious'] == 7
df = df.loc[mask, ]

# Remove World aggregate
mask = df['Country/Region'] == 'World'
df = df.loc[~mask, ]

# Load estimates of R using the Cori et al method
df_epi = pd.read_csv('{}/R_EpiEstim.csv'.format(input_folder))

## Clean up EpiEstim estimates

# Only look at country-level estimates
mask = (df_epi['resolution'] == 'country')
df_temp = df_epi.loc[mask, ].copy()

# Manually set missing codes to NaN & clean up
mask = (df_temp['Rt_plot'] == -888) | (df_temp['Rt_plot'] == -88)
df_temp.loc[mask, 'Rt_plot'] = np.nan

df_temp = df_temp[['dispID', 'date', 'Rt_plot']]
df_temp.rename(columns = {'dispID': 'Country/Region',
                          'date': 'Date',
                          'Rt_plot': 'R_EpiEstim'},
               inplace = True)
df_temp['Date'] = pd.to_datetime(df_temp['Date'])

# Replace country names with consistency
# with our naming conventions
mask = (df_temp['Country/Region'] == 'Korea, South')
df_temp.loc[mask, 'Country/Region'] = 'South Korea'

mask = (df_temp['Country/Region'] == 'Taiwan*')
df_temp.loc[mask, 'Country/Region'] = 'Taiwan'

# Merge in to main dataset
df = pd.merge(df, df_temp, 
  on = ['Country/Region', 'Date'], how = 'left')

############################
## Calculate correlations ##
############################

res = []
for country in df['Country/Region'].unique():
    mask = df['Country/Region'] == country
    df_temp = df.loc[mask, ].copy()
    corr = df_temp[['R', 'R_EpiEstim']].corr().values[0, 1]
    N = np.min([df_temp['R'].count(), df_temp['R_EpiEstim'].count()])
    res.append({'Country/Region': country,
                'corr': corr,
                'T': N})
res = pd.DataFrame(res)

# Collect results and save
mask = (res['T'] >= min_T)
N = res.loc[mask, 'corr'].count()
mean_T = res.loc[mask, 'T'].mean()
mean_corr = res.loc[mask, 'corr'].mean()
p5 = res.loc[mask, 'corr'].quantile(0.05)
p25 = res.loc[mask, 'corr'].quantile(0.25)
p50 = res.loc[mask, 'corr'].quantile(0.50)
p75 = res.loc[mask, 'corr'].quantile(0.75)
p95 = res.loc[mask, 'corr'].quantile(0.95)

# Create table
table = pd.DataFrame()
table['No. Countries'] = pd.Series(N)
table['Avg. Sample Size'] = mean_T
table['Avg. Corr.'] = mean_corr
table['P5'] = p5
table['P25'] = p25
table['P50'] = p50
table['P75'] = p75
table['P95'] = p95

# Convert to LaTeX input
latex = table.to_latex(column_format = 'll|llllll', 
                       escape = False, 
                       index = False,
                       formatters = [f1, f2, f2, f2, f2, f2, f2, f2])

with open('{}/correlate_epiestim.txt'.format(output_folder), 'w+') as text_file:
        text_file.write(latex)