import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
import pandas as pd

from python.tools import (
    clean_folder
)


def construct_dataset(file_name, var_name):
    """Convenience function for constructing
    a clean Pandas dataframe from the CSV
    files provided by JH CSSE on their Github
    repo
    
    Args:
        file_name (str): File name / URL of CSV file
        var_name (name): Variable name
    
    Returns:
        df: Dataframe
    """
    df = pd.read_csv(file_name)
    del df['Lat'], df['Long']
    
    # Melt to long format
    df = pd.melt(df, 
                 id_vars = ['Province/State', 'Country/Region'], 
                 value_vars = list(df.columns.values[2:]))
    df.rename(columns = {'variable': 'Date',
                         'value': var_name},
              inplace = True)
    
    # For some countries, data are reported
    # by regions / states; aggregate to country level
    return df.groupby(['Country/Region', 'Date']).sum().reset_index()


################
## Parameters ##
################

input_folder         = './construct_dataset/input'
output_folder        = './construct_dataset/output'
min_cases            = 100
days_infectious_list = [5, 6, 7, 8, 9, 10] # Values of (1 / gamma) used in constructing
                                           # time series of infected individuals
end_date             = '2020-05-06'        # End of sample
restrict_end_sample  = False

#######################
## Construct dataset ##
#######################

clean_folder(output_folder)

# Read in data on total cases
df = construct_dataset(file_name = '{}/time_series_covid19_confirmed_global.csv'.format(input_folder), 
                       var_name = 'total_cases')

# Merge in recovered cases and deaths
for file_name, var_name in zip(['{}/time_series_covid19_recovered_global.csv'.format(input_folder), 
                                '{}/time_series_covid19_deaths_global.csv'.format(input_folder)],
                               ['total_recovered', 'total_deaths']):
    df_temp = construct_dataset(file_name = file_name, 
                                var_name = var_name)
    df = pd.merge(df, df_temp, 
                  on = ['Country/Region', 'Date'], 
                  how = 'left')

# Only consider days after a minimum
# number of total cases has been reached
mask = (df['total_cases'] >= min_cases)
df = df.loc[mask, ]

# Calculate world aggregates
df_temp = df.groupby('Date').sum().reset_index()
df_temp['Country/Region'] = 'World'
df = pd.concat([df_temp, df])

# Clean up the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df.reset_index(inplace = True)
del df['index']

# Remove "Diamond Princess" from the data
mask = df['Country/Region'] != 'Diamond Princess'
df = df.loc[mask, ]

# Fix a few country names
df['Country/Region'] = df['Country/Region'].apply(lambda x: 'Taiwan' if x == 'Taiwan*' else x)
df['Country/Region'] = df['Country/Region'].apply(lambda x: 'South Korea' if x == 'Korea, South' else x)

# Sort by date
df.sort_values(by = ['Country/Region', 'Date'], ascending = True,
               inplace = True)

# Construct derived flow variables (new cases / 
# recoveries / deaths)
for var_name in ['cases', 'recovered', 'deaths']:
    df['new_{}'.format(var_name)] = (df['total_{}'.format(var_name)] 
                                     - df.groupby('Country/Region').shift()['total_{}'.format(var_name)])
# Replace flow variables with 7-day moving averages
#for var_name in ['cases', 'recovered', 'deaths']:
#    df['new_{}'.format(var_name)] = df.groupby('Country/Region')['new_{}'.format(var_name)].transform(lambda x: x.rolling(7).mean())                                     

# Construct number of infected
for days_infectious in days_infectious_list:
  df['infected_{}'.format(days_infectious)] = np.nan
  for country in df['Country/Region'].unique():
      mask = df['Country/Region'] == country
      df_country = df.loc[mask, ].copy().reset_index()
      T = df_country.shape[0]

      # Initialize number of infected
      infected = np.zeros(T) * np.nan
      infected[0] = df_country['total_cases'][0]

      # Main loop
      for tt in range(1, T):
          gamma = 1 / float(days_infectious)

          # Calculate number of infected recursively;
          # In the JH CSSE dataset, there are some
          # data problems whereby new cases are occasionally
          # reported to be negative; in these case, take zero
          # when constructing time series for # of invected,
          # and then change values to NaN's later on
          infected[tt] = ((1 - gamma) * infected[tt - 1] 
                          + np.maximum(df_country['new_cases'][tt], 0.0))
      df.loc[mask, 'infected_{}'.format(days_infectious)] = infected

# In the original JH CSSE dataset, there are
# some inconsistencies in the data
# Replace with NaN's in these cases
mask = df['new_cases'] < 0
df.loc[mask, 'new_cases'] = np.nan
print('     Inconsistent observations in new_cases in JH CSSE dataset: {:}'.format(mask.sum()))
for days_infectious in days_infectious_list:
  df.loc[mask, 'infected_{}'.format(days_infectious)] = np.nan

# Calculate growth rate of infected
for days_infectious in days_infectious_list:
  df['gr_infected_{}'.format(days_infectious)] = ((df['infected_{}'.format(days_infectious)] 
      / df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)]) - 1)
  mask = df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)] == 0.0
  df.loc[mask, 'gr_infected_{}'.format(days_infectious)] = np.nan

# Deal with potential consecutive zeros in the number of infected
for days_infectious in days_infectious_list:
  mask = (df['infected_{}'.format(days_infectious)] == 0.0) & (df.groupby('Country/Region').shift(1)['infected_{}'.format(days_infectious)] == 0.0)
  df.loc[mask, 'gr_infected_{}'.format(days_infectious)] = - (1 / days_infectious)
  if mask.sum() > 0:
    print('     Number of observations with zero infected (with {} infectious days) over two consecutive days: {:}'.format(days_infectious, mask.sum()))

# Set to NaN observations with very small
# number of cases but very high growth rates
# to avoid these observations acting as
# large outliers
for days_infectious in days_infectious_list:
  gamma = 1 / float(days_infectious)
  mask = (df['new_cases'] <= 25) & (df['gr_infected_{}'.format(days_infectious)] >= gamma * (5 - 1)) # Implicit upper bound on R
  df.loc[mask, ['infected_{}'.format(days_infectious),
                'gr_infected_{}'.format(days_infectious)]] = np.nan

# Set to NaN observations implausibly
# high growth rates that are likely
# due to data issues 
for days_infectious in days_infectious_list:
    gamma = 1 / float(days_infectious)
    mask = (df['gr_infected_{}'.format(days_infectious)] >= gamma * (10 - 1)) # Implicit upper bound on R
    df.loc[mask, ['infected_{}'.format(days_infectious),
                  'gr_infected_{}'.format(days_infectious)]] = np.nan

# Remove initial NaN values for growth rates
for country in df['Country/Region'].unique():
  mask = df['Country/Region'] == country
  T = df.loc[mask, ].shape[0]
  df.loc[mask, 'days_since_min_cases'] = range(T)
mask = df['days_since_min_cases'] >= 1
df = df.loc[mask, ]
del df['days_since_min_cases']

# If requested, restrict sample period
if restrict_end_sample:
  mask = df['Date'] <= end_date
  df = df.loc[mask, ]

# Save final dataset
df.to_csv('{}/dataset.csv'.format(output_folder), index = False)