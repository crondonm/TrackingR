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
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns

from python.tools import (
    clean_folder
)

################
## Parameters ##
################

input_folder  = './get_graphs/input'
output_folder = './get_graphs/output/plot_raw_growth_rate'
days_infectious = 7

#######################
## Construct dataset ##
#######################

clean_folder(output_folder)

# Read data 
df = pd.read_csv('{}/dataset.csv'.format(input_folder))

# Merge in estimates
df_temp = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
mask = df_temp['days_infectious'] == days_infectious
df_temp = df_temp.loc[mask, ]
df = pd.merge(df, df_temp[['Date', 'Country/Region', 'R']], how = 'left')
df['gr_infected_smoothed'] = (1 / float(days_infectious)) * (df['R'] - 1)
df['Date'] = pd.to_datetime(df['Date'])

# Plot raw data on growth of infected
# for China, Italy, and US
country_list = ['China', 'Italy', 'US', ]
fig, ax = plt.subplots(figsize = (5.0, 4.0))
colors = ['r', 'b', 'g']
df_data_graph_temp = pd.DataFrame()
for country, color in zip(country_list, colors):
    mask = (df['Country/Region'] == country)
    df_temp = df.loc[mask, ].copy()
    plt.plot(df_temp['Date'], df_temp['gr_infected_{}'.format(days_infectious)],
             color = color, 
             linestyle = '-',
             alpha = 0.9,
             label = country)
    plt.plot(df_temp['Date'], df_temp['gr_infected_smoothed'],
             color = color, 
             linestyle = '--',
             alpha = 0.9)
    df_data_graph_temp = pd.concat([df_data_graph_temp, df_temp[['Country/Region', 'Date', 
                                                                 'gr_infected_{}'.format(days_infectious), 'gr_infected_smoothed']]])
plt.legend(frameon = False,
           fontsize = 12)

plt.ylabel('Growth Rate Number of Infected', fontsize = 12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
df_data_graph_temp.to_csv('{}/growth_rate_countries.csv'.format(output_folder), index = False)
fig.savefig("{}/raw_data_infected.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/raw_data_infected.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)