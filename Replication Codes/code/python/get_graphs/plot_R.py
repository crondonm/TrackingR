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
output_folder = './get_graphs/output/R'
days_infectious = 7.0

#######################
## Construct dataset ##
#######################

clean_folder(output_folder)

# Read data 
df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))
df['Date'] = pd.to_datetime(df['Date'])

##########################################
## Calculate number of days until R < 1 ##
##########################################

days_to_below_one = []
for country in ['China', 'US', 'Italy', 'Germany']:
    mask = (df['Country/Region'] == country) & (df['days_infectious'] == days_infectious)
    df_temp = df.loc[mask, ].copy()
    start_epidemic = df_temp['Date'].min()
    mask = (df_temp['R'] < 1)
    date_below_one = df_temp.loc[mask, 'Date'].min()
    diff = (date_below_one - start_epidemic).days
    days_to_below_one.append({'Country/Region': country,
                'days_to_below_one': diff})
days_to_below_one = pd.DataFrame(days_to_below_one)
days_to_below_one.to_csv('{}/days_to_below_one.csv'.format(output_folder), index = False)

###############
## Get plots ##
###############

# Plot estimated R for China, US, and Italy
country_list = ['China', 'Italy', 'US', ]
fig, ax = plt.subplots(figsize = (5.0, 4.0))
colors = ['r', 'b', 'g']
styles = ['-', '--', '-.']
df_data_graph_temp = pd.DataFrame()
for country, color, style in zip(country_list, colors, styles):
    mask = (df['Country/Region'] == country) & (df['days_infectious'] == days_infectious)
    df_temp = df.loc[mask, ].copy()
    plt.plot(df_temp['Date'], df_temp['R'],
             color = color, 
             linestyle = style,
             alpha = 0.9,
             label = country)
    plt.fill_between(df_temp['Date'], 
                     df_temp['ci_65_l'], 
                     df_temp['ci_65_u'],
                     color = color, alpha = 0.15)
    df_data_graph_temp = pd.concat([df_data_graph_temp, df_temp[['Country/Region', 'Date', 'R', 'ci_65_l', 'ci_65_u']]])
plt.legend(frameon = False,
           fontsize = 12)

plt.axhline(1, linestyle = '--', color = 'k', alpha = 0.25)
plt.axhline(0, linestyle = '--', color = 'k', alpha = 0.25)
plt.ylabel('Effective Repr. Number ($\\mathcal{R}$)', fontsize = 12)
plt.ylim(-0.1, 4.15)
plt.yticks([0, 1, 2, 3, 4])
ax.set_xlim([dt.date(2020, 1, 21), dt.date(2020, 5, 8)])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
df_data_graph_temp.to_csv('{}/R_countries.csv'.format(output_folder), index = False)
fig.savefig("{}/R_countries.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/R_countries.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)

# Plot estimated R for India, Brazil, and Germany
country_list = ['Brazil', 'India', 'Germany']
fig, ax = plt.subplots(figsize = (5.0, 4.0))
colors = ['m', 'c', 'y']
styles = ['-', '--', '-.']
df_data_graph_temp = pd.DataFrame()
for country, color, style in zip(country_list, colors, styles):
    mask = (df['Country/Region'] == country) & (df['days_infectious'] == days_infectious)
    df_temp = df.loc[mask, ].copy()
    plt.plot(df_temp['Date'], df_temp['R'],
             color = color, 
             linestyle = style,
             alpha = 0.9,
             label = country)
    plt.fill_between(df_temp['Date'], 
                     df_temp['ci_65_l'], 
                     df_temp['ci_65_u'],
                     color = color, alpha = 0.15)
    df_data_graph_temp = pd.concat([df_data_graph_temp, df_temp[['Country/Region', 'Date', 'R', 'ci_65_l', 'ci_65_u']]])
plt.legend(frameon = False,
           fontsize = 12)

plt.axhline(1, linestyle = '--', color = 'k', alpha = 0.25)
plt.axhline(0, linestyle = '--', color = 'k', alpha = 0.25)
plt.ylabel('Effective Repr. Number ($\\mathcal{R}$)', fontsize = 12)
plt.ylim(-0.1, 4.15)
plt.yticks([0, 1, 2, 3, 4])
ax.set_xlim([dt.date(2020, 1, 21), dt.date(2020, 5, 8)])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
df_data_graph_temp.to_csv('{}/R_countries_2.csv'.format(output_folder), index = False)
fig.savefig("{}/R_countries_2.png".format(output_folder), bbox_inches = 'tight', dpi = 600)
fig.savefig("{}/R_countries_2.pgf".format(output_folder), bbox_inches = 'tight', dpi = 600)

# Plot estimated R for every country
for country in df['Country/Region'].unique():
    mask = (df['Country/Region'] == country) & (df['days_infectious'] == days_infectious)
    df_temp = df.loc[mask, ].copy()
    fig, ax = plt.subplots(figsize = (5.0, 4.0))
    plt.plot(df_temp['Date'], df_temp['R'], 'k', alpha = 0.9)

    # Plot confidence bounds
    # Significance levels are hard-coded
    levels = [95, 65]
    colors = ['b', 'r']
    for level, cc in zip(levels, colors):
        plt.fill_between(df_temp['Date'], 
                         df_temp['ci_{}_l'.format(level)], 
                         df_temp['ci_{}_u'.format(level)],
                         color = cc, alpha = 0.15)
        
    # Add legend for confidence bounds
    conf_95 = mpatches.Patch(color = colors[0], alpha = 0.15,
                             label = '95$\%$ Credible Interval')
    conf_65 = mpatches.Patch(color = colors[1], alpha = 0.15, 
                             label = '65$\%$ Credible Interval')
    plt.legend(handles = [conf_65, conf_95],
               frameon = False,
               fontsize = 10)

    plt.axhline(1, linestyle = '--', color = 'k', alpha = 0.25)
    plt.axhline(0, linestyle = '--', color = 'k', alpha = 0.25)
    plt.ylabel('Effective Repr. Number ($\\mathcal{R}$)', fontsize = 12)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    if country == 'World':
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 14))
    else:
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 7))
        fig.autofmt_xdate()
    fig.savefig("{}/R_{}.png".format(output_folder, country), bbox_inches = 'tight', dpi = 600)
    fig.savefig("{}/R_{}.pgf".format(output_folder, country), bbox_inches = 'tight', dpi = 600)