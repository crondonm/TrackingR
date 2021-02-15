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
output_folder = './get_graphs/output/table_R0'

#######################
## Construct dataset ##
#######################

clean_folder(output_folder)

df = pd.read_csv('{}/estimated_R.csv'.format(input_folder))

# Get names of countries
df_temp = pd.read_csv('{}/interventions.csv'.format(input_folder))
country_list = list(df_temp['Country'].unique())
mask = df['Country/Region'].apply(lambda x: x in country_list)
df = df.loc[mask, ]

# Get table for days infectious between 5 and 10
mask = (df['days_infectious'] >= 5) & (df['days_infectious'] <= 10)
df = df.loc[mask, ]

# Only consider the first 7 days of epidemic
# when calculating R0
for country in df['Country/Region'].unique():
    for dd in df['days_infectious'].unique():
        mask = (df['Country/Region'] == country) & (df['days_infectious'] == dd)
        df_temp = df.loc[mask, ].copy()
        df.loc[mask, 'days_since_t0'] = range(1, df_temp.shape[0] + 1)
mask = df['days_since_t0'] <= 7
df = df.loc[mask, ]

# Average over countries
df_res = df.groupby('days_infectious').mean().reset_index()

# Get LaTeX table
df_res = df_res[['days_infectious', 'R', 'ci_95_l', 'ci_95_u']]
df_res.rename(columns = {'R': '$\\hat{\\mathcal{R}}_0$',
                         'ci_95_u': 'CI Upper Bound (95\%)',
                         'ci_95_l': 'CI Lower Bound (95\%)'},
              inplace = True)
df_res.index = df_res['days_infectious']
del df_res['days_infectious']
df_res = df_res.T
f = lambda x: '{:.2f}'.format(x) 
latex = df_res.to_latex(column_format = 'l' + 'c' * 7, 
                     formatters = [f, f, f, f, f, f],
                     escape = False)
latex = latex.replace('days_infectious', 'Number of Days Infectious:')

with open('{}/table_R0.txt'.format(output_folder), 'w+') as text_file:
        text_file.write(latex)