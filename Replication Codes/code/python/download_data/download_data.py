import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

from datetime import datetime
import numpy as np
import pandas as pd

from python.tools import (
    clean_folder
)

################
## Parameters ##
################

output_folder = './download_data/output'
url_cases     = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
url_deaths    = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_NPI       = 'https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/interventions.csv'
url_mobility  = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
url_epiestim  = 'https://hsph-covid-study.s3.us-east-2.amazonaws.com/website_files/rt_table_export.csv.zip'
url_tests     = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'

##################
## Dowload data ##
##################

clean_folder(output_folder)

# Download data and save locally

#######################
## John Hopkins data ##
#######################

df = pd.read_csv(url_cases)
df.to_csv('{}/time_series_covid19_confirmed_global.csv'.format(output_folder), index = False)

df = pd.read_csv(url_recovered)
df.to_csv('{}/time_series_covid19_recovered_global.csv'.format(output_folder), index = False)

df = pd.read_csv(url_deaths)
df.to_csv('{}/time_series_covid19_deaths_global.csv'.format(output_folder), index = False)

############################################
## Imperial College data on interventions ##
############################################

df = pd.read_csv(url_NPI)
df.to_csv('{}/interventions.csv'.format(output_folder), index = False)

##########################
## Google mobility data ##
##########################

df = pd.read_csv(url_mobility)
df.to_csv('{}/Global_Mobility_Report.csv'.format(output_folder), index = False)

############################################
## Estimates of R using Cori et al method ##
############################################

df = pd.read_csv(url_epiestim)
df.to_csv('{}/R_EpiEstim.csv'.format(output_folder), index = False)

####################################
## Our World in Data Testing Data ##
####################################

df = pd.read_csv(url_tests)
df.rename(columns={
    "ISO code": "Code",
    "Date": "Day",
    "Daily change in cumulative total per thousand": "new_tests_per_thousand",
    }, inplace=True)
df[['Entity', '142606-annotations']] = df['Entity'].str.split(' - ', expand=True)
df = df[["Entity", "Code", "Day", "new_tests_per_thousand", "142606-annotations"]] 
df = df.dropna()
df.to_csv('{}/full-list-daily-covid-19-tests-per-thousand.csv'.format(output_folder), index = False)

######################
## Data description ##
######################

data_description = """# Data description

## interventions.csv

Downloaded by Simas Kucinskas from the Github repository of "Report 13 published by MRC Centre for Global Infectious Disease Analysis, Imperial College London" on {date}:

https://github.com/ImperialCollegeLondon/covid19model

## time_series_covid19_confirmed_global.csv

Downloaded by Simas Kucinskas from the 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE on {date}:

https://github.com/CSSEGISandData/COVID-19

## time_series_covid19_recovered_global.csv

Downloaded by Simas Kucinskas from the 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE on {date}:

https://github.com/CSSEGISandData/COVID-19

## time_series_covid19_deaths_global.csv

Downloaded by Simas Kucinskas from the 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE on {date}:

https://github.com/CSSEGISandData/COVID-19

## full-list-daily-covid-19-tests-per-thousand.csv

Downloaded by Simas Kucinskas from "Our World in Data" on 2020-05-07:

https://ourworldindata.org/grapher/full-list-daily-covid-19-tests-per-thousand

## Global_Mobility_Report.csv

Downloaded by Simas Kucinskas from Google Community Mobility reports on {date}:

https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

## R_EpiEstim.csv

Downloaded by Simas Kucinskas from Xihong Lin's Group in the Department of Biostatistics at the Harvard Chan School of Public Health (http://metrics.covid19-analysis.org/) on {date}:

https://hsph-covid-study.s3.us-east-2.amazonaws.com/website_files/rt_table_export.csv.zip

""".format(date = datetime.today().strftime('%Y-%m-%d'))

with open('{}/data_notes.md'.format(output_folder), 'w+') as text_file:
            text_file.write(data_description)