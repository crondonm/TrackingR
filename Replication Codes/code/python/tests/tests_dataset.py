import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

def tests_positive_infected():
    # Check that the time series for 
    # infected individuals is always weakly positive
    df = pd.read_csv('../../../fixed_revisions/derived_data/dataset/dataset.csv')
    for days_infectious in range(5, 10 + 1):
        if days_infectious == 5:
            mask = df['infected_{}'.format(days_infectious)] < 0.0
        else:
            mask_temp = df['infected_{}'.format(days_infectious)] < 0.0
            mask = mask | mask_temp
    test = df.loc[mask, ].shape[0] 
    print (df.loc[mask, ])
    assert_allclose(0, test)


def test_new_cases():
    # Check that new cases are always
    # weakly positive
    df = pd.read_csv('../../../fixed_revisions/derived_data/dataset/dataset.csv')
    mask = df['new_cases'] < 0
    test = df.loc[mask, ].shape[0] 
    print (df.loc[mask, ])
    assert_allclose(0, test)


def tests_growth_rates():
    # Check that the growth rate of infected
    # individuals is equal to (-gamma) when
    # new cases are zero
    df = pd.read_csv('../../../fixed_revisions/derived_data/dataset/dataset.csv')
    for days_infectious in range(5, 10 + 1):
        mask = (df['new_cases'] == 0.0) & ~np.isnan((df['gr_infected_{}'.format(days_infectious)]))
        assert_allclose(df.loc[mask, 'gr_infected_{}'.format(days_infectious)], - 1 / float(days_infectious))


def tests_growth_rates_2():
    # Check that the growth rate of infected
    # individuals is bounded below by gamma
    # when new cases are positive
    df = pd.read_csv('../../../fixed_revisions/derived_data/dataset/dataset.csv')
    for days_infectious in range(5, 10 + 1):
        mask = (df['new_cases'] > 0.0) & df['gr_infected_{}'.format(days_infectious)] <= -1 / float(days_infectious)
        assert_allclose(df.loc[mask,].shape[0], 0)


def tests_growth_rates_zero_new_cases():
    # Additional check for growth rates
    # with zero new cases
    df = pd.read_csv('../../../fixed_revisions/derived_data/dataset/dataset.csv')
    mask = df['new_cases'] == 0.0
    for days_infectious in range(5, 10 + 1):
        df['temp'] = df['infected_{}'.format(days_infectious)] / df.groupby('Country/Region').shift()['infected_{}'.format(days_infectious)] - 1
        df_temp = df.loc[:, ['temp', 'gr_infected_{}'.format(days_infectious)]].dropna()
        assert_allclose(df_temp['temp'], df_temp['gr_infected_{}'.format(days_infectious)])