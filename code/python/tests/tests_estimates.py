import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

def tests_positive_estimates():
    # Check that the estimates are positive
    df = pd.read_csv('../../../fixed_revisions/derived_data/R_estimates/estimated_R.csv')
    for var_name in ['R', 'ci_95_u', 'ci_95_l', 'ci_65_u', 'ci_65_l']:
        mask = (df[var_name] < 0)
        print(df.loc[mask, ])
        assert_allclose(df.loc[mask, ].shape[0], 0)


def tests_correlate_with_KF():
    # Check correlation with frequentist
    # KF estimates; the estimates are not going
    # to be perfectly correlated, so this is a
    # sanity check.
    df_freq = pd.read_csv('../estimate_R/output/estimate_R_KF/estimated_R.csv')
    df_stan = pd.read_csv('../../../fixed_revisions/derived_data/R_estimates/estimated_R.csv')
    for var_name in ['R', 'ci_95_u', 'ci_95_l', 'ci_65_u', 'ci_65_l']:
        df_stan.rename(columns = {var_name: '{}_stan'.format(var_name)},
                       inplace = True)
    df = pd.merge(df_freq, df_stan, how = 'left', on = ['Date', 'Country/Region'])
    for var_name in ['R', 'ci_95_u', 'ci_95_l', 'ci_65_u', 'ci_65_l']:
        df_temp = df.loc[:, [var_name, '{}_stan'.format(var_name)]].copy()
        corr = df_temp.corr().values[0, 1]
        print(corr)
        assert corr >= 0.80