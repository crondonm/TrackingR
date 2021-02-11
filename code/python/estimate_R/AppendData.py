import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
import pandas as pd


################
## Parameters ##
################

input_folder  = './estimate_R/output/estimate_R_STAN'
output_folder = './estimate_R/output/estimate_R_STAN'

#######################
## Construct dataset ##
#######################

df_A = pd.read_csv('{}/estimated_R_A.csv'.format(output_folder))
df_B = pd.read_csv('{}/estimated_R_B.csv'.format(output_folder))
df_C = pd.read_csv('{}/estimated_R_C.csv'.format(output_folder))
df_D = pd.read_csv('{}/estimated_R_D.csv'.format(output_folder))
df_EF = pd.read_csv('{}/estimated_R_EF.csv'.format(output_folder))
df_GH = pd.read_csv('{}/estimated_R_GH.csv'.format(output_folder))
df_IJ = pd.read_csv('{}/estimated_R_IJ.csv'.format(output_folder))
df_KL = pd.read_csv('{}/estimated_R_KL.csv'.format(output_folder))
df_MN = pd.read_csv('{}/estimated_R_MN.csv'.format(output_folder))
df_OP = pd.read_csv('{}/estimated_R_OP.csv'.format(output_folder))
df_QR = pd.read_csv('{}/estimated_R_QR.csv'.format(output_folder))
df_ST = pd.read_csv('{}/estimated_R_ST.csv'.format(output_folder))
df_UZ = pd.read_csv('{}/estimated_R_UZ.csv'.format(output_folder))

# Append all datasets
df=df_A.append([df_B, df_C, df_D, df_EF, df_GH, df_IJ, df_KL, df_MN, df_OP, df_QR, df_ST, df_UZ],ignore_index=True)

del df['n_eff_pct']
del df['Rhat_diff'] 
del df['signal_to_noise']
del df['var_irregular']


# Save data
df.to_csv('{}/database.csv'.format(output_folder), index = False)

