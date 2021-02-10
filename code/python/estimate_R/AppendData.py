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

df_AB = pd.read_csv('{}/estimated_R_AB.csv'.format(output_folder))
df_CD = pd.read_csv('{}/estimated_R_CD.csv'.format(output_folder))
df_EH = pd.read_csv('{}/estimated_R_EH.csv'.format(output_folder))
df_IL = pd.read_csv('{}/estimated_R_IL.csv'.format(output_folder))
df_MP = pd.read_csv('{}/estimated_R_MP.csv'.format(output_folder))
df_QT = pd.read_csv('{}/estimated_R_QT.csv'.format(output_folder))
df_UZ = pd.read_csv('{}/estimated_R_UZ.csv'.format(output_folder))

# Append all datasets
df=df_AB.append([df_CD,df_EH,df_IL, df_MP, df_QT, df_UZ],ignore_index=True)

# Save data
df.to_csv('{}/estimated_R.csv'.format(output_folder), index = False)

