import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))

import numpy as np
import pandas as pd



################
## Parameters ##
################

input_folder         = './construct_dataset/input'
output_folder        = './construct_dataset/output'

#######################
## Construct dataset ##
#######################

df = pd.read_csv('{}/dataset.csv'.format(output_folder))

# Separar de A-D
df_AB=df[df["Country/Region"].str[0:1]=="A"]
df_temp1=df[df["Country/Region"].str[0:1]=="B"]
df_AB=df_AB.append(df_temp1)

df_CD=df[df["Country/Region"].str[0:1]=="C"]
df_temp1=df[df["Country/Region"].str[0:1]=="D"]
df_CD=df_CD.append(df_temp1)

df_EH=df[df["Country/Region"].str[0:1]=="E"]
for i in ["F","G","H"]:
    df_temp2=df[df["Country/Region"].str[0:1]==i]
    df_EH=df_EH.append(df_temp2)

df_IL=df[df["Country/Region"].str[0:1]=="I"]
for i in ["J","K","L"]:
    df_temp2=df[df["Country/Region"].str[0:1]==i]
    df_IL=df_IL.append(df_temp2)

df_MP=df[df["Country/Region"].str[0:1]=="M"]
for i in ["N","O","P"]:
    df_temp2=df[df["Country/Region"].str[0:1]==i]
    df_MP=df_MP.append(df_temp2)

df_QT=df[df["Country/Region"].str[0:1]=="Q"]
for i in ["R","S","T"]:
    df_temp3=df[df["Country/Region"].str[0:1]==i]
    df_QT=df_QT.append(df_temp3)

df_UZ=df[df["Country/Region"].str[0:1]=="U"]
for i in ["V","W","X","Y","Z"]:
    df_temp3=df[df["Country/Region"].str[0:1]==i]
    df_UZ=df_UZ.append(df_temp3)


# Save final dataset
df_AB.to_csv('{}/dataset_AB.csv'.format(output_folder), index = False)
df_CD.to_csv('{}/dataset_CD.csv'.format(output_folder), index = False)
df_EH.to_csv('{}/dataset_EH.csv'.format(output_folder), index = False)
df_IL.to_csv('{}/dataset_IL.csv'.format(output_folder), index = False)
df_MP.to_csv('{}/dataset_MP.csv'.format(output_folder), index = False)
df_QT.to_csv('{}/dataset_QT.csv'.format(output_folder), index = False)
df_UZ.to_csv('{}/dataset_UZ.csv'.format(output_folder), index = False)


