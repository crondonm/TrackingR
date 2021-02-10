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

# Splice Database

df_A=df[df["Country/Region"].str[0:1]=="A"]
df_B=df[df["Country/Region"].str[0:1]=="B"]
df_C=df[df["Country/Region"].str[0:1]=="C"]
df_D=df[df["Country/Region"].str[0:1]=="D"]

df_EF=df[df["Country/Region"].str[0:1]=="E"]
df_temp1=df[df["Country/Region"].str[0:1]=="F"]
df_EF=df_EF.append(df_temp1)

df_GH=df[df["Country/Region"].str[0:1]=="G"]
df_temp1=df[df["Country/Region"].str[0:1]=="H"]
df_GH=df_GH.append(df_temp1)

df_IJ=df[df["Country/Region"].str[0:1]=="I"]
df_temp1=df[df["Country/Region"].str[0:1]=="J"]
df_IJ=df_IJ.append(df_temp1)

df_KL=df[df["Country/Region"].str[0:1]=="K"]
df_temp1=df[df["Country/Region"].str[0:1]=="L"]
df_KL=df_KL.append(df_temp1)

df_MN=df[df["Country/Region"].str[0:1]=="M"]
df_temp1=df[df["Country/Region"].str[0:1]=="N"]
df_MN=df_MN.append(df_temp1)

df_OP=df[df["Country/Region"].str[0:1]=="O"]
df_temp1=df[df["Country/Region"].str[0:1]=="P"]
df_OP=df_OP.append(df_temp1)

df_QR=df[df["Country/Region"].str[0:1]=="Q"]
df_temp1=df[df["Country/Region"].str[0:1]=="R"]
df_QR=df_QR.append(df_temp1)

df_ST=df[df["Country/Region"].str[0:1]=="S"]
df_temp1=df[df["Country/Region"].str[0:1]=="T"]
df_ST=df_ST.append(df_temp1)

df_UZ=df[df["Country/Region"].str[0:1]=="U"]
for i in ["V","W","X","Y","Z"]:
    df_temp3=df[df["Country/Region"].str[0:1]==i]
    df_UZ=df_UZ.append(df_temp3)


# Save final dataset
df_A.to_csv('{}/dataset_A.csv'.format(output_folder), index = False)
df_B.to_csv('{}/dataset_B.csv'.format(output_folder), index = False)
df_C.to_csv('{}/dataset_C.csv'.format(output_folder), index = False)
df_D.to_csv('{}/dataset_D.csv'.format(output_folder), index = False)
df_EF.to_csv('{}/dataset_EF.csv'.format(output_folder), index = False)
df_GH.to_csv('{}/dataset_GH.csv'.format(output_folder), index = False)
df_IJ.to_csv('{}/dataset_IJ.csv'.format(output_folder), index = False)
df_KL.to_csv('{}/dataset_KL.csv'.format(output_folder), index = False)
df_MN.to_csv('{}/dataset_MN.csv'.format(output_folder), index = False)
df_OP.to_csv('{}/dataset_OP.csv'.format(output_folder), index = False)
df_QR.to_csv('{}/dataset_QR.csv'.format(output_folder), index = False)
df_ST.to_csv('{}/dataset_ST.csv'.format(output_folder), index = False)
df_UZ.to_csv('{}/dataset_UZ.csv'.format(output_folder), index = False)


