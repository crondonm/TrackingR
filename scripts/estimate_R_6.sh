#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N Tr_Ns_6       # Specify job name


module load python
cd ../code/python
# python3 -W ignore -m estimate_R.estimate_R_KF
# cp ./estimate_R/output/estimate_R_KF/optim_res.csv ../../fixed_revisions/derived_data/KF_results/

# # Estimate R using Bayesian filtering
# echo "Constructing STAN models"
# cd ./estimate_R
# cp ../../../fixed_revisions/derived_data/KF_results/optim_res.csv ./input/construct_STAN_models/
# cd ..
# python3 -W ignore -m estimate_R.construct_STAN_models
# cp ./estimate_R/output/construct_STAN_models/model_missing.pkl ../../fixed_revisions/derived_data/STAN_models/
# cp ./estimate_R/output/construct_STAN_models/model_no_missing.pkl ../../fixed_revisions/derived_data/STAN_models/

echo "Estimating R with STAN"
cd ./estimate_R
cp ../../../fixed_revisions/derived_data/dataset/dataset_QT.csv ./input/estimate_R_STAN/
cp ../../../fixed_revisions/derived_data/STAN_models/model_missing.pkl ./input/estimate_R_STAN/
cp ../../../fixed_revisions/derived_data/STAN_models/model_no_missing.pkl ./input/estimate_R_STAN/
cd ..
python3 -W ignore -m estimate_R.estimate_R_STAN_6
cp ./estimate_R/output/estimate_R_STAN/estimated_R_QT.csv ../../fixed_revisions/derived_data/R_estimates/
