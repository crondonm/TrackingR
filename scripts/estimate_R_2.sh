#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N Tr_Ns_2        # Specify job name


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
# cp ./estimate_R/output/construct_STAN_models/model_missing.pkl    ../../fixed_revisions/derived_data/STAN_models/
# cp ./estimate_R/output/construct_STAN_models/model_no_missing.pkl ../../fixed_revisions/derived_data/STAN_models/

echo "Estimating R with STAN"
cd ./estimate_R
cp ../../../fixed_revisions/derived_data/dataset/dataset_CD.csv           ./input/estimate_R_STAN/
cp ../../../fixed_revisions/derived_data/STAN_models/model_missing.pkl    ./input/estimate_R_STAN/
cp ../../../fixed_revisions/derived_data/STAN_models/model_no_missing.pkl ./input/estimate_R_STAN/
cd ..
python3 -W ignore -m estimate_R.estimate_R_STAN_2
cp ./estimate_R/output/estimate_R_STAN/estimated_R_CD.csv ../../fixed_revisions/derived_data/R_estimates/


# Append full dataset

# "I want to set the job execution order and then submit the job."

#     This is possible by executing the command below:
#     % qsub -N job1 [Script name]
#     % qsub -N job2 -hold_jid job1 [Script name]
#     % qsub -N job3 -hold_jid job1,job2 [Script name]
#     Each job is set with a new name: 'job1', 'job2' and 'job3'. When 'job1' execution is complete, 'job2' is executed; when both 'job1' and 'job2' are complete, 'job3' is executed.
#     The same setting for the execution order can be achieved by using the expression shown below:
#     % qsub -N job1 [Script name]
#     % qsub -N job2 -hold_jid job1 [Script name]
#     % qsub -N job3 -hold_jid "job*" [Script name]


# .
# "I want to set the job execution order and to control a job to be executed later based on the outcome of an earlier job."

#     "'Job B' is to be executed after the execution of 'Job A'. In the case of an error in 'Job A', I want to cancel the execution of 'Job B'." This is possible using the setting below:
#     Describe the command below in the error processing of 'Job A'.
#     % qdel jobB

#     Set the execution order of the job and submit
#     % qsub -N jobA [Script name]
#     % qsub -N jobB -hold_jid jobA [Script name]

 
