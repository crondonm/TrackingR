#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N TrackingR           # Specify job name


# Remove existing fixed data revisions 
rm -r ../fixed_revisions/derived_data/dataset/
rm -r ../fixed_revisions/derived_data/KF_results/
rm -r ../fixed_revisions/derived_data/STAN_models/
rm -r ../fixed_revisions/derived_data/R_estimates/


mkdir ../fixed_revisions/derived_data/dataset/
mkdir ../fixed_revisions/derived_data/KF_results/
mkdir ../fixed_revisions/derived_data/STAN_models/
mkdir ../fixed_revisions/derived_data/R_estimates/


# Construct dataset for empirical analysis
echo "Constructing dataset"

rm -r ../code/python/construct_dataset/input/
mkdir ../code/python/construct_dataset/input/

cp ../fixed_revisions/original_data/time_series_covid19_confirmed_global.csv  ../code/python/construct_dataset/input/
cp ../fixed_revisions/original_data/time_series_covid19_deaths_global.csv     ../code/python/construct_dataset/input/
cp ../fixed_revisions/original_data/time_series_covid19_recovered_global.csv  ../code/python/construct_dataset/input/
cd ../code/python

module load python
python3 -W ignore -m construct_dataset.construct_dataset
echo "Splicing database into subsets"
python3 -W ignore -m construct_dataset.Build_parallel_datasets
cp ./construct_dataset/output/dataset.csv    ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_A.csv  ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_B.csv  ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_C.csv  ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_D.csv  ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_EF.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_GH.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_IJ.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_KL.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_MN.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_OP.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_QR.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_ST.csv ../../fixed_revisions/derived_data/dataset/
cp ./construct_dataset/output/dataset_UZ.csv ../../fixed_revisions/derived_data/dataset/

# # Run unit tests on the constructed dataset
# cd ./tests/
# nosetests tests_dataset.py -v > tests_dataset.txt 2>&1
# cd ..

# # Estimate R using Kalman filter to obtain
# # estimates of the variance of disturbances
# # and signal-to-noise ratio (used for calibrating
# # priors in Bayesian filtering with STAN)

# echo "Getting Kalman filter estimates"
# cd ./estimate_R
# rm -r ./input/
# mkdir ./input/
# mkdir ./input/estimate_R_KF/
# mkdir ./input/construct_STAN_models/
# mkdir ./input/estimate_R_STAN/
# cp ../../../fixed_revisions/derived_data/dataset/dataset.csv    ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_A.csv  ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_B.csv  ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_C.csv  ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_D.csv  ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_EF.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_GH.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_IJ.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_KL.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_MN.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_OP.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_QR.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_ST.csv ./input/estimate_R_KF/
# cp ../../../fixed_revisions/derived_data/dataset/dataset_UZ.csv ./input/estimate_R_KF/
# cd ..
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
# cd ../../scripts/

# echo "Submitting parallel routines to estimate Rt"

# # Each database receives a code:

# qsub estimate_R_1.sh # 1: dataset_AB.csv
# qsub estimate_R_2.sh # 2: dataset_CD.csv
# qsub estimate_R_3.sh # 3: dataset_EH.csv
# qsub estimate_R_4.sh # 4: dataset_IL.csv
# qsub estimate_R_5.sh # 5: dataset_MP.csv
# qsub estimate_R_6.sh # 6: dataset_QT.csv
# qsub estimate_R_7.sh # 7: dataset_UZ.csv

# echo "Appending results into single file"

