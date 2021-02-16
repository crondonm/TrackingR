#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N AppendData          # Specify job name

module load python
echo "Appending Datasets"
cd ../code/python
python3 -W ignore -m estimate_R.AppendData
cp ./estimate_R/output/estimate_R_STAN/database.csv ../../fixed_revisions/derived_data/R_estimates/
cp ./estimate_R/output/estimate_R_STAN/database.csv ../../../TrackingR/Estimates-Database/
cp ./estimate_R/output/estimate_R_STAN/database.csv ../../../TrackingR-app/data/
pwd

