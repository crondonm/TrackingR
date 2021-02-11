#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N Update Data         # Specify job name



# Remove existing fixed data revisions (if any exist)
rm -r ../fixed_revisions/original_data/
mkdir ../fixed_revisions/original_data/

# load module
module load python

# Update datasets
echo "Downloading data"
cd ../code/python
python3 -W ignore -m download_data.download_data
cp -a ./download_data/input/full-list-daily-covid-19-tests-per-thousand.csv ../../fixed_revisions/original_data/ # Downloaded manually
cp -a ./download_data/output/. ../../fixed_revisions/original_data/