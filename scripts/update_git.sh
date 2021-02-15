#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N Update_Git          # Specify job name


echo "Updating Repositories"
cd  ../../TrackingR/
git add -A 
git commit -m "updated estimates"
git push https://crondonm:nVg5U5UCaIxn@github.com/crondonm/TrackingR.git

echo "Updating Website"
cd  ../TrackingR-app/
git add -A 
git commit -m "updated estimates"
git push https://crondonm:nVg5U5UCaIxn@github.com/crondonm/TrackingR-app.git
echo "Done"