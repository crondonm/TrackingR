#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N TrackingR           # Specify job name



echo "Download Data"
sh update_data.sh        # Download Data
echo "Construct Datasets and Estimate Priors"
sh prior_estim.sh        # Prepare estimation: Construct databases + Run priors
echo "Submit Estimations to Clusters"
qsub -N job3 estimate_R_A.sh         
qsub -N job4  estimate_R_B.sh
qsub -N job5  estimate_R_C.sh
qsub -N job6  estimate_R_D.sh
qsub -N job7  estimate_R_EF.sh
qsub -N job8  estimate_R_GH.sh
qsub -N job9  estimate_R_IJ.sh
qsub -N job10  estimate_R_KL.sh
qsub -N job11  estimate_R_MN.sh
qsub -N job12  estimate_R_OP.sh
qsub -N job14  estimate_R_QR.sh
qsub -N job15  estimate_R_ST.sh
qsub -N job16  estimate_R_UZ.sh

echo "Append New Dataset"




# echo "Appending results into single file"
