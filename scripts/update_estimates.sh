#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N TrackingR           # Specify job name




qsub -N job1 update_data.sh                         # Download Data
qsub -N job2 -hold_jid job1 prior_estim.sh        # Prepare estimation: Construct databases + Run priors
qsub -N job3 -hold_jid job2 estimate_R_A.sh         # Estimation for each letter:
# qsub -N job4 -hold_jid job2 estimate_R_B.sh
# qsub -N job5 -hold_jid job2 estimate_R_C.sh
# qsub -N job6 -hold_jid job2 estimate_R_D.sh
# qsub -N job7 -hold_jid job2 estimate_R_EF.sh
# qsub -N job8 -hold_jid job2 estimate_R_GH.sh
# qsub -N job9 -hold_jid job2 estimate_R_IJ.sh
# qsub -N job10 -hold_jid job2 estimate_R_KL.sh
# qsub -N job11 -hold_jid job2 estimate_R_MN.sh
# qsub -N job12 -hold_jid job2 estimate_R_OP.sh
# qsub -N job14 -hold_jid job2 estimate_R_QR.sh
# qsub -N job15 -hold_jid job2 estimate_R_ST.sh
# qsub -N job16 -hold_jid job2 estimate_R_UZ.sh




# echo "Appending results into single file"
