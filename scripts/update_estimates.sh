#!/bin/bash
#$ -M crondonm@nd.edu     # Email address for job notification
#$ -m abe                 # Send mail when job begins, ends and aborts
#$ -q long                # Specify queue
#$ -pe smp 1              # Specify number of cores to use.
#$ -N TrackingR           # Specify job name

# module load python

# echo "Download Data"
# sh update_data.sh        # Download Data

# echo "Construct Datasets and Estimate Priors"
# sh prior_estim.sh        # Prepare estimation: Construct databases + Run priors

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
qsub -N job13  estimate_R_QR.sh
qsub -N job14  estimate_R_ST.sh
qsub -N job15  estimate_R_UZ.sh

echo "Append New Dataset"
qsub -N job16 -hold_jid "job*" Appenddata.sh

#     This is possible by executing the command below:
#     % qsub -N job1 [Script name]
#     % qsub -N job2 -hold_jid job1 [Script name]
#     % qsub -N job3 -hold_jid job1,job2 [Script name]
#     Each job is set with a new name: 'job1', 'job2' and 'job3'. When 'job1' execution is complete, 'job2' is executed; when both 'job1' and 'job2' are complete, 'job3' is executed.
#     The same setting for the execution order can be achieved by using the expression shown below:
#     % qsub -N job1 [Script name]
#     % qsub -N job2 -hold_jid job1 [Script name]
#     % qsub -N job3 -hold_jid "job*" [Script name]


# echo "Appending results into single file"
