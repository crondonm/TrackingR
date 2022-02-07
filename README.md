# TrackingR

Python code for "Tracking _R_ of COVID-19: A New Real-Time Estimation Using the Kalman Filter".
Authors: Francisco Arroyo, Francisco Bullano, Simas Kucinskas, and Carlos Rondón-Moreno.

Suggested citation: Arroyo-Marioli F, Bullano F, Kucinskas S, Rondón-Moreno C (2021) Tracking _R_ of COVID-19: A new real-time estimation using the Kalman filter. PLoS ONE 16(1): e0244474. https://doi.org/10.1371/journal.pone.0244474

## Download Estimates (.CSV)

We try to update the database daily. However, as we add more and more datapoints, the computation takes longer. Currently, it takes about 24 hours to compute the new set of estimates. If you see a delay of three or more days, please let us know as this is probably due a to a technical issue with our GitHub repository. Instructions for the database are available in the README file included in the folder "Estimates-Database". The stable link to the dataset is: https://github.com/crondonm/TrackingR/tree/main/Estimates-Database

Currently we provide several files to download. The file "Database.csv" includes the whole set of estimates for each serial interval option. Each of the files called "Database_X" contains the estimates calculated setting the serial interval equal to X".

## Replication Code

See [README](<Replication Codes/README.md>) file in  for detailed instructions. 

## Source of the Data

The original data are collected by the John Hopkins CSSE team and are publicly available online (https://github.com/CSSEGISandData/COVID-19).

## Change Log

* 26/3/2021: We have updated the estimation procedure to (i) use more informative priors; and (ii) allow for intra-week seasonality. With these changes, we get estimates of R that are more consistent with the growth rate of infections seen in each country.

## Questions?

You can write an email to simas [dot] kucinskas [at] hu [dash] berlin [dot] de  - or to - crondonm [at] pm [dot] me - all comments and suggestions are most welcome.

## Install Requirements

Before installing, to use the most recent version of `pip` (21.3.1 at the time
of this writing). To upgrade pip on a Linux system, run

    sudo -H pip install --upgrade pip

You could use a virtual environment, install the dependencies to your user or
make a system-wide installation.
To install all required dependencies to your current user, execute

    pip install -r requirements.txt --upgrade --user

If you want to install the dependencies system-wide, run

    sudo -H pip install -r requirements.txt

instead.
