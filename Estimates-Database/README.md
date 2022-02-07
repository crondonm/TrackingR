## TrackingR-Estimates
Most recent estimates for "Tracking R of COVID-19 A New Real-Time Estimation Using the Kalman Filter".
Authors: Francisco Arroyo, Francisco Bullano, Simas Kucinskas, and Carlos Rondón-Moreno.

Suggested Citation: Arroyo-Marioli F, Bullano F, Kucinskas S, Rondón-Moreno C (2021) Tracking R of COVID-19: A new real-time estimation using the Kalman filter. PLoS ONE 16(1): e0244474. https://doi.org/10.1371/journal.pone.0244474

# Changes

As of 2/7/2022, we are uploading a separate database for each serial interval option: 

  - The file "Database_X" contains the Rt estimates for a serial interval to X days.
  - The file "Database" contains the whole set of estimates.

## Variables Included

Country/Region

Date

R: contains the median estimate for Rt

CI_95_u: Upper limit 95% credible interval

CI_95_l: Lower limit 95% credible interval

CI_65_u: Upper limit 65% credible interval

CI_65_l: Lower limit 65% credible interval

Days_infectious: Serial interval for Covid-19. We provide a range from 5 to 10 days. An extense discussion on the effect of changing the serial interval can be found on the paper. Recent studies find that estimates of the serial interval for COVID-19 range between 4 and 9 days (Nishiura et al.,2020b;Park et al.,2020;Sanche et al.,2020). We suggest using 7 days (the average of the range) as point of reference. 

## Replication Files

The python code used to replicate "Tracking R of COVID-19 A New Real-Time Estimation Using the Kalman Filter" can be found at: 
https://github.com/crondonm/TrackingR

## Source of the Data

The original data are collected by the John Hopkins CSSE team and are publicly available online (https://github.com/CSSEGISandData/COVID-19).

## Questions?

You can write an email to simas [dot] kucinskas [at] hu [dash] berlin [dot] de –or to - crondonm [at] pm [dot] me -- all comments and suggestions are most welcome.
