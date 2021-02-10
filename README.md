# Code for "Anomalies in China’s COVID-19 Statistics"

Python code for "Tracking R of COVID-19".

## Replication instructions

Results in the paper can be replicated by running the bash script in ./scripts:

> ./run_all_analysis

To update the data, run

> ./update_data

Currently, the data on daily testing needs to be downloaded
manually.

If the relevant Python packages are installed
and system requirements are met, these bash scripts should work out of the box
on Linux and MacOS machines. The scripts will *not* run on Windows. In that case, the
scripts should be helpful for understanding the structure of the code, and the
sequence of the analysis. The Python code itself
works across platforms, including on Windows.

The code is currently not good in collecting
the prerequisite packages in a reproducible manner 
(eg via Docker). That's on the to-do list.

## Expected runtime

Performing the empirical analysis in run_all_empirics takes ~12 hours
on an Ubuntu laptop with 2.3GHz (4 cores) and 8GB RAM. 

## System requirements

Replication files requires Python and relevant Python packages (including pandas, numpy,
and statsmodels, in particular). We recomend using the Anaconda distribution to
get these Python packages.

To get PGF files (for input in LaTeX), you may need additional
LaTeX packages installed (for example, via TeX Live on Linux).
If you do not have these packages, comment out the relevant parts in the code.

The repository also includes a modified version of the Stargazer library 
(modified to add checkmarks and some other small things) 
created by Matthew Burke (https://pypi.org/project/stargazer/).

## Sub-folder structure

The code that performs the analysis is located in ./code/.

The ./fixed_revisions/ folder holds fixed revisions of the data. This way,
parts of the analysis can be completed without running all of the code.