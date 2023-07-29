# Individual to colony-level response to heat stress inside a honey bee colony

This contains code needed to reproduce the figures in the paper:

Jhawar, J., Davidson, J.D., Weidenmuller, A., Dormagen, D.M., Landgraf, T., Couzin, I.D., Smith, M.L., Individual to colony-level response to heat stress inside a honey bee colony. In review.

The full dataset, including x-y trajectories and behavioral metrics calculated at different timescales (per-hour,per-5 minute, per-1 minute), is available at Zenodo:  http://doi.org/10.5281/zenodo.7298798

Data for the queen is at http://doi.org/10.5281/zenodo.7298798

Main analysis files descriptions:
- **'Fig1-Temp+Comb.ipynb', 'Fig1-TimeSeriesTrends.ipynb','Figs2-5-Hist+Embeddings.ipynb', 'Queen-analysis.ipynb'**:  These contain all the plots and results in the paper, and can be run with the included data
  
Other files:
- **datafunctions.py, definitions_2019.py, displayfunctions.py, all_cohorts_2019.csv, summary_experiments_2019.csv**:  Contain functions and definitions used in the analysis
- **all_heat_trials_temp_loggers.csv.zip**:  data for temperature within the hive used in the analysis

See also the associated repository that lists experimental details at https://github.com/jacobdavidson/bees_lifetimetracking_2019data
