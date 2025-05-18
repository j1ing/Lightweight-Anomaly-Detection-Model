import numpy as np
from scipy.stats import norm

def calculate_mean(series):
  return np.mean(series)

def calculate_std(series):
  #population std dev: ddof=0; sample std dev: ddof=1;
  return np.std(series, ddof=1)

def calculate_median(series):
  return np.median(series)

def calculate_mad(series):
  median = np.median(series)
  return np.median(np.abs(series - median))

def calculate_z_score_mean(x, mean, std):
    if std == 0 or std is None:
        return 0
    return (x - mean) / std

def calculate_z_score_mad(x, median, mad):
    if mad == 0 or mad is None:
        return 0
    return 0.6745 * (x - median) / mad

def calculate_threshold(confidence_level):
   #Given a confidence level (e.g., 0.95), return the corresponding two-tailed Z-score threshold.
   return norm.ppf(1 - (1 - confidence_level) / 2)
   

def cap_outliers(series, lower_quantile, upper_quantile):
  lower_bound = series.quantile(lower_quantile)
  upper_bound = series.quantile(upper_quantile)
  return lower_bound, upper_bound