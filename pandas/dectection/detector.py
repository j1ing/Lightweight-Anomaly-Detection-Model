import pandas as pd
from dectection import utils, stats

def cleanup_outlier(df, column_interest_list):
  for column in column_interest_list:
    series = df[column]
    lower_bound, upper_bound = stats.cap_outliers(series, lower_quantile=0.05, upper_quantile=0.95)
    
    # Cap values using vectorized operations
    df.loc[:, column] = df[column].clip(lower=lower_bound, upper=upper_bound)

  return df

def calculate_stats(df, column_interest_list):
  stats_object = {
    'mean':{},
    'std':{},
    'median':{},
    'mad':{}
  }
  for column in column_interest_list:
    series = df[column]
    stats_object['mean'][column] = stats.calculate_mean(series)
    stats_object['std'][column] = stats.calculate_std(series)
    stats_object['median'][column] = stats.calculate_median(series)
    stats_object['mad'][column] = stats.calculate_mad(series)

  return stats_object


def get_baseline(df, mode, column_interest_list, cutoff):
  df_baseline_extract = df.iloc[:cutoff]
  if mode == 'cleaned':
    df_baseline = cleanup_outlier(df_baseline_extract, column_interest_list)
  else:
    df_baseline = df_baseline_extract[df_baseline_extract['Anomaly_Label']==0]
  
  return df_baseline

def establishBaseline(df, composite_key_list, column_interest_list, sample_size, mode):
  #Series: 1D (single column) data.
  #DataFrame: 2D data with columns.
  unique_pairs = df[composite_key_list].drop_duplicates().sort_values(composite_key_list).reset_index(drop=True)

  baselines = []

  for _, row in unique_pairs.iterrows():
    #Build a boolean mask for filtering df based on all composite keys
    #initialize a 1D series with index matches the original df and default value as T
    mask = pd.Series(True, index=df.index)
    id = {}
    for key in composite_key_list:
      id[key] = row[key]
      # Update mask to True where df[key] (column values) match row[key] (value from current unique pair)
      mask &= df[key] == row[key] 
    
    #select only rows where mask is True
    filtered_df  = df[mask]
    baseline_select = utils.baseline_select(total_number=len(filtered_df), percentage=sample_size)

    df_baseline = get_baseline(df=filtered_df,
                               mode=mode, 
                               column_interest_list=column_interest_list, 
                               cutoff=baseline_select)
    
    stats_object = calculate_stats(df_baseline, column_interest_list)

    baseline_object = {
      'id': id,
      'df_baseline': df_baseline,
      'mean': stats_object['mean'],
      'std': stats_object['std'],
      'median': stats_object['median'],
      'mad': stats_object['mad']
    }
    
    baselines.append(baseline_object)
  
  return baselines

def assign_overall_label(row, column_interest_list):
  anomalies = {}
  for column in column_interest_list:
    anomalies[column] = (
      row.get(f'{column} z mean label') == 'abnormal' 
      and row.get(f'{column} z mad label') == 'abnormal'
    )
  
  count_abnormal = sum(anomalies.values())
  
  if count_abnormal == 0:
    return 0  # Normal
  elif count_abnormal > 1:
    return 4  # Combined Anomaly
  else:
    # Exactly one abnormal, find which
    if anomalies[column_interest_list[0]]:
      return 1
    elif anomalies[column_interest_list[1]]:
      return 2
    elif anomalies[column_interest_list[2]]:
      return 3

def compute_status(df, baselines, column_interest_list, confidence_level):
  threshold = stats.calculate_threshold(confidence_level=confidence_level)
  for baseline in baselines:
    group_id = baseline['id']
    df_baseline = baseline['df_baseline']
    mean = baseline['mean']
    std = baseline['std']
    median = baseline['median']
    mad = baseline['mad']

    # Build group mask
    group_mask = pd.Series(True, index=df.index)
    for key, val in group_id.items():
      group_mask &= df[key] == val

    df_selected = df[group_mask]

    # 1. Calculate all z-scores and labels first
    for column in column_interest_list:
      mean_selected = mean[column]
      std_selected = std[column]
      median_selected = median[column]
      mad_selected = mad[column]

      # Compute and assign mean z-score and labels
      df.loc[df_selected.index, f'{column} z mean'] = stats.calculate_z_score_mean(df_selected[column], mean_selected, std_selected)
      df.loc[df_selected.index, f'{column} z mean label'] = df.loc[df_selected.index, f'{column} z mean'].apply(
            lambda z: 'abnormal' if abs(z) > threshold else 'normal'
        )

      # Compute and assign mean z-score and labels
      df.loc[df_selected.index, f'{column} z mad'] = stats.calculate_z_score_mad(df_selected[column], median_selected, mad_selected)
      df.loc[df_selected.index, f'{column} z mad label'] = df.loc[df_selected.index, f'{column} z mad'].apply(
            lambda z: 'abnormal' if abs(z) > threshold else 'normal'
      )
      
    # 2. Apply overall anomaly label only for this group (after all z-scores are ready)
    df.loc[df_selected.index, 'Overall_Anomaly_Label'] = df.apply(
      lambda row: assign_overall_label(row, column_interest_list),
      axis=1
    )
    
    # 3. Mark baselines as 'baseline'
    df.loc[df_baseline.index, 'Overall_Anomaly_Label'] = 'baseline'

  return df

def detect(df, composite_key_list, column_interest_list, sample_size, confidence_level='0.997',mode='trusted'):
  baselines = establishBaseline(df, composite_key_list, column_interest_list, sample_size, mode)
  df = compute_status(df, baselines, column_interest_list, confidence_level)
  return df
