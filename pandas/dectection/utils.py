import numpy as np
import matplotlib.pyplot as plt

def baseline_select(total_number, percentage):
  if total_number <= 0:
    return 0
  return max(1, round(total_number * percentage))

def vector_label(row, num):
  array = np.zeros(num)
  array[row] = 1
  return array

def get_reporting_values(df, true_label, pred_label, baseline_value, anomaly_label):
  df_eval = df[df[pred_label] != baseline_value].copy()
  df_eval['y_true'] = df_eval[true_label].apply(lambda x: vector_label(int(x), len(anomaly_label)))
  df_eval['y_pred'] = df_eval[pred_label].apply(lambda x: vector_label(int(x), len(anomaly_label)))

  y_true_stacked = np.stack(df_eval['y_true'].values)
  y_pred_stacked = np.stack(df_eval['y_pred'].values)

  label_to_list = [label for idx, label in anomaly_label]

  return y_true_stacked, y_pred_stacked, label_to_list

def plot_graph(mode, sample_sizes,macro_precisions,macro_recalls,macro_f1s):
  plt.figure(figsize=(10, 6))
  plt.plot(sample_sizes, macro_precisions, label='Macro Precision', marker='o')
  plt.plot(sample_sizes, macro_recalls, label='Macro Recall', marker='o')
  plt.plot(sample_sizes, macro_f1s, label='Macro F1-score', marker='o')

  plt.xlabel(f'Mode: [{mode}] Baseline Sample Size (Fraction)')
  plt.ylabel('Metric Score')
  plt.title(f'Performance Metrics vs. [{mode}] Baseline Sample Size')
  plt.grid(True)
  plt.legend()
  plt.show()