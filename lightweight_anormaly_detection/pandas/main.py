import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
from dectection import detector, utils

base_dir = Path(__file__).resolve().parent.parent
file = base_dir /'power_system_multiclass_anomaly_data.csv'
composite_key_list = ['Sensor_ID', 'Location']
column_interst_list = ['Voltage (V)','Frequency (Hz)','Power_Factor']
sample_size_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
confidence_level = 0.997
mode_list = ['trusted','cleaned']
anomaly_label = [(0,'Normal'),
                 (1,'Voltage Anomaly'),
                 (2,'Frequency Anomaly'),
                 (3,'Power Factor Anomaly'),
                 (4,'Combined Anomaly')]

def main():
  df = pd.read_csv(file)
  df_sorted = df.sort_values(['Timestamp']).reset_index(drop=True)
  for mode in mode_list:
    macro_precisions_list = []
    macro_recalls_list = []
    macro_f1s_list = []
    for sample_size in sample_size_list:
      df_detect_input = df_sorted.copy()
      df = detector.detect(df=df_detect_input, 
                          composite_key_list=composite_key_list,
                          column_interest_list=column_interst_list,
                          sample_size=sample_size,
                          confidence_level=confidence_level,
                          mode=mode)

      df.to_csv(base_dir /f'pandas/eval_data_files/eval_{mode}_{sample_size}.csv', index=False)

      y_true_stacked, y_pred_stacked, label_to_list = utils.get_reporting_values(df=df,
                                                                                 true_label='Anomaly_Label', 
                                                                                 pred_label='Overall_Anomaly_Label', 
                                                                                 baseline_value='baseline', 
                                                                                 anomaly_label=anomaly_label)
      print(f'Sampling mode: {mode}')
      print(f'Sample size: {sample_size*100}%')
      report = classification_report(y_true=y_true_stacked, 
                                     y_pred=y_pred_stacked, 
                                     target_names=label_to_list)
      print(report)

      report_object = classification_report(y_true=y_true_stacked, 
                                            y_pred=y_pred_stacked, 
                                            target_names=label_to_list,
                                            output_dict=True)
      
      macro_precisions_list.append(report_object['macro avg']['precision'])
      macro_recalls_list.append(report_object['macro avg']['recall'])
      macro_f1s_list.append(report_object['macro avg']['f1-score'])

    df_metrics = pd.DataFrame({
        'sample_size': sample_size_list,
        'macro_precision': macro_precisions_list,
        'macro_recall': macro_recalls_list,
        'macro_f1': macro_f1s_list
    })
    df_metrics.to_csv(base_dir /f'pandas/eval_metric_files/eval_metrics_{mode}.csv', index=False)

    utils.plot_graph(mode, sample_size_list,macro_precisions_list,macro_recalls_list,macro_f1s_list)

main()