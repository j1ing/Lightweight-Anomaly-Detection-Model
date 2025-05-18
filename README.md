# Lightweight-Anomaly-Detection-Model

## 🔧Execution:
1. Change directory to:
<pre> <code>\lightweight_anormaly_detection</code> </pre>
3. Create a virtual environment

Windows (Command Prompt):
<pre> <code>py -m venv .venv</code> </pre>
<pre> <code>.venv\Scripts\activate</code> </pre>

  
macOS/Linux (Terminal):
<pre> <code>python3 -m venv .venv</code> </pre>
<pre> <code>source .venv/bin/activate</code> </pre>

4. Install dependencies
<pre> <code>pip install -r requirements.txt</code> </pre>

5. Run the command for pandas version:
<pre> <code>python pandas\main.py</code> </pre>



## 📈 Purpose

This project implements a lightweight anomaly detection model using statistical heuristics instead of AI.

While it may not match the performance of advanced AI models, it achieves comparable results using small sample sizes.

The main goals are simplicity, efficiency, and interpretability.

### 🧠 Quick Comparison:
<pre> <code>
Aspect			Lightweight Model		AI Model
Accuracy		Moderate			High
Training Time		Very Low			High
Data Needs		Low				High
Interpretability	High (transparent)		Low (black box)
Complexity		Simple				Complex
Best Use Case		Fast, low-resource setups	High-scale, data-rich environment
</code> </pre>
Despite its simplicity, the lightweight model achieves high performance with just 20% training data:

<pre> <code>
precision	94.66%
recall 		99.4%
F1 score 	96.93% 
</code> </pre>
To demonstrate its effectiveness, I use the Power Grid Sense dataset from Kaggle.

https://www.kaggle.com/datasets/ziya07/powergridsense-dataset/data

This model uses a baseline-based z-score method to detect abnormalities in voltage, frequency, and power factor readings.

Due to the nature of the dataset, a one-time baseline calibration is used, as the sensor data is expected to remain consistent over time.

More advanced methods—such as sliding window baselines, trigger-based recalibration, or time-based recalibration — are not applied


## 📂 Project Structure
<pre> <code>
.
├── power_system_multiclass_anomaly_data.csv
├── pandas/
│   ├── main.py
│   └── eval_data_files/
│   └── eval_metric_files/
│   └── eval_performance_graphs/
│   └── detection/
│   	├── detector.py
│   	├── stats.py
│   	└── utils.py
├── requirement.txt
└── README.txt
</code> </pre>


## 🚀 How It Works:

- Builds per-sensor baselines using statistical measures (mean/std, median/MAD)

- Flags readings as anomalous using dual z-score checks

- Aggregates results into multiclass anomaly labels:

  0 = Normal (All values are within normal operating range)

  1 = Voltage Anomaly (Abnormal voltage (e.g., sag/surge))

  2 = Frequency Anomaly (Frequency deviates from nominal 50Hz)

  3 = Power Factor Anomaly (Power factor is abnormally low (< 0.7))

  4 = Combined Anomaly (Two or more anomalies present simultaneously)


### Baseline Sampling

For each unique (Sensor_ID, Location) pair, a subset of rows is selected as the baseline. 

#### Two modes are supported:
- trusted

   Uses only explicitly labeled normal data (Anomaly Label == 0)(default mode)
- cleaned

   Uses outlier-capped initial samples

#### Statistical Computation:
- Mean & Standard Deviation

  Accurate & sensitive & efficient when data is under normal distribution.
- Median & Median Absolute Deviation (MAD)

  very robust and reliable under noise (outliers) or non-normal distributions.

Typically, mean/std is used initially, with median/MAD introduced later for robustness. 

This model uses both simultaneously to increase confidence:

Only when both agree on an anomaly does the model flag a point, greatly reducing false positives.

### Anomaly Detection:

Dual z-score method compares current readings to both mean and median baselines. 

A point is considered anomalous only if both z-scores exceed the threshold (default: 99.7% confidence).

### Evaluation:
- Performs multiple experiments over varying sample sizes
- Reports macro-averaged precision, recall, F1-score.
- Report data and performance graphs are saved.



## 🔍 Core Files
<pre> <code>
File		Purpose
main.py		Runs experiments across modes/sample sizes
detector.py	Contains baseline building and anomaly detection logic
stats.py	Computes statistical metrics and z-scores
utils.py	Evaluation utilities (e.g., metrics, plotting)
</code> </pre>


