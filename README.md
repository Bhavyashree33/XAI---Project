# Explainable and Uncertainty-Aware Sensor Data Manipulation Detection

A beginner-friendly prototype using simulated agricultural IoT readings.

## Features
- Simulated normal and manipulated sensor readings
- Temporal features: changes, rolling statistics, frozen readings
- Cross-sensor disagreement features
- Calibrated Random Forest classifier
- Isolation Forest comparison signal
- SHAP explanation panel (optional)
- Streamlit dashboard and CSV download
- `Likely normal`, `Suspicious`, and `Needs review` categories

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Ubuntu:
source .venv/bin/activate

pip install -r requirements.txt
```

Generate the sample dataset and train the models:

```bash
python generate_data.py
python train_model.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## CSV format for uploads

Required numeric columns:
- `temperature`
- `humidity`
- `soil_moisture`
- `temperature_peer`
- `soil_moisture_peer`

An optional `timestamp` column is recommended. An optional `label` column can be `0` for normal and `1` for manipulated, for evaluation only.

## Important limitations
- The data is synthetic and intended for implementation practice, not as evidence of real-world performance.
- The three statuses are operational categories. A suspicious reading is not proof of malicious activity.
- The displayed manipulation score is a model probability estimate calibrated on the training distribution; it may be unreliable on different datasets or under distribution shift.
- The dashboard's plain-language rule explanation is a heuristic. SHAP is optional and explains the base Random Forest output, not necessarily the calibrated probability.
- The default thresholds are demonstration settings. Tune them on validation data before any real deployment.
- The generated dataset contains deliberately injected manipulation labels. Report evaluation results as synthetic-data results, not real-world accuracy.
- For formal research, use time-aware validation, test unseen manipulation types, and avoid training/testing leakage between overlapping time windows.
