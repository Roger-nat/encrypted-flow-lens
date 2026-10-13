# Encrypted Flow Lens
## Network Security Monitoring Without Payload Inspection

A learning and portfolio project that demonstrates how machine learning can classify network flows using metadata such as duration, packet counts, byte counts, packet rate, inter-arrival time, destination port, and SYN-flag ratio. It does not inspect packet payload content.

> **Important limitation:** The included dataset is synthetic and the labels are generated for demonstration. This is not a validated detector, not trained on production traffic, and not suitable for security decisions without real datasets, rigorous evaluation, and analyst review.

## Features
- Reproducible synthetic network-flow dataset generation
- Feature-based workflow that avoids payload inspection
- Logistic Regression and Random Forest baselines
- Stratified train/test split and macro precision/recall/F1
- Confusion matrix and classification report
- Streamlit dashboard for dataset exploration and interactive single-flow classification
- CSV download for the sample dataset

## Tech stack
Python, Pandas, NumPy, scikit-learn, Joblib, Streamlit.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, run `python -m pip install -r requirements.txt` without activating the environment, or use a terminal where activation is permitted.

## Dataset schema
- `flow_duration_ms`
- `fwd_packets`, `bwd_packets`
- `fwd_bytes`, `bwd_bytes`
- `packet_rate`
- `mean_interarrival_ms`
- `destination_port`
- `syn_flag_ratio`
- `bytes_per_packet`
- `label` (`benign` or `suspicious`)

## Methodology
1. Generate reproducible synthetic flow records.
2. Separate metadata features (`X`) from labels (`y`).
3. Split train/test data with stratification.
4. Train Logistic Regression and Random Forest pipelines.
5. Save models and evaluation metrics.
6. Explore sample records and run a single flow through the selected model.

## Responsible interpretation
- Synthetic data is useful for software demonstration, not evidence of real-world generalization.
- Do not claim the reported metrics reflect production detection accuracy.
- Destination port or one flow statistic alone does not establish maliciousness.
- To extend this project, use a suitable, lawfully obtained, documented flow dataset; perform leakage checks; compare against class baselines; tune thresholds; evaluate by time/source split where appropriate; and document false positives/false negatives.
- Avoid capturing or processing network traffic you are not authorized to inspect.

## Possible extensions
- Add a public, documented dataset such as a suitable CIC flow dataset (check its license and feature definitions)
- Compare models using cross-validation and calibration
- Add feature importance and drift monitoring
- Add batch prediction for uploaded flow CSVs
- Add a report export and analyst review queue
