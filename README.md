# Adaptive BiLSTM-TFT Time-Series Intelligence Engine

Research-grade multivariate forecasting pipeline with:
- engineered lag, rolling, and calendar features
- attention-enhanced BiLSTM baseline
- Temporal Fusion Transformer baseline
- adaptive BiLSTM+TFT hybrid model with target-wise internal gating
- adaptive per-target model switcher based on recent normalized RMSE
- concept drift detection and online fine-tuning simulation
- intelligent anomaly detection for urban event discovery with Isolation Forest and autoencoder scoring
- explainable next-hour forecaster for smaller datasets with plain-language output and prediction ranges
- four-domain scoring across traffic, AQI, weather, and energy
- Pearson/Spearman correlation analysis and Granger causality tables
- SHAP-ready explainability reports with a transparent fallback when SHAP is unavailable

## Structure
- `data/`
- `models/`
- `engine/`
- `utils/`
- `docs/intelligent_anomaly_detection.md`
- `train.py`
- `evaluate.py`
- `main.py`
- `generate_artifacts.py`

## Run
```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

If `Banglore_traffic_Dataset_raw.csv` and `AirQualityUCI.csv` are present in the project root, the loader now auto-builds `data/urban_multivariate_timeseries.csv` from them before training.
The merged file keeps the expected schema:
`timestamp`, `traffic_flow`, `aqi`, `electricity_demand`, `temperature`, `humidity`.

If you already have checkpoints and only want metrics/plots:
```bash
python3 generate_artifacts.py
```

For a simpler and more realistic next-hour forecast on small datasets:
```bash
python3 predict_next_hour.py
```

To view the forecast in a browser:
```bash
python3 app.py
```

Then open `http://127.0.0.1:8000`.

Dashboard pages:
- `http://127.0.0.1:8000/` for the live forecast, analytics, and explainable model metrics
- `http://127.0.0.1:8000/comparison` for model comparison across `BiLSTM`, `TFT`, `Hybrid`, and `AdaptiveSwitcher`
- `http://127.0.0.1:8000/anomalies` for the dark AI anomaly dashboard

Urban event discovery:
- `http://127.0.0.1:8000/api/anomalies` returns normalized Hybrid Anomaly Scores, event severity, SHAP-style drivers, heatmap data, chart series, insights, and recommendations
- `http://127.0.0.1:8000/api/anomalies/export.csv` exports detected anomaly events
- `http://127.0.0.1:8000/api/anomalies/export.pdf` exports a compact anomaly report with charts
- detected event artifacts are saved as `urban_anomaly_timeline.csv` and `urban_events.json` under the active output directory
- the anomaly module uses uploaded historical CSV data only; it does not require live sensors or real-time streaming data
- the full objective, formulas, architecture, pseudocode, evaluation plan, streaming deployment, and future scope are documented in `docs/intelligent_anomaly_detection.md`
