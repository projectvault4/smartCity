# Adaptive Hybrid Time-Series Intelligence Engine

Research-grade multivariate forecasting pipeline with:
- engineered lag, rolling, and calendar features
- attention-enhanced BiLSTM baseline
- encoder-only Transformer baseline with positional encoding
- parallel BiLSTM+Transformer hybrid model
- adaptive inverse-error ensemble with validation-trained meta-learner
- adaptive per-domain model switching based on recent normalized RMSE
- concept drift detection and online fine-tuning simulation
- explainable next-hour forecaster for smaller datasets with plain-language output and prediction ranges
- four-domain scoring across traffic, AQI, weather, and energy
- Pearson/Spearman correlation analysis and Granger causality tables
- SHAP-ready explainability reports with a transparent fallback when SHAP is unavailable

## Structure
- `data/`
- `models/`
- `engine/`
- `utils/`
- `train.py`
- `evaluate.py`
- `main.py`
- `generate_artifacts.py`

## Run
```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

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
- `http://127.0.0.1:8000/comparison` for model comparison across `BiLSTM`, `Transformer`, `Hybrid`, `AdaptiveSwitcher`, and `AdaptiveEnsemble`
