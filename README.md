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
- `backend/` Node.js Express API (users, risk, advisories, notifications, model forecast/anomaly/multivariate)
- `foresightx 2/` React + Vite citizen/operator dashboard (City Pulse, Voice Briefing, XAI, Risk, Reports)

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

## Live Web Dashboard (ForeSightX)

The React dashboard + Express backend read the trained artifacts under `outputs/`:

```bash
# 1. backend (API on :5001)
cd backend && cp .env.example .env && npm install && npm start

# 2. frontend (dev on :3000)
cd "foresightx 2" && npm install && npm run dev
```

Open `http://localhost:3000`. The dashboard renders:
- **City Pulse forecast** — 6-hour outlook re-anchored to the live server clock
- **2026 Outlook** — full-year prediction (monthly aggregates + charts + hourly timeline) generated
  from the trained hybrid model and the 2022–2025 history
- **AI Voice Briefing** — spoken forecast from model conditions
- **XAI panel**, **Ripple simulation**, **Risk advisory**, **Drift monitor**, **Reports**
- **Anomaly dashboard** and **multivariate coupling analysis** from `outputs/urban_*`

The model forecast service (`backend/src/services/modelForecast.service.js`) re-bases future-row
timestamps to `now + step_ahead` on every request when `MODEL_FORECAST_LIVE=true`, so the dashboard
always shows a rolling "today -> tomorrow" outlook regardless of when the training data ended.

## 2026 Forecast

The project trains on 2022–2025 data and generates a full-year 2026 prediction from the saved
hybrid-model forecaster (`outputs/checkpoints/explainable_forecaster.pkl`):

```bash
python3 generate_2026_forecast.py   # writes outputs/forecast_2026.csv
```

The script runs a damped recursive forecast: at every hourly step it rebuilds the engineered
lag/rolling/calendar features, scales them with the trained scaler, predicts the four targets with
the fitted models, and blends with the learned seasonal profile so the year-ahead outlook stays
stable. Served live by the React dashboard at `/api/model/forecast-yearly` (backend).

## Production Deployment (Docker)

```bash
cp backend/.env.production.example backend/.env.production
# fill in CLIENT_ORIGIN, DB passwords, API keys, and GROQ/GEMINI keys
docker compose up --build -d
```

Services:
- `web` — nginx serving the built React app on port 80, proxying `/api` to the app container
- `app` — Node backend on port 5000 (migrations run before start)
- `postgres` — PostgreSQL 16 with persistent volume
- `redis` — Redis 7 with append-only persistence

Build-time args for the frontend (`GROQ_API_KEY`, `GEMINI_API_KEY`, `VITE_API_BASE_URL=/api`)
come from your shell environment or `.env` in the compose directory. The backend image bakes the
small forecast/anomaly artifacts (`outputs/past_present_future_forecast.csv`,
`outputs/forecast_2026.csv`, `outputs/urban_events.json`, `outputs/urban_anomaly_timeline.csv`)
into the container, so the model API works out of the box. See `backend/README.md` for the full stack guide.
