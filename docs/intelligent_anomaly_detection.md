# Intelligent Anomaly Detection for Urban Event Discovery

## Detailed Objective

The objective of this module is to extend ForeSightX with an intelligent anomaly detection and urban event discovery layer that continuously identifies unusual, high-impact changes in multivariate smart-city time-series data. Instead of only forecasting traffic flow, AQI, temperature, humidity, and electricity demand, the proposed system learns normal spatio-temporal behavior across these domains and detects deviations that may represent real urban events. The framework combines statistical feature engineering, Isolation Forest, autoencoder-based reconstruction modeling, and an optional LSTM autoencoder for sequence-aware anomaly detection. It analyzes rolling trends, lagged behavior, time-of-day effects, and cross-domain interactions so that sudden congestion, accidents, AQI spikes, abnormal power demand, sensor faults, public gatherings, and weather disruptions can be discovered early.

The module converts raw outlier scores into explainable urban events by clustering anomalies over time, identifying dominant contributing features, assigning event categories, and ranking severity as Critical, High, Medium, or Low. It also integrates anomaly scores with the existing TFT, GRU, and SARIMA forecasting stack by comparing anomalies with model residuals, widening uncertainty intervals during abnormal conditions, and flagging periods where forecasts alone are not enough for decision-making. Explainable AI methods such as SHAP, or the existing standardized fallback contribution reports, justify why an event was detected. The final goal is to transform ForeSightX from a predictive forecasting system into an intelligent decision-support platform that discovers emerging urban incidents, explains them, visualizes them on dashboards, and supports faster operational response.

## Problem Statement

Smart cities generate continuous multivariate data from traffic sensors, air-quality monitors, weather stations, and energy infrastructure. Forecasting future values is useful, but city operators also need to know when the present behavior is abnormal. A traffic spike may indicate an accident, an AQI jump may indicate a pollution episode, and a sudden electricity surge may signal unexpected public activity or grid stress. Traditional threshold rules fail because urban data is seasonal, noisy, and strongly dependent on hour, weekday, weather, and cross-domain effects. AI-based anomaly detection learns normal behavior and highlights unusual deviations even when fixed thresholds are not obvious.

## Pipeline Architecture

1. Data ingestion from historical CSV files or real-time Kafka/MQTT streams.
2. Preprocessing with timestamp alignment, interpolation, missing-value repair, duplicate removal, normalization, and clipping.
3. Feature engineering with rolling statistics, lag features, percentage changes, temporal encodings, and cross-domain indicators.
4. Feature selection using variance filtering and correlation pruning.
5. Model layer with Isolation Forest, dense autoencoder, and optional LSTM autoencoder.
6. Score fusion into a unified anomaly score.
7. Event conversion using temporal grouping, dominant-feature explanation, event typing, and severity classification.
8. Integration with TFT, GRU, and SARIMA residuals.
9. Explainability using SHAP or fallback standardized feature contributions.
10. Dashboard visualization using React with Leaflet or Mapbox markers, heatmaps, timelines, and alerts.

## Preprocessing and Feature Engineering

Missing values are handled using timestamp-aware interpolation followed by forward/backward filling. Numerical columns are normalized with StandardScaler or MinMaxScaler depending on the model. Feature engineering includes lag features such as `x_{t-1}`, `x_{t-6}`, and `x_{t-24}`, rolling mean/std/min/max over 6, 12, and 24 hour windows, rolling z-scores, first differences, percentage changes, hour/day/weekend encodings, and cyclic time features:

`hour_sin = sin(2*pi*hour/24)`, `hour_cos = cos(2*pi*hour/24)`.

Feature selection removes near-constant features, then prunes highly correlated features so the anomaly models remain stable and efficient.

## Isolation Forest

Isolation Forest isolates anomalies by recursively selecting random features and random split values. Anomalies are easier to isolate, so they require shorter path lengths in random trees. For sample `x`, the anomaly score is:

`s(x, n) = 2^(-E(h(x)) / c(n))`

where `E(h(x))` is the expected path length and `c(n)` is the average path length of unsuccessful search in a binary tree:

`c(n) = 2H(n - 1) - 2(n - 1)/n`

Higher scores indicate more anomalous samples. Isolation Forest is suitable for urban datasets because it handles high-dimensional engineered features, does not require labeled anomalies, and can detect rare combinations such as high traffic plus abnormal AQI plus unusual electricity demand.

## Autoencoder Anomaly Detection

An autoencoder learns to reconstruct normal input vectors. Given input `x`, encoder `f_theta` maps it into latent vector `z`, and decoder `g_phi` reconstructs it:

`z = f_theta(x)`, `x_hat = g_phi(z)`

The reconstruction error is:

`RE(x) = (1/d) * sum_i (x_i - x_hat_i)^2`

An anomaly score can be normalized as:

`A_auto(x) = (RE(x) - min(RE)) / (max(RE) - min(RE))`

The threshold is selected using a robust percentile/MAD rule:

`T = max(P95(RE), median(RE) + 3 * 1.4826 * MAD(RE))`

Samples above the threshold are marked anomalous. Reconstruction error is useful because the model reconstructs normal seasonal urban behavior well, but struggles with unseen disruptions.

## Optional LSTM Autoencoder

An LSTM autoencoder models sequences instead of independent feature rows. It receives a window `X_t = [x_{t-k}, ..., x_t]`, encodes temporal context, and reconstructs the sequence. This is better for detecting gradual temporal disruptions and delayed effects. Isolation Forest is faster and strong for point anomalies; LSTM autoencoders are stronger for sequential anomalies but require more training time and careful tuning.

## Event Types

The framework detects:

- Sudden traffic congestion
- Road accidents
- AQI spikes
- Unexpected electricity demand
- Sensor malfunction
- Public event detection
- Weather-related disruptions

## Outlier-to-Event Conversion

ForeSightX does not stop at outlier labels. Consecutive anomalous timestamps are grouped into candidate events. The system identifies dominant features using z-score or SHAP-style contributions, assigns a likely event type, attaches location metadata, calculates duration and peak anomaly score, and ranks operational severity. This creates actionable urban events instead of isolated statistical warnings.

## Severity Classification

Severity is based on fused anomaly score, affected domains, and persistence:

- Critical: score >= 0.90 or multi-domain disruption
- High: score >= 0.75
- Medium: score >= 0.58
- Low: score below Medium but still above threshold

## Integration With TFT, GRU, and SARIMA

Anomaly scores are integrated with existing forecasting models by comparing observed values with TFT, GRU, and SARIMA predictions. If residuals and anomaly scores are high, the system treats the point as a real urban event. If forecasts expected the increase, the event severity is reduced. During anomalies, ForeSightX can widen prediction intervals, down-weight stale models, trigger retraining checks, and add event context to next-hour forecasts.

## Explainable AI

SHAP can explain anomaly detection by estimating feature contributions to anomaly probability or reconstruction error. When SHAP is unavailable, ForeSightX uses standardized feature contribution fallback reports. Example explanation: traffic flow rolling z-score, AQI 24-hour deviation, and electricity demand lag may jointly explain why an event was detected.

## Dashboard Visualization

The dashboard can be implemented in React with Leaflet or Mapbox. It should show severity-colored anomaly markers, city heatmaps, time-series anomaly score timelines, threshold overlays, event cards, and alert notifications. Critical and High events should appear first, with driver features and timestamps shown for quick operator interpretation.

## Pseudocode

```text
Input: multivariate urban time-series D
Clean missing values and duplicate timestamps
Generate lag, rolling, difference, percentage-change, and time-encoding features
Select stable non-redundant features
Fit Isolation Forest on normalized features
Compute isolation anomaly score
Fit autoencoder to reconstruct normalized features
Compute reconstruction error
Normalize both scores
combined_score = 0.55 * isolation_score + 0.45 * autoencoder_score
threshold = max(P95(combined_score), median + 3 * 1.4826 * MAD)
For each timestamp above threshold:
    find top contributing features
    infer event type
    assign severity
    attach timestamp and map coordinates
Return event list, scores, explanations, and dashboard artifacts
```

## Evaluation Metrics

Use Precision, Recall, F1-score, ROC-AUC, PR-AUC, False Positive Rate, Detection Delay, and Mean Time to Detect. If labels are unavailable, use expert validation, synthetic anomaly injection, residual agreement with forecasting models, and alert stability analysis.

## Real-Time Deployment

In streaming mode, Kafka or MQTT receives sensor events. A stream processor maintains rolling windows, updates lag features, normalizes values using saved scalers, computes anomaly scores, and publishes detected events to an `urban-events` topic. The dashboard subscribes through a backend API or WebSocket. Critical events can be sent to notification services.

## Limitations and Future Enhancements

Limitations include scarce labeled anomaly data, false positives during festivals or planned maintenance, sensor drift, and difficulty separating correlated causes. Future enhancements include Graph Neural Networks for road/sensor networks, Transformer-based anomaly detection for long-range temporal patterns, federated learning across cities without sharing raw data, and active learning from operator feedback.

## Impact on ForeSightX

This module transforms ForeSightX from a forecasting-only system into an intelligent urban event discovery and decision-support platform. Forecasting answers what may happen next; anomaly detection explains what unusual event is happening now, why it matters, where it is happening, and how severe it is.
