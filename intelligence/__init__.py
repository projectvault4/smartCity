"""ForeSightX decision-intelligence layer.

Offline analyses that run on the deployed TFT+GRU hybrid (see utils/hybrid_runtime.py)
and write JSON artifacts consumed by the backend and dashboard:

- benchmark   : skill vs naive / seasonal-naive / SARIMA with rolling windows and DM tests
- conformal   : adaptive conformal prediction intervals and exceedance probabilities
- explain     : SHAP (GradientExplainer) attributions on the real hybrid network
- whatif      : counterfactual scenarios re-run through the full prediction chain
- anomaly     : leakage-free anomaly scores evaluated against a z-score baseline
- replay      : one alert pipeline replayed over the held-out window, with a scorecard
"""

from pathlib import Path
import json

POLICY_PATH = Path(__file__).with_name("policy.json")


def load_policy() -> dict:
    with open(POLICY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)
