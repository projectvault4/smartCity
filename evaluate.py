from __future__ import annotations

import pandas as pd

from engine.adaptive_ensemble import AdaptiveDomainSwitcher
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model


def evaluate_models(models, datasets, config):
    processor = datasets["processor"]
    val_groups = datasets["val_tpt"]
    test_groups = datasets["test_tpt"]
    x_val, y_val_scaled = ({key: val_groups[key] for key in ("closeness", "period", "trend")}, val_groups["target"])
    x_test, y_test_scaled = ({key: test_groups[key] for key in ("closeness", "period", "trend")}, test_groups["target"])
    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_true = processor.inverse_transform_targets(y_test_scaled)

    val_predictions = {}
    predictions = {}
    feature_weights = {}
    for name, model in models.items():
        val_scaled = predict_model(model, x_val, config)
        test_scaled = predict_model(model, x_test, config)
        val_predictions[name] = processor.inverse_transform_targets(val_scaled)
        predictions[name] = processor.inverse_transform_targets(test_scaled)
        feature_weights[name] = getattr(model, "latest_feature_weights", None)

    switcher = AdaptiveDomainSwitcher(
        model_names=list(predictions.keys()),
        target_names=config.target_columns,
        switch_window=config.adaptive_switch_window,
    )
    switcher.update(y_val, val_predictions)
    switched_predictions = switcher.predict(predictions)
    predictions["AdaptiveSwitcher"] = switched_predictions

    metrics = {name: compute_all_metrics(y_true, pred) for name, pred in predictions.items()}
    metrics_df = pd.DataFrame(metrics).T
    metrics_df["UPS"] = [
        compute_urban_prediction_score(y_true, predictions[name], config.target_columns)
        for name in metrics_df.index
    ]
    per_target_metrics = {
        name: compute_metrics_by_target(y_true, pred, config.target_columns)
        for name, pred in predictions.items()
    }
    return metrics_df, per_target_metrics, predictions, feature_weights, switcher.selected_models
