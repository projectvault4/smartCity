from __future__ import annotations

import pandas as pd

from engine.adaptive_ensemble import AdaptiveDomainSwitcher, AdaptiveEnsemble
from utils.metrics import compute_all_metrics, compute_metrics_by_target, compute_urban_prediction_score
from utils.training import predict_model


def evaluate_models(models, datasets, config):
    processor = datasets["processor"]
    x_val, y_val_scaled = datasets["val_seq"]
    x_test, y_test_scaled = datasets["test_seq"]
    y_val = processor.inverse_transform_targets(y_val_scaled)
    y_true = processor.inverse_transform_targets(y_test_scaled)

    val_predictions = {}
    test_predictions = {}
    feature_weights = {}
    for name, model in models.items():
        val_scaled = predict_model(model, x_val, config)
        test_scaled = predict_model(model, x_test, config)
        val_predictions[name] = processor.inverse_transform_targets(val_scaled)
        test_predictions[name] = processor.inverse_transform_targets(test_scaled)
        feature_weights[name] = getattr(model, "latest_feature_weights", None)

    ensemble = AdaptiveEnsemble(model_names=list(test_predictions.keys()), error_window=config.ensemble_error_window)
    switcher = AdaptiveDomainSwitcher(
        model_names=list(test_predictions.keys()),
        target_names=config.target_columns,
        switch_window=config.adaptive_switch_window,
    )
    ensemble.update_errors(y_val, val_predictions)
    ensemble.fit_meta_learner(y_val, val_predictions)
    switcher.update(y_val, val_predictions)
    metrics = {name: compute_all_metrics(y_true, pred) for name, pred in test_predictions.items()}
    ensemble_pred = ensemble.predict(test_predictions)
    switched_pred = switcher.predict(test_predictions)
    metrics["AdaptiveEnsemble"] = compute_all_metrics(y_true, ensemble_pred)
    metrics["AdaptiveSwitcher"] = compute_all_metrics(y_true, switched_pred)
    test_predictions["AdaptiveEnsemble"] = ensemble_pred
    test_predictions["AdaptiveSwitcher"] = switched_pred

    metrics_df = pd.DataFrame(metrics).T
    metrics_df["UPS"] = [
        compute_urban_prediction_score(y_true, test_predictions[name], config.target_columns)
        for name in metrics_df.index
    ]
    per_target_metrics = {
        name: compute_metrics_by_target(y_true, pred, config.target_columns)
        for name, pred in test_predictions.items()
    }
    return metrics_df, per_target_metrics, test_predictions, feature_weights, ensemble.weights, switcher.selected_models
