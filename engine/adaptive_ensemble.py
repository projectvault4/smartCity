from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.linear_model import Ridge

from utils.metrics import compute_all_metrics, nrmse


class AdaptiveEnsemble:
    def __init__(self, model_names, error_window: int = 72, epsilon: float = 1e-6, use_meta_learner: bool = True):
        self.model_names = model_names
        self.error_window = error_window
        self.epsilon = epsilon
        self.use_meta_learner = use_meta_learner
        self.error_history = {name: deque(maxlen=error_window) for name in model_names}
        self.weights = {name: 1.0 / len(model_names) for name in model_names}
        self.meta_learner = Ridge(alpha=1.0, positive=True) if use_meta_learner else None
        self.meta_is_fitted = False

    def update_errors(self, y_true, predictions):
        for name, pred in predictions.items():
            errors = np.abs(np.asarray(y_true).reshape(-1) - np.asarray(pred).reshape(-1))
            for err in errors:
                self.error_history[name].append(float(err))
        self._refresh_weights()

    def _refresh_weights(self):
        inverse_errors = {}
        for name in self.model_names:
            if len(self.error_history[name]) == 0:
                inverse_errors[name] = 1.0
            else:
                mean_error = np.mean(self.error_history[name])
                inverse_errors[name] = 1.0 / max(mean_error, self.epsilon)

        total = sum(inverse_errors.values())
        self.weights = {name: value / total for name, value in inverse_errors.items()}

    def _prediction_matrix(self, predictions):
        return np.concatenate([np.asarray(predictions[name]) for name in self.model_names], axis=1)

    def fit_meta_learner(self, y_true, predictions):
        if not self.use_meta_learner:
            return
        prediction_matrix = self._prediction_matrix(predictions)
        weighted_prediction = self.combine(predictions)
        meta_features = np.concatenate([prediction_matrix, weighted_prediction], axis=1)
        self.meta_learner.fit(meta_features, np.asarray(y_true))
        self.meta_is_fitted = True

    def combine(self, predictions):
        combined = np.zeros_like(next(iter(predictions.values())), dtype=np.float64)
        for name in self.model_names:
            combined += self.weights[name] * np.asarray(predictions[name])
        return combined

    def predict(self, predictions):
        weighted_prediction = self.combine(predictions)
        if not self.meta_is_fitted:
            return weighted_prediction
        prediction_matrix = self._prediction_matrix(predictions)
        meta_features = np.concatenate([prediction_matrix, weighted_prediction], axis=1)
        return self.meta_learner.predict(meta_features)

    def evaluate(self, y_true, predictions):
        ensemble_pred = self.predict(predictions)
        metrics = {name: compute_all_metrics(y_true, pred) for name, pred in predictions.items()}
        metrics["AdaptiveEnsemble"] = compute_all_metrics(y_true, ensemble_pred)
        return metrics, ensemble_pred


class DriftDetector:
    def __init__(self, error_window: int, threshold: float):
        self.error_window = error_window
        self.threshold = threshold
        self.recent_errors = deque(maxlen=error_window)
        self.reference_errors = deque(maxlen=error_window)

    def update(self, error_value: float) -> bool:
        self.recent_errors.append(float(error_value))
        if len(self.reference_errors) < self.error_window:
            self.reference_errors.append(float(error_value))
            return False
        baseline = np.mean(self.reference_errors)
        current = np.mean(self.recent_errors)
        drift_detected = current > baseline * self.threshold
        if not drift_detected:
            self.reference_errors.append(float(error_value))
        return drift_detected


class AdaptiveDomainSwitcher:
    def __init__(self, model_names, target_names, switch_window: int = 48):
        self.model_names = list(model_names)
        self.target_names = list(target_names)
        self.switch_window = switch_window
        self.selected_models = {target: self.model_names[0] for target in self.target_names}
        self.domain_scores = {
            target: {name: deque(maxlen=switch_window) for name in self.model_names}
            for target in self.target_names
        }

    def update(self, y_true, predictions):
        y_true = np.asarray(y_true)
        for target_idx, target_name in enumerate(self.target_names):
            true_values = y_true[:, target_idx]
            for model_name in self.model_names:
                pred_values = np.asarray(predictions[model_name])[:, target_idx]
                score = nrmse(true_values, pred_values)
                self.domain_scores[target_name][model_name].append(float(score))

            mean_scores = {
                model_name: float(np.mean(scores)) if scores else float("inf")
                for model_name, scores in self.domain_scores[target_name].items()
            }
            self.selected_models[target_name] = min(mean_scores, key=mean_scores.get)

    def predict(self, predictions):
        stacked_predictions = np.zeros_like(next(iter(predictions.values())), dtype=float)
        for target_idx, target_name in enumerate(self.target_names):
            model_name = self.selected_models[target_name]
            stacked_predictions[:, target_idx] = np.asarray(predictions[model_name])[:, target_idx]
        return stacked_predictions
