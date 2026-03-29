from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def save_explainability_report(forecaster, prepared_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths = {}

    try:
        import shap  # type: ignore

        feature_frame = prepared_df[forecaster.feature_columns].tail(128)
        transformed = forecaster.scaler.transform(feature_frame)
        for target_name, model in forecaster.models.items():
            explainer = shap.LinearExplainer(model, transformed)
            shap_values = explainer.shap_values(transformed)
            mean_abs = np.abs(shap_values).mean(axis=0)
            summary = (
                pd.DataFrame(
                    {
                        "feature": forecaster.feature_columns,
                        "importance": mean_abs,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            save_path = output_dir / f"shap_{target_name}.csv"
            summary.to_csv(save_path, index=False)
            report_paths[target_name] = str(save_path)
        return report_paths
    except Exception:
        # Fallback: standardized linear contributions preserve interpretability even without SHAP installed.
        latest_scaled = forecaster.scaler.transform(prepared_df[forecaster.feature_columns].tail(1))[0]
        for target_name, model in forecaster.models.items():
            contributions = np.abs(model.coef_ * latest_scaled)
            summary = (
                pd.DataFrame(
                    {
                        "feature": forecaster.feature_columns,
                        "importance": contributions,
                    }
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            save_path = output_dir / f"shap_fallback_{target_name}.csv"
            summary.to_csv(save_path, index=False)
            report_paths[target_name] = str(save_path)
        return report_paths
