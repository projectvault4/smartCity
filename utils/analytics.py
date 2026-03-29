from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


def compute_correlation_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "pearson": numeric_df.corr(method="pearson"),
        "spearman": numeric_df.corr(method="spearman"),
    }


def _lagged_matrix(series: np.ndarray, max_lag: int) -> np.ndarray:
    rows = []
    for idx in range(max_lag, len(series)):
        rows.append([series[idx - lag] for lag in range(1, max_lag + 1)])
    return np.asarray(rows, dtype=float)


def granger_causality_table(df: pd.DataFrame, columns: List[str], max_lag: int = 6) -> pd.DataFrame:
    records = []
    for cause in columns:
        for effect in columns:
            if cause == effect:
                continue
            series_x = df[cause].to_numpy(dtype=float)
            series_y = df[effect].to_numpy(dtype=float)
            best_p = 1.0
            best_lag = None

            for lag in range(1, max_lag + 1):
                if len(series_x) <= lag + 5:
                    continue

                y_target = series_y[lag:]
                y_lags = _lagged_matrix(series_y, lag)
                x_lags = _lagged_matrix(series_x, lag)
                n_obs = min(len(y_target), len(y_lags), len(x_lags))
                y_target = y_target[-n_obs:]
                y_lags = y_lags[-n_obs:]
                x_lags = x_lags[-n_obs:]

                x_restricted = np.column_stack([np.ones(n_obs), y_lags])
                x_full = np.column_stack([np.ones(n_obs), y_lags, x_lags])

                beta_restricted, *_ = np.linalg.lstsq(x_restricted, y_target, rcond=None)
                beta_full, *_ = np.linalg.lstsq(x_full, y_target, rcond=None)
                rss_restricted = float(np.sum((y_target - x_restricted @ beta_restricted) ** 2))
                rss_full = float(np.sum((y_target - x_full @ beta_full) ** 2))

                df_num = lag
                df_den = n_obs - x_full.shape[1]
                if df_den <= 0 or rss_full <= 1e-12:
                    continue

                f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_den)
                f_stat = max(f_stat, 0.0)
                p_value = float(f_dist.sf(f_stat, df_num, df_den))

                if p_value < best_p:
                    best_p = p_value
                    best_lag = lag

            records.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "best_lag": best_lag,
                    "p_value": best_p,
                    "significant_5pct": bool(best_p < 0.05),
                }
            )

    return pd.DataFrame(records).sort_values(["p_value", "cause", "effect"]).reset_index(drop=True)


def save_analysis_tables(correlations: Dict[str, pd.DataFrame], granger_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, corr_df in correlations.items():
        corr_df.to_csv(output_dir / f"{name}_correlation_matrix.csv")
    granger_df.to_csv(output_dir / "granger_causality.csv", index=False)
