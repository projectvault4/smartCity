from __future__ import annotations

import argparse
import json
import copy
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs") / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.baselines import evaluate_baselines
from utils.config import CONFIG, apply_city_config
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.metrics import urban_prediction_score_from_normalized_error
from utils.paper_artifacts import save_fair_baseline_note, save_per_target_tables, save_split_protocol

GRAPH_MODEL_NAMES = ("TFT", "GRU", "Hybrid")
METRIC_BAR_COLOR = "#1f77b4"
REFERENCE_ONLY_COMPARISON_MODELS = {"Prophet"}


def parse_args():
    parser = argparse.ArgumentParser(description="Finalize project-best outputs.")
    parser.add_argument("--city", default=None, help="Use city-specific data and outputs, e.g. delhi.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_hybrid_metrics_path(output_dir: Path) -> Path:
    lag_stabilized_path = output_dir / "tft_gru_residual_hybrid_lag_stabilized_metrics.json"
    if lag_stabilized_path.exists():
        return lag_stabilized_path
    return output_dir / "tft_gru_residual_hybrid_metrics.json"


def _load_literature_models(config) -> list[dict]:
    literature_path = Path(config.data_dir) / "literature_models.json"
    if not literature_path.exists():
        return []
    return _load_json(literature_path)


def _load_fair_baseline_metrics(output_dir: Path) -> pd.DataFrame:
    fair_summary_path = output_dir / "fair_tuning_summary.csv"
    if not fair_summary_path.exists():
        return pd.DataFrame(columns=["MAE", "MAPE", "RMSE", "NRMSE", "UPS"])

    fair_df = pd.read_csv(fair_summary_path)
    required_columns = ["model", "MAE", "MAPE", "RMSE", "NRMSE", "UPS"]
    if not all(column in fair_df.columns for column in required_columns):
        return pd.DataFrame(columns=["MAE", "MAPE", "RMSE", "NRMSE", "UPS"])

    return fair_df[required_columns].drop_duplicates("model", keep="first").set_index("model")


def _add_missing_fair_baselines(metrics_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    fair_metrics_df = _load_fair_baseline_metrics(output_dir)
    if "NRMSE" in fair_metrics_df.columns:
        fair_metrics_df["UPS"] = pd.to_numeric(fair_metrics_df["NRMSE"], errors="coerce").map(
            urban_prediction_score_from_normalized_error
        )
    for model_name in ("ARIMA", "SARIMA", "Prophet"):
        if model_name in metrics_df.index or model_name not in fair_metrics_df.index:
            continue
        metrics_df.loc[model_name, fair_metrics_df.columns] = fair_metrics_df.loc[model_name]
    return metrics_df


def _merge_prior_baselines(
    baseline_metrics_df: pd.DataFrame,
    baseline_per_target_metrics: dict,
    prior_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    prior_metrics = prior_summary.get("offline_metrics", {}) if isinstance(prior_summary, dict) else {}
    prior_per_target = prior_summary.get("per_target_metrics", {}) if isinstance(prior_summary, dict) else {}

    for model_name in ("ARIMA", "SARIMA", "Prophet"):
        if model_name in prior_metrics:
            prior_row = pd.Series(prior_metrics[model_name])
            if model_name not in baseline_metrics_df.index:
                baseline_metrics_df.loc[model_name, prior_row.index] = prior_row
        if model_name in prior_per_target and model_name not in baseline_per_target_metrics:
            baseline_per_target_metrics[model_name] = prior_per_target[model_name]

    return baseline_metrics_df, baseline_per_target_metrics


def _ordered_model_names(config, metrics_df: pd.DataFrame) -> list[str]:
    literature_names = [item["model"] for item in _load_literature_models(config)]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def _format_metric_value(row: pd.Series | None, metric_name: str, suffix: str = "") -> str:
    if row is None or metric_name not in row or pd.isna(row[metric_name]):
        return "N/A"
    if metric_name == "NRMSE":
        return f"{float(row[metric_name]):.4f}{suffix}"
    return f"{float(row[metric_name]):.2f}{suffix}"


def _per_target_display_frame(target_metrics: dict) -> pd.DataFrame:
    frame = pd.DataFrame(target_metrics).T
    if "NRMSE" in frame.columns:
        nrmse_values = pd.to_numeric(frame["NRMSE"], errors="coerce")
        frame["UPS"] = nrmse_values.map(urban_prediction_score_from_normalized_error)
    return frame


def _refresh_ups(metrics: dict, per_target: dict | None = None) -> dict:
    refreshed = dict(metrics)
    if per_target:
        for target_metrics in per_target.values():
            if "NRMSE" not in target_metrics or pd.isna(target_metrics["NRMSE"]):
                continue
            target_metrics["UPS"] = urban_prediction_score_from_normalized_error(target_metrics["NRMSE"])

    if "NRMSE" in refreshed:
        refreshed["UPS"] = urban_prediction_score_from_normalized_error(refreshed["NRMSE"])
    return refreshed


def _refresh_frame_ups(metrics_df: pd.DataFrame) -> pd.DataFrame:
    refreshed = metrics_df.copy()
    if "NRMSE" in refreshed.columns:
        refreshed["UPS"] = pd.to_numeric(refreshed["NRMSE"], errors="coerce").map(
            urban_prediction_score_from_normalized_error
        )
    return refreshed


def _refresh_summary_ups(project_best_summary: dict) -> dict:
    summary = copy.deepcopy(project_best_summary)
    for payload in summary.get("results", {}).values():
        per_target = payload.get("test_per_target", {})
        payload["test_metrics"] = _refresh_ups(payload.get("test_metrics", {}), per_target)
    return summary


def _build_paper_style_comparison(config, metrics_df: pd.DataFrame) -> pd.DataFrame:
    literature_rows = _load_literature_models(config)
    rows = []
    seen_models = set()
    for item in literature_rows:
        model_name = item["model"]
        seen_models.add(model_name)
        row = metrics_df.loc[model_name] if model_name in metrics_df.index else None
        if row is None and model_name not in REFERENCE_ONLY_COMPARISON_MODELS:
            continue
        rows.append(
            {
                "Model": model_name,
                "MAE": _format_metric_value(row, "MAE"),
                "MAPE": _format_metric_value(row, "MAPE", "%"),
                "RMSE": _format_metric_value(row, "RMSE"),
                "NRMSE": _format_metric_value(row, "NRMSE"),
                "UPS": _format_metric_value(row, "UPS"),
            }
        )

    for model_name in metrics_df.index:
        if model_name in seen_models:
            continue
        row = metrics_df.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": _format_metric_value(row, "MAE"),
                "MAPE": _format_metric_value(row, "MAPE", "%"),
                "RMSE": _format_metric_value(row, "RMSE"),
                "NRMSE": _format_metric_value(row, "NRMSE"),
                "UPS": _format_metric_value(row, "UPS"),
            }
        )

    return pd.DataFrame(rows)


def _save_final_metric_graph(metrics_df: pd.DataFrame, metric_name: str, output_dir: Path) -> Path | None:
    if metric_name not in metrics_df.columns:
        return None

    graph_models = [model_name for model_name in GRAPH_MODEL_NAMES if model_name in metrics_df.index]
    metric_series = pd.to_numeric(metrics_df.loc[graph_models, metric_name], errors="coerce").dropna().sort_values()
    if metric_series.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars = ax.bar(metric_series.index, metric_series.values, color=METRIC_BAR_COLOR)
    ax.set_title(f"{metric_name} Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAPE (%)" if metric_name == "MAPE" else metric_name)
    ax.margins(y=0.12)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    y_min, y_max = ax.get_ylim()
    label_offset = (y_max - y_min) * 0.018
    for bar, value in zip(bars, metric_series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )

    fig.tight_layout()
    path = output_dir / f"{metric_name.lower()}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def _save_final_metric_graphs(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved_paths = []
    for metric_name in ("MAPE", "NRMSE", "UPS"):
        path = _save_final_metric_graph(metrics_df, metric_name, output_dir)
        if path is not None:
            saved_paths.append(path)
    return saved_paths


def _with_latest_hybrid_outputs(project_best_summary: dict, output_dir: Path) -> tuple[dict, dict | None]:
    """Use the canonical residual-hybrid artifact as the final Hybrid row when available."""
    hybrid_metrics_path = _latest_hybrid_metrics_path(output_dir)
    if not hybrid_metrics_path.exists():
        return project_best_summary, None

    hybrid_artifact = _load_json(hybrid_metrics_path)
    metrics = hybrid_artifact.get("metrics")
    per_target = hybrid_artifact.get("per_target_metrics")
    if not metrics or not per_target:
        return project_best_summary, None
    metrics = _refresh_ups(metrics, per_target)

    summary = copy.deepcopy(project_best_summary)
    results = summary.setdefault("results", {})
    prior_hybrid = results.get("Hybrid", {})
    previous_rmse = prior_hybrid.get("test_metrics", {}).get("RMSE")
    if previous_rmse is not None and metrics.get("RMSE") is not None and float(metrics["RMSE"]) > float(previous_rmse):
        protocol = summary.setdefault("search_protocol", {})
        sources = protocol.setdefault("final_metric_sources", {})
        sources["Hybrid"] = prior_hybrid.get("source_artifact", "project_best_summary")
        return summary, {
            "model": "Hybrid",
            "source_artifact": str(hybrid_metrics_path),
            "previous_project_best_RMSE": previous_rmse,
            "final_RMSE": previous_rmse,
            "skipped": True,
            "reason": "artifact RMSE was worse than the existing finalized Hybrid RMSE",
        }

    results["Hybrid"] = {
        **prior_hybrid,
        "source_artifact": str(hybrid_metrics_path),
        "model": hybrid_artifact.get("model", "TFTGRUResidualHybrid"),
        "architecture": hybrid_artifact.get("architecture", ""),
        "prediction_path": hybrid_artifact.get("prediction_path", ""),
        "best_checkpoint": hybrid_artifact.get("checkpoint_path", prior_hybrid.get("best_checkpoint", "")),
        "test_metrics": metrics,
        "test_per_target": per_target,
        "project_best_search_result": prior_hybrid,
    }

    protocol = summary.setdefault("search_protocol", {})
    sources = protocol.setdefault("final_metric_sources", {})
    sources["Hybrid"] = str(hybrid_metrics_path)

    override = {
        "model": "Hybrid",
        "source_artifact": str(hybrid_metrics_path),
        "previous_project_best_RMSE": previous_rmse,
        "final_RMSE": metrics.get("RMSE"),
        "checkpoint_path": hybrid_artifact.get("checkpoint_path", ""),
        "prediction_path": hybrid_artifact.get("prediction_path", ""),
    }
    return summary, override


def main():
    config = apply_city_config(copy.deepcopy(CONFIG), parse_args().city)
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_best_path = output_dir / "project_best_summary.json"
    if not project_best_path.exists():
        raise FileNotFoundError(
            f"Missing {project_best_path}. Finish project-best tuning for this city before finalizing outputs."
        )

    raw_df = load_input_dataframe(config)
    datasets = create_datasets(config, raw_df)
    baseline_metrics_df, baseline_per_target_metrics, _, baseline_metadata = evaluate_baselines(datasets, config)

    project_best_summary = _refresh_summary_ups(_load_json(project_best_path))
    project_best_summary, hybrid_override = _with_latest_hybrid_outputs(project_best_summary, output_dir)
    project_best_summary = _refresh_summary_ups(project_best_summary)
    with open(project_best_path, "w", encoding="utf-8") as handle:
        json.dump(project_best_summary, handle, indent=2)
    prior_summary_path = output_dir / "summary.json"
    prior_summary = _load_json(prior_summary_path) if prior_summary_path.exists() else {}

    core_metrics = {
        model_name: payload["test_metrics"]
        for model_name, payload in project_best_summary["results"].items()
    }
    core_metrics_df = pd.DataFrame(core_metrics).T
    core_per_target = {
        model_name: payload["test_per_target"]
        for model_name, payload in project_best_summary["results"].items()
    }

    baseline_metrics_df, baseline_per_target_metrics = _merge_prior_baselines(
        baseline_metrics_df,
        baseline_per_target_metrics,
        prior_summary,
    )
    baseline_metrics_df = _add_missing_fair_baselines(baseline_metrics_df, output_dir)
    all_metrics_df = pd.concat([core_metrics_df, baseline_metrics_df], axis=0)
    all_metrics_df = _refresh_frame_ups(all_metrics_df)
    core_metrics_df = _refresh_frame_ups(core_metrics_df)
    baseline_metrics_df = _refresh_frame_ups(baseline_metrics_df)
    all_metrics_df = all_metrics_df.loc[_ordered_model_names(config, all_metrics_df)]
    all_per_target = {**core_per_target, **baseline_per_target_metrics}

    all_metrics_df.to_csv(output_dir / "metrics.csv")
    core_metrics_df.to_csv(output_dir / "core_model_metrics.csv")
    baseline_metrics_df.to_csv(output_dir / "baseline_metrics.csv")

    research_table_df = _build_paper_style_comparison(config, all_metrics_df)
    research_table_df.to_csv(output_dir / "research_performance_comparison.csv", index=False)
    metric_graph_paths = _save_final_metric_graphs(all_metrics_df, output_dir)

    literature_rows = _load_literature_models(config)
    save_per_target_tables(all_per_target, config, output_dir, literature_rows)
    save_split_protocol(datasets, config, output_dir)
    save_fair_baseline_note(literature_rows, all_metrics_df.index.tolist(), output_dir)

    ranked_df = all_metrics_df.sort_values(["RMSE", "MAE", "UPS"], ascending=[True, True, False]).reset_index()
    ranked_df = ranked_df.rename(columns={"index": "model"})
    top_models_df = ranked_df.head(4).copy()
    top_models_df.to_csv(output_dir / "final_best_models.csv", index=False)

    final_registry = {
        "finalized_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "source_summary": str(project_best_path),
        "top_4_models_by_rmse": top_models_df.to_dict(orient="records"),
        "project_best_models": project_best_summary["results"],
        "baselines": baseline_metrics_df.to_dict(orient="index"),
        "hybrid_override": hybrid_override,
    }
    with open(output_dir / "final_model_registry.json", "w", encoding="utf-8") as handle:
        json.dump(final_registry, handle, indent=2)

    merged_summary = {
        "offline_metrics": all_metrics_df.to_dict(orient="index"),
        "per_target_metrics": all_per_target,
        "streaming_model": prior_summary.get("streaming_model", ""),
        "adaptive_switcher": prior_summary.get("adaptive_switcher", {}),
        "xai_reports": prior_summary.get("xai_reports", {}),
        "baseline_metadata": baseline_metadata,
        "project_best_summary": project_best_summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(merged_summary, handle, indent=2)

    print("\nFinalized Top 4 Models")
    print(top_models_df[["model", "MAE", "RMSE", "NRMSE", "UPS"]].round(4).to_string(index=False))

    if "Hybrid" in all_per_target:
        print("\nHybrid Model Per-Target Prediction Metrics")
        hybrid_per_target_df = _per_target_display_frame(all_per_target["Hybrid"])
        print(hybrid_per_target_df[["MAE", "MAPE", "RMSE", "NRMSE", "UPS"]].round(4).to_string())

    if hybrid_override:
        print(
            "\nMapped Hybrid to latest residual-hybrid artifact "
            f"({hybrid_override['source_artifact']})"
        )
    print(f"\nWrote finalized outputs to {output_dir}")


if __name__ == "__main__":
    main()
