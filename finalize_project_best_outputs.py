from __future__ import annotations

import json
import copy
from pathlib import Path

import pandas as pd

from utils.baselines import evaluate_baselines
from utils.config import CONFIG
from utils.data_utils import create_datasets, load_input_dataframe, set_seed
from utils.paper_artifacts import save_fair_baseline_note, save_per_target_tables, save_split_protocol


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_literature_models(config) -> list[dict]:
    literature_path = Path(config.data_dir) / "literature_models.json"
    if not literature_path.exists():
        return []
    return _load_json(literature_path)


def _ordered_model_names(config, metrics_df: pd.DataFrame) -> list[str]:
    literature_names = [item["model"] for item in _load_literature_models(config)]
    present_names = list(metrics_df.index)
    ordered = [name for name in literature_names if name in present_names]
    ordered.extend(name for name in present_names if name not in ordered)
    return ordered


def _build_paper_style_comparison(config, metrics_df: pd.DataFrame) -> pd.DataFrame:
    literature_rows = _load_literature_models(config)
    rows = []
    seen_models = set()
    for item in literature_rows:
        model_name = item["model"]
        seen_models.add(model_name)
        if model_name not in metrics_df.index:
            continue

        row = metrics_df.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{float(row['MAE']):.2f}",
                "MAPE": f"{float(row['MAPE']):.2f}%",
                "RMSE": f"{float(row['RMSE']):.2f}",
            }
        )

    for model_name in metrics_df.index:
        if model_name in seen_models:
            continue
        row = metrics_df.loc[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": f"{float(row['MAE']):.2f}",
                "MAPE": f"{float(row['MAPE']):.2f}%",
                "RMSE": f"{float(row['RMSE']):.2f}",
            }
        )

    return pd.DataFrame(rows)


def _with_latest_hybrid_outputs(project_best_summary: dict, output_dir: Path) -> tuple[dict, dict | None]:
    """Use the canonical residual-hybrid artifact as the final Hybrid row when available."""
    hybrid_metrics_path = output_dir / "tft_gru_residual_hybrid_metrics.json"
    if not hybrid_metrics_path.exists():
        return project_best_summary, None

    hybrid_artifact = _load_json(hybrid_metrics_path)
    metrics = hybrid_artifact.get("metrics")
    per_target = hybrid_artifact.get("per_target_metrics")
    if not metrics or not per_target:
        return project_best_summary, None

    summary = copy.deepcopy(project_best_summary)
    results = summary.setdefault("results", {})
    prior_hybrid = results.get("Hybrid", {})
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

    previous_rmse = prior_hybrid.get("test_metrics", {}).get("RMSE")
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
    config = CONFIG
    set_seed(config.random_seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_best_path = output_dir / "project_best_summary.json"
    if not project_best_path.exists():
        raise FileNotFoundError(
            "Missing outputs/project_best_summary.json. Finish the project-best tuning before finalizing outputs."
        )

    raw_df = load_input_dataframe(config)
    datasets = create_datasets(config, raw_df)
    baseline_metrics_df, baseline_per_target_metrics, _, baseline_metadata = evaluate_baselines(datasets, config)

    project_best_summary = _load_json(project_best_path)
    project_best_summary, hybrid_override = _with_latest_hybrid_outputs(project_best_summary, output_dir)
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

    all_metrics_df = pd.concat([core_metrics_df, baseline_metrics_df], axis=0)
    all_metrics_df = all_metrics_df.loc[_ordered_model_names(config, all_metrics_df)]
    all_per_target = {**core_per_target, **baseline_per_target_metrics}

    all_metrics_df.to_csv(output_dir / "metrics.csv")
    core_metrics_df.to_csv(output_dir / "core_model_metrics.csv")
    baseline_metrics_df.to_csv(output_dir / "baseline_metrics.csv")

    research_table_df = _build_paper_style_comparison(config, all_metrics_df)
    research_table_df.to_csv(output_dir / "research_performance_comparison.csv", index=False)

    literature_rows = _load_literature_models(config)
    save_per_target_tables(all_per_target, config, output_dir, literature_rows)
    save_split_protocol(datasets, config, output_dir)
    save_fair_baseline_note(literature_rows, all_metrics_df.index.tolist(), output_dir)

    ranked_df = all_metrics_df.sort_values(["RMSE", "MAE", "UPS"], ascending=[True, True, False]).reset_index()
    ranked_df = ranked_df.rename(columns={"index": "model"})
    best_four_df = ranked_df.head(4).copy()
    best_four_df.to_csv(output_dir / "final_best_models.csv", index=False)

    final_registry = {
        "finalized_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "source_summary": str(project_best_path),
        "top_4_models_by_rmse": best_four_df.to_dict(orient="records"),
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
    print(best_four_df[["model", "MAE", "RMSE", "NRMSE", "UPS"]].round(4).to_string(index=False))
    if hybrid_override:
        print(
            "\nMapped Hybrid to latest residual-hybrid artifact "
            f"({hybrid_override['source_artifact']})"
        )
    print(f"\nWrote finalized outputs to {output_dir}")


if __name__ == "__main__":
    main()
