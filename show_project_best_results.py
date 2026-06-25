from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from utils.config import CONFIG, apply_city_config


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def parse_args():
    parser = argparse.ArgumentParser(description="Show project-best tuning results.")
    parser.add_argument("--city", default=None, help="Use city-specific outputs, e.g. delhi.")
    return parser.parse_args()


def main() -> None:
    config = apply_city_config(copy.deepcopy(CONFIG), parse_args().city)
    output_dir = Path(config.output_dir)
    summary_path = output_dir / "project_best_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run `python3 tune_project_best_models.py --city {config.city}` first."
        )

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    _print_header("Project-Best Headline Metrics")
    for model_name, payload in summary["results"].items():
        metrics = payload["test_metrics"]
        print(
            f"{model_name:<8} "
            f"MAE={metrics['MAE']:.4f} "
            f"RMSE={metrics['RMSE']:.4f} "
            f"NRMSE={metrics['NRMSE']:.4f} "
            f"UPS={metrics['UPS']:.4f}"
        )

    _print_header("Comparison To Current Saved Results")
    for model_name, payload in summary["results"].items():
        comparison = payload.get("comparison_to_current", {})
        if not comparison:
            continue
        rmse_delta = comparison["RMSE"]["delta"]
        mae_delta = comparison["MAE"]["delta"]
        ups_delta = comparison["UPS"]["delta"]
        print(
            f"{model_name:<8} "
            f"delta_MAE={mae_delta:+.4f} "
            f"delta_RMSE={rmse_delta:+.4f} "
            f"delta_UPS={ups_delta:+.4f}"
        )

    for model_name in summary["results"]:
        csv_path = output_dir / f"project_best_{model_name.lower()}_search.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        columns = [col for col in df.columns if not col.startswith("checkpoint_path")]
        _print_header(f"{model_name} Search Table")
        print(df[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
