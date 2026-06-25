from __future__ import annotations

"""Generate a compact IEEE-style 2x2 figure of actual vs hybrid predictions.

Targets: Traffic Flow, AQI, Temperature, Electricity Demand.

The figure is sized for an IEEE two-column layout (~7 x 5 in), uses a single
shared legend, shows only the final test window, and is exported as both a
300 DPI PNG and a vector PDF for direct insertion into the Experimental
Results section.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent

# Ordered so the 2x2 grid reads (a) top-left, (b) top-right, (c), (d).
TARGETS = [
    ("traffic_flow", "(a) Traffic Flow", "Vehicles/hr"),
    ("aqi", "(b) AQI", "AQI"),
    ("temperature", "(c) Temperature", "\u00b0C"),
    ("electricity_demand", "(d) Electricity Demand", "MW"),
]

# Single accent colour for the hybrid prediction across all panels.
HYBRID_COLOR = "#d62728"

ACTUAL_LW = 1.8
HYBRID_LW = 1.6
TAIL_SAMPLES = 48


def _apply_ieee_style() -> None:
    """Matplotlib rc settings tuned for IEEE publication figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_plot(predictions_path: Path, out_stub: Path, tail: int) -> None:
    df = pd.read_csv(predictions_path)

    has_time = "timestamp" in df.columns
    if has_time:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Keep only the final `tail` test samples to reduce clutter.
    if tail and len(df) > tail:
        df = df.tail(tail).reset_index(drop=True)

    x = df["timestamp"] if has_time else np.arange(len(df))
    x_label = "Time" if has_time else "Test sample"

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=False)
    axes = axes.ravel()

    actual_handle = hybrid_handle = None

    for idx, (ax, (target, title, unit)) in enumerate(zip(axes, TARGETS)):
        actual_col = f"actual_{target}"
        predicted_col = f"predicted_{target}"
        if actual_col not in df.columns or predicted_col not in df.columns:
            ax.set_visible(False)
            continue

        actual = df[actual_col].to_numpy(dtype=float)
        predicted = df[predicted_col].to_numpy(dtype=float)

        (line_actual,) = ax.plot(x, actual, color="black", linewidth=ACTUAL_LW, label="Actual")
        (line_hybrid,) = ax.plot(
            x, predicted, color=HYBRID_COLOR, linewidth=HYBRID_LW, linestyle="--", label="Hybrid Model"
        )
        actual_handle, hybrid_handle = line_actual, line_hybrid

        ax.set_title(title, pad=4)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.margins(x=0.02)

        # Scientific notation for the large traffic-flow magnitudes.
        if target == "traffic_flow":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(6)

        # X-axis labels only on the bottom row (indices 2 and 3).
        if idx in (2, 3):
            ax.set_xlabel(x_label)
            if has_time:
                ax.tick_params(axis="x", labelrotation=30)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # One shared legend for the whole figure.
    if actual_handle is not None and hybrid_handle is not None:
        fig.legend(
            handles=[actual_handle, hybrid_handle],
            labels=["Actual", "Hybrid Model"],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
        )

    # Leave a thin strip at the top for the shared legend; no overall title.
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_stub.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_stub.with_suffix(".png")
    pdf_path = out_stub.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot compact IEEE actual vs hybrid figure.")
    parser.add_argument("--city", default="bangalore", help="City output folder under outputs/.")
    parser.add_argument(
        "--tail",
        type=int,
        default=TAIL_SAMPLES,
        help="Number of final test samples to show (default 48).",
    )
    args = parser.parse_args()

    _apply_ieee_style()

    city_dir = ROOT_DIR / "outputs" / args.city
    predictions_path = city_dir / "tft_gru_residual_hybrid_predictions.csv"
    if not predictions_path.exists():
        raise SystemExit(f"Predictions file not found: {predictions_path}")

    out_stub = city_dir / "actual_vs_predicted_ieee"
    build_plot(predictions_path, out_stub, args.tail)


if __name__ == "__main__":
    main()
