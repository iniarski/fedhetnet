
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ===========================================================================
# CONFIG — edit this block
# ===========================================================================

RUNS: dict[str, str] = {
   # "Centralized": "/home/filip/mgr_csv/centralized.csv",
  "FL-IID FedAdam":      "/home/filip/mgr_csv/fl_iid_fed_avg.csv",
# "FL-IID FedProx":      "/home/filip/mgr_csv/fl_iid_fed_prox.csv",
   "FL-Heterogeneous FedAdam":   "/home/filip/mgr_csv/fl_custom_fed_avg.csv",
#"FL-Heterogeneous FedProx":   "/home/filip/mgr_csv/fl_custom_fed_prox.csv",
}

METRICS: list[str] = [
    "val/loss",
    "val/acc",
   #a "val/f1",
    # "val/rec",
    # "val/prec",
   # "val/class_1_f1",
    # "val/class_2_f1",
    # "train/loss",
#    "train/mean_delta_l2_norm"
]

SMOOTH: int = 1   # rolling-average window; 1 = no smoothing
SAVE:   str = ""  # file path to save figure, e.g. "comparison.pdf"; "" = show

# ===========================================================================


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = [
    "#E63946",   # centralized  — vivid red
    "#457B9D",   # fl_iid       — steel blue
    "#2A9D8F",   # fl_custom    — teal
    "#E76F51",   # extra
    "#A8DADC",   # extra
    "#F4A261",   # extra
]

STYLE = {
    "axes.grid":            True,
    "font.family":          "monospace",
    "font.size":  20,
    "lines.linewidth":      1.8,
    "lines.solid_capstyle": "round",
}

METRIC_LABELS: dict[str, str] = {
    "val/loss":   "Validation Loss",
    "val/acc":    "Accuracy",
    "val/prec":   "Macro Precision",
    "val/rec":    "Macro Recall",
    "val/f1":     "Macro F1",
    "train/loss": "Train Loss",
    "train/mean_delta_l2_norm" : "Mean squared parameter difference",
    **{f"val/class_{i}_f1":   f"Class {i} F1"       for i in range(6)},
    **{f"val/class_{i}_prec": f"Class {i} Precision" for i in range(6)},
    **{f"val/class_{i}_rec":  f"Class {i} Recall"    for i in range(6)},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.replace("", np.nan, inplace=True)

    # Unify step/round → step
    if "round" in df.columns and "step" not in df.columns:
        df = df.rename(columns={"round": "step"})
    elif "round" in df.columns and df["step"].isna().all():
        df["step"] = df["round"]

    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series if window <= 1 else series.rolling(window, min_periods=1, center=True).mean()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(
    runs:    dict[str, pd.DataFrame],
    metrics: list[str],
    smooth_window: int,
    save:    str,
) -> None:
    n_cols = min(len(metrics), 3)
    n_rows = (len(metrics) + n_cols - 1) // n_cols

    with matplotlib.rc_context(STYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4.2 * n_rows),
            squeeze=False,
        )

        for ax_idx, metric in enumerate(metrics):
            row, col = divmod(ax_idx, n_cols)
            ax = axes[row][col]
            any_plotted = False

            for (label, df), color in zip(runs.items(), PALETTE):
                if metric not in df.columns:
                    continue
                sub = df[["step", metric]].dropna()
                if sub.empty:
                    continue

                x    = sub["step"].values
                raw  = sub[metric].values
                smth = smooth(sub[metric], smooth_window).values

                ax.plot(x, smth, label=label, color=color, alpha=0.92, marker='s')
                if smooth_window > 1:
                    ax.plot(x, raw, color=color, alpha=0.15, linewidth=0.7)

                any_plotted = True

            ax.set_title(METRIC_LABELS.get(metric, metric), pad=8)
            ax.set_xlabel("Round")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
            )
            if 'loss' in metric:
                ax.set_yscale('log')

            if not any_plotted:
                ax.text(0.5, 0.5, f'"{metric}" not found',
                        ha="center", va="center", transform=ax.transAxes,
                        color="#6B6F80")

            if ax_idx == 0:
                ax.legend(framealpha=0.6)

        # Hide unused subplot slots
        for ax_idx in range(len(metrics), n_rows * n_cols):
            row, col = divmod(ax_idx, n_cols)
            axes[row][col].set_visible(False)

        fig.tight_layout()

        if save:
            fig.savefig(save, dpi=180, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {save}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    loaded: dict[str, pd.DataFrame] = {}

    for label, path in RUNS.items():
        if not Path(path).exists():
            print(f"WARNING: '{label}' — file not found: {path}", file=sys.stderr)
            continue
        try:
            df = load_csv(path)
            print(f"Loaded  [{label}]  {path}  ({len(df)} rows)")
            loaded[label] = df
        except Exception as e:
            print(f"ERROR loading '{label}' ({path}): {e}", file=sys.stderr)

    if not loaded:
        print("No valid CSV files loaded — check paths in RUNS.", file=sys.stderr)
        sys.exit(1)

    plot(loaded, METRICS, SMOOTH, SAVE)


if __name__ == "__main__":
    main()
