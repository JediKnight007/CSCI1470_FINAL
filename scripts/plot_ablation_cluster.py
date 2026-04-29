from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt



RUNS = {
    "MambaVision-T full": "Runs/Max Mamba Run",
    "No bypass": "Ablations/Ablation 1 - No Bypass Run",
    "First-half attn": "Ablations/Ablation 2 - First-Half Attn (err)",
    "No attn": "Ablations/Ablation 3 - No Attn (err)",
}


PARAMS = {
    "MambaVision-T full": "31.8M",
    "No bypass": "~31.8M",
    "First-half attn": "31.8M",
    "No attn": "31.8M",
}

FOUR_HOUR_STOP_EPOCHS = {
    "MambaVision-T full": 290,
    "No bypass": 280,
    "First-half attn": 270,
    "No attn": 259,
}


PLOT_METRIC = "top1"

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_RE = re.compile(r"Train:\s*(\d+)\s*\[")
TEST_RE = re.compile(
    r"^Test(\s+\(EMA\))?:.*?Acc@1:\s*([0-9.]+)\s*\(\s*([0-9.]+)\s*\)"
)


def clean_line(line):
    """Remove terminal color codes."""
    return ANSI_RE.sub("", line).strip()


def parse_log(log_path):
    """
    Parse one training log.

    For lines like:
    Test: [62/62] ... Acc@1: 12.5000 (29.4250)

    We use the value inside parentheses because that is the running average
    over the validation set.
    """
    log_path = Path(log_path)

    if not log_path.exists():
        print(f"[WARNING] Missing file: {log_path}")
        return pd.DataFrame()

    records = {}
    current_epoch = None

    with log_path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = clean_line(raw_line)

            train_match = TRAIN_RE.search(line)
            if train_match:
                current_epoch = int(train_match.group(1))

            test_match = TEST_RE.search(line)
            if test_match and current_epoch is not None:
                is_ema = test_match.group(1) is not None

                # group(2) = current batch Acc@1
                # group(3) = running average Acc@1
                avg_acc1 = float(test_match.group(3))

                if current_epoch not in records:
                    records[current_epoch] = {"epoch": current_epoch}

                key = "ema_top1" if is_ema else "top1"
                records[current_epoch][key] = avg_acc1

    df = pd.DataFrame(records.values())

    if df.empty:
        print(f"[WARNING] No validation records parsed from: {log_path}")
        return df

    return df.sort_values("epoch").reset_index(drop=True)


def restrict_to_stop_epoch(name, df):
    """
    Keep only epochs up to the 4-hour stop epoch.

    This makes the comparison fair because all runs are compared under
    the same walltime budget, even though they reach different epoch counts.
    """
    if name not in FOUR_HOUR_STOP_EPOCHS:
        return df

    stop_epoch = FOUR_HOUR_STOP_EPOCHS[name]
    return df[df["epoch"] <= stop_epoch].copy()


def summarize_run(name, df, metric):
    """Get best epoch, best accuracy, and final accuracy for one run."""
    if df.empty or metric not in df.columns:
        return None

    valid = df.dropna(subset=[metric])

    if valid.empty:
        return None

    best_idx = valid[metric].idxmax()
    best_row = valid.loc[best_idx]

    stop_epoch = FOUR_HOUR_STOP_EPOCHS.get(name, int(valid["epoch"].max()))

    return {
        "Model": name,
        "Params": PARAMS.get(name, "—"),
        "Best Top-1": float(best_row[metric]),
        "Best Epoch": int(best_row["epoch"]),
        "4h Stop Epoch": int(stop_epoch),
        "Final Top-1": float(valid.iloc[-1][metric]),
    }


def df_to_markdown_no_tabulate(df):
    """
    Convert a DataFrame to a markdown table without using pandas.to_markdown().
    This avoids the missing 'tabulate' dependency error.
    """
    if df.empty:
        return ""

    df_str = df.astype(str)
    headers = list(df_str.columns)
    rows = df_str.values.tolist()

    widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(row[i]))
        widths.append(max_width)

    def format_row(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"

    header_line = format_row(headers)
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    row_lines = [format_row(row) for row in rows]

    return "\n".join([header_line, sep_line] + row_lines)


def main():
    all_curves = {}
    summary_rows = []

    for name, path in RUNS.items():
        df = parse_log(path)

        if df.empty:
            continue

        metric = PLOT_METRIC

        if metric not in df.columns:
            if "top1" in df.columns:
                metric = "top1"
            elif "ema_top1" in df.columns:
                metric = "ema_top1"
            else:
                print(f"[WARNING] No usable Top-1 metric for: {name}")
                continue

        # Important: restrict to the 4-hour stop epoch
        df = restrict_to_stop_epoch(name, df)

        if df.empty:
            print(f"[WARNING] No data left after stop-epoch restriction for: {name}")
            continue

        all_curves[name] = (df, metric)

        summary = summarize_run(name, df, metric)
        if summary is not None:
            summary_rows.append(summary)

    if not all_curves:
        raise RuntimeError("No curves found. Check your log file paths.")


    fig, ax = plt.subplots(figsize=(11, 6.5))

    plotted_stop_lines = []

    for name, (df, metric) in all_curves.items():
        valid = df.dropna(subset=[metric])

        best_idx = valid[metric].idxmax()
        best_epoch = int(valid.loc[best_idx, "epoch"])
        best_acc = float(valid.loc[best_idx, metric])

        line, = ax.plot(
            valid["epoch"],
            valid[metric],
            linewidth=2,
            label=f"{name}: best {best_acc:.3f}% @ epoch {best_epoch}",
        )

        color = line.get_color()

        ax.scatter(
            best_epoch,
            best_acc,
            s=45,
            color=color,
            zorder=4,
        )

        stop_epoch = FOUR_HOUR_STOP_EPOCHS.get(name, int(valid["epoch"].max()))

        ax.axvline(
            x=stop_epoch,
            linestyle="--",
            linewidth=1.3,
            alpha=0.55,
            color=color,
        )

        plotted_stop_lines.append((name, stop_epoch, color))


    for i, (name, stop_epoch, color) in enumerate(plotted_stop_lines):
        y_pos = 0.03 + 0.07 * (i % 4)

        ax.text(
            stop_epoch + 1,
            y_pos,
            f"4h stop\n{name}\nep {stop_epoch}",
            rotation=90,
            transform=ax.get_xaxis_transform(),
            color=color,
            fontsize=8,
            va="bottom",
            ha="left",
        )

    ax.set_title("MambaVision-T Ablation Convergence on STL-10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Top-1 Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2
)
    fig.tight_layout()

    png_path = OUT_DIR / "mamba_ablation_cluster_convergence.png"
    pdf_path = OUT_DIR / "mamba_ablation_cluster_convergence.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved plot to: {png_path}")
    print(f"Saved plot to: {pdf_path}")


    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        baseline_name = "MambaVision-T full"

        if baseline_name in summary_df["Model"].values:
            baseline_best = float(
                summary_df.loc[
                    summary_df["Model"] == baseline_name,
                    "Best Top-1",
                ].iloc[0]
            )

            summary_df["Delta vs Baseline"] = summary_df["Best Top-1"] - baseline_best
        else:
            summary_df["Delta vs Baseline"] = pd.NA


        summary_df["Best Top-1"] = summary_df["Best Top-1"].map(lambda x: f"{x:.3f}%")
        summary_df["Final Top-1"] = summary_df["Final Top-1"].map(lambda x: f"{x:.3f}%")

        def format_delta(x):
            if pd.isna(x):
                return "—"
            if abs(x) < 1e-12:
                return "—"
            return f"{x:+.3f}%"

        summary_df["Delta vs Baseline"] = summary_df["Delta vs Baseline"].map(format_delta)


        summary_df = summary_df[
            [
                "Model",
                "Params",
                "Best Top-1",
                "Best Epoch",
                "4h Stop Epoch",
                "Final Top-1",
                "Delta vs Baseline",
            ]
        ]

        csv_path = OUT_DIR / "mamba_ablation_cluster_summary.csv"
        md_path = OUT_DIR / "mamba_ablation_cluster_summary.md"

        summary_df.to_csv(csv_path, index=False)

        markdown_table = df_to_markdown_no_tabulate(summary_df)

        with md_path.open("w") as f:
            f.write(markdown_table)

        print("\nSummary:")
        print(markdown_table)

        print(f"\nSaved table to: {csv_path}")
        print(f"Saved markdown table to: {md_path}")


if __name__ == "__main__":
    main()