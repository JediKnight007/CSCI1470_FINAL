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

PLOT_METRIC = "top1"

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_RE = re.compile(r"Train:\s*(\d+)\s*\[")
TEST_RE = re.compile(
    r"^Test(\s+\(EMA\))?:.*?Acc@1:\s*([0-9.]+)\s*\(\s*([0-9.]+)\s*\)"
)


MARKER_LABEL_Y = {
    "No attn": 0.46,
    "First-half attn": 0.52,
    "No bypass": 0.58,
    "MambaVision-T full": 0.64,
}


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

                # group(2) = current batch acc
                # group(3) = running average acc
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


def summarize_run(name, df, metric):
    """Get best epoch, best accuracy, final epoch, and final accuracy."""
    if df.empty or metric not in df.columns:
        return None

    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None

    best_idx = valid[metric].idxmax()
    best_row = valid.loc[best_idx]
    final_row = valid.iloc[-1]

    return {
        "Model": name,
        "Best Top-1": best_row[metric],
        "Best Epoch": int(best_row["epoch"]),
        "Last Epoch": int(final_row["epoch"]),
        "Final Top-1": final_row[metric],
    }


def simple_markdown_table(df):
    """
    Create a markdown table without requiring the optional 'tabulate' package.
    This avoids the pandas to_markdown() dependency error.
    """
    df_str = df.astype(str)

    header = "| " + " | ".join(df_str.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df_str.columns)) + " |"

    rows = []
    for _, row in df_str.iterrows():
        rows.append("| " + " | ".join(row.values) + " |")

    return "\n".join([header, separator] + rows)


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

        all_curves[name] = (df, metric)

        summary = summarize_run(name, df, metric)
        if summary is not None:
            summary_rows.append(summary)

    if not all_curves:
        raise RuntimeError("No curves found. Check your log file paths.")

    fig, ax = plt.subplots(figsize=(10, 6))

    max_last_epoch = 0

    for name, (df, metric) in all_curves.items():
        valid = df.dropna(subset=[metric])

        best_idx = valid[metric].idxmax()
        best_epoch = int(valid.loc[best_idx, "epoch"])
        best_acc = valid.loc[best_idx, metric]

        final_row = valid.iloc[-1]
        last_epoch = int(final_row["epoch"])
        final_acc = final_row[metric]

        max_last_epoch = max(max_last_epoch, last_epoch)

        line, = ax.plot(
            valid["epoch"],
            valid[metric],
            linewidth=2,
            label=f"{name}: best {best_acc:.3f}% @ epoch {best_epoch}",
        )

        color = line.get_color()

        # Mark best point
        ax.scatter(
            best_epoch,
            best_acc,
            s=45,
            color=color,
            zorder=5,
        )

        # Mark 4-hour stopping point
        ax.axvline(
            last_epoch,
            linestyle="--",
            linewidth=1.4,
            color=color,
            alpha=0.45,
        )

        ax.scatter(
            last_epoch,
            final_acc,
            s=45,
            color=color,
            zorder=6,
        )

        # Put text around the middle of the plot instead of near the bottom.
        # transform=ax.get_xaxis_transform():
        #   x = epoch value
        #   y = fraction of plot height, where 0.5 is vertical middle.
        label_y = MARKER_LABEL_Y.get(name, 0.55)

        ax.text(
            last_epoch + 1.5,
            label_y,
            f"4h stop\n{name}\nep {last_epoch}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="center",
            ha="left",
            fontsize=8,
            color=color,
            alpha=0.95,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1.5,
            ),
            clip_on=False,
        )

    ax.set_title("MambaVision-T Ablation Convergence on STL-10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Top-1 Accuracy (%)")
    ax.grid(True, alpha=0.3)

  
    ax.set_xlim(right=max_last_epoch + 18)

    ax.legend(
    fontsize=9,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
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
        summary_df["Best Top-1"] = summary_df["Best Top-1"].map(lambda x: f"{x:.3f}%")
        summary_df["Final Top-1"] = summary_df["Final Top-1"].map(lambda x: f"{x:.3f}%")

        csv_path = OUT_DIR / "mamba_ablation_cluster_summary.csv"
        md_path = OUT_DIR / "mamba_ablation_cluster_summary.md"

        summary_df.to_csv(csv_path, index=False)

        markdown_text = simple_markdown_table(summary_df)

        with md_path.open("w") as f:
            f.write(markdown_text)

        print("\nSummary:")
        print(markdown_text)
        print(f"\nSaved table to: {csv_path}")
        print(f"Saved markdown table to: {md_path}")


if __name__ == "__main__":
    main()