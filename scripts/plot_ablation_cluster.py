from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


RUNS = {
    "MambaVision-T full": "Runs/Max Mamba Run",
    "No bypass": "Ablations/Ablation 1 - No Bypass Run",
    "First-half attn": "Ablations/Ablation 2 - First-Half Attn (err)",
    "No attn": "Ablations/Ablation 3 - No Attn (err)",
    "ViT-Small": "Runs/ViT Small Run",
    "ViT-Tiny": "Runs/Max ViTTiny Run",
}

PARAMS = {
    "MambaVision-T full": "31.8M",
    "No bypass": "~31.8M",
    "First-half attn": "31.8M",
    "No attn": "31.8M",
    "ViT-Small": "—",
    "ViT-Tiny": "—",
}

FOUR_HOUR_STOP_EPOCHS = {
    "MambaVision-T full": 290,
    "No bypass": 280,
    "First-half attn": 270,
    "No attn": 259,
}

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAMBA_TRAIN_RE = re.compile(r"Train:\s*(\d+)\s*\[")
MAMBA_TEST_RE = re.compile(
    r"^Test(\s+\(EMA\))?:.*?Acc@1:\s*([0-9.]+)\s*\(\s*([0-9.]+)\s*\)"
)
VIT_VAL_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*\d+\s*[\u2014\u2013-]\s*Val\s+Acc:\s*([0-9.]+)\s*%"
)


def clean_line(line):
    return ANSI_RE.sub("", line).strip()


def parse_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"[WARNING] Missing file: {log_path}")
        return pd.DataFrame()

    records = {}
    current_epoch = None

    with log_path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = clean_line(raw_line)

            vit_match = VIT_VAL_RE.search(line)
            if vit_match:
                epoch = int(vit_match.group(1))
                val_acc = float(vit_match.group(2))
                if epoch not in records:
                    records[epoch] = {"epoch": epoch}
                records[epoch]["top1"] = val_acc
                continue

            train_match = MAMBA_TRAIN_RE.search(line)
            if train_match:
                current_epoch = int(train_match.group(1))
                continue

            test_match = MAMBA_TEST_RE.search(line)
            if test_match and current_epoch is not None:
                is_ema = test_match.group(1) is not None
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


def choose_metric(df):
    if "ema_top1" in df.columns and df["ema_top1"].notna().any():
        return "ema_top1", "EMA"
    if "top1" in df.columns and df["top1"].notna().any():
        return "top1", "Val"
    return None, None


def restrict_to_stop_epoch(name, df):
    stop_epoch = FOUR_HOUR_STOP_EPOCHS.get(name, None)
    if stop_epoch is None:
        return df.copy()
    return df[df["epoch"] <= stop_epoch].copy()


def summarize_run(name, df, metric, metric_label):
    if df.empty or metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None
    best_idx = valid[metric].idxmax()
    best_row = valid.loc[best_idx]
    last_row = valid.iloc[-1]
    return {
        "Model": name,
        "Metric": metric_label,
        "Params": PARAMS.get(name, "—"),
        "Best Top-1": float(best_row[metric]),
        "Best Epoch": int(best_row["epoch"]),
        "Last Epoch": int(last_row["epoch"]),
        "Last Top-1": float(last_row[metric]),
    }


def df_to_markdown_no_tabulate(df):
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
        return "| " + " | ".join(
            str(row[i]).ljust(widths[i]) for i in range(len(row))
        ) + " |"

    header_line = format_row(headers)
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    row_lines = [format_row(row) for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def annotate_point(ax, x, y, text, color, xytext):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8,
        color=color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="square,pad=0.25", fc="white", ec=color, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.1, alpha=0.9),
        clip_on=False,
    )


def main():
    all_curves = {}
    summary_rows = []

    for name, path in RUNS.items():
        df = parse_log(path)
        if df.empty:
            continue
        df = restrict_to_stop_epoch(name, df)
        if df.empty:
            print(f"[WARNING] No data left after stop-epoch restriction for: {name}")
            continue
        metric, metric_label = choose_metric(df)
        if metric is None:
            print(f"[WARNING] No usable Top-1 metric for: {name}")
            continue
        all_curves[name] = (df, metric, metric_label)
        summary = summarize_run(name, df, metric, metric_label)
        if summary is not None:
            summary_rows.append(summary)

    if not all_curves:
        raise RuntimeError("No curves found. Check your log file paths.")

    fig, ax = plt.subplots(figsize=(12.5, 6.8))

    # ---------------------------------------------------------------
    # FIXED: spread blue (MambaVision-T full, idx=0) and teal
    # (First-half attn, idx=2) annotations so they no longer overlap.
    # ---------------------------------------------------------------
    best_offsets = [
        (-75, 95),   # 0 MambaVision-T full – pushed higher
        (-90, -35),  # 1 No bypass
        (-70, 45),   # 2 First-half attn – pulled down from 70 → 45
        (-85, -55),  # 3 No attn
        (65, 45),    # 4 ViT-Small
        (65, -40),   # 5 ViT-Tiny
    ]

    last_offsets = [
        (105, 70),   # 0 MambaVision-T full – pushed right & up
        (70, -55),   # 1 No bypass
        (80, 100),   # 2 First-half attn – pushed higher
        (75, -35),   # 3 No attn
        (75, 15),    # 4 ViT-Small
        (75, -15),   # 5 ViT-Tiny
    ]

    for i, (name, (df, metric, metric_label)) in enumerate(all_curves.items()):
        valid = df.dropna(subset=[metric])

        best_idx = valid[metric].idxmax()
        best_epoch = int(valid.loc[best_idx, "epoch"])
        best_acc = float(valid.loc[best_idx, metric])
        last_epoch = int(valid.iloc[-1]["epoch"])
        last_acc = float(valid.iloc[-1][metric])

        line, = ax.plot(
            valid["epoch"],
            valid[metric],
            linewidth=2.2,
            label=(
                f"{name} ({metric_label}): "
                f"best {best_acc:.3f}% @ ep {best_epoch}; "
                f"last {last_acc:.3f}% @ ep {last_epoch}"
            ),
        )

        color = line.get_color()

        ax.scatter(best_epoch, best_acc, s=55, color=color, edgecolor="black",
                   linewidth=0.6, zorder=5, marker="o")
        ax.scatter(last_epoch, last_acc, s=75, color=color, edgecolor="black",
                   linewidth=0.6, zorder=6, marker="*")
        ax.axvline(x=last_epoch, linestyle="--", linewidth=1.1, alpha=0.35, color=color)

        same_point = (best_epoch == last_epoch and abs(best_acc - last_acc) < 1e-10)

        if same_point:
            annotate_point(
                ax, best_epoch, best_acc,
                f"Best = Last {metric_label}\n{best_acc:.3f}%\nep {best_epoch}",
                color, best_offsets[i % len(best_offsets)],
            )
        else:
            annotate_point(
                ax, best_epoch, best_acc,
                f"Best {metric_label}\n{best_acc:.3f}%\nep {best_epoch}",
                color, best_offsets[i % len(best_offsets)],
            )
            annotate_point(
                ax, last_epoch, last_acc,
                f"Last {metric_label}\n{last_acc:.3f}%\nep {last_epoch}",
                color, last_offsets[i % len(last_offsets)],
            )

    ax.set_title(
        "SSM–Transformer Synergy: a Robust Design Essential for Low-Data Regimes",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Ablation insight box – placed in the blank mid-left region
    # ------------------------------------------------------------------
    insight_lines = [
        "Ablation findings (STL-10)",
        "",
        "❶  No Bypass: removing the non-SSM branch\n"
        "    severely hurts accuracy → the symmetric\n"
        "    path is critical for global context (H₁ ✓)",
        "",
        "❷  Early Attention: shifting self-attention to\n"
        "    early layers degrades performance → attention\n"
        "    is most effective at refining late-stage tokens",
        "",
        "❸  Pure Mamba: largest drop on STL-10 vs ImageNet\n"
        "    → hybrid integration is vital for small datasets",
    ]
    insight_text = "\n".join(insight_lines)

    ax.text(
        0.36, 0.54,            # mid-left blank region
        insight_text,
        transform=ax.transAxes,
        fontsize=7.8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#f7f7f7",
            edgecolor="#888888",
            alpha=0.92,
            linewidth=0.9,
        ),
        linespacing=1.45,
        family="monospace",
    )
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    png_path = OUT_DIR / "mamba_vit_ablation_cluster_convergence.png"
    pdf_path = OUT_DIR / "mamba_vit_ablation_cluster_convergence.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {png_path}")
    print(f"Saved plot to: {pdf_path}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        baseline_name = "MambaVision-T full"
        if baseline_name in summary_df["Model"].values:
            baseline_best = float(
                summary_df.loc[summary_df["Model"] == baseline_name, "Best Top-1"].iloc[0]
            )
            summary_df["Delta vs MambaVision-T full"] = (
                summary_df["Best Top-1"] - baseline_best
            )
        else:
            summary_df["Delta vs MambaVision-T full"] = pd.NA

        summary_df["Best Top-1"] = summary_df["Best Top-1"].map(lambda x: f"{x:.3f}%")
        summary_df["Last Top-1"] = summary_df["Last Top-1"].map(lambda x: f"{x:.3f}%")

        def format_delta(x):
            if pd.isna(x): return "—"
            if abs(x) < 1e-12: return "—"
            return f"{x:+.3f}%"

        summary_df["Delta vs MambaVision-T full"] = summary_df[
            "Delta vs MambaVision-T full"
        ].map(format_delta)

        summary_df = summary_df[[
            "Model", "Metric", "Params", "Best Top-1", "Best Epoch",
            "Last Top-1", "Last Epoch", "Delta vs MambaVision-T full",
        ]]

        csv_path = OUT_DIR / "mamba_vit_ablation_cluster_summary.csv"
        md_path  = OUT_DIR / "mamba_vit_ablation_cluster_summary.md"
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