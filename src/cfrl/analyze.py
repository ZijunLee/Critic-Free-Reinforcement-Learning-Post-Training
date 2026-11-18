"""Aggregate runs into a results table and plots.

Usage:
    python -m cfrl.analyze runs --out results

Runs are grouped by variant = run_name without the trailing "_s<seed>".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in root.iterdir() if (p / "config.json").exists()):
        cfg = json.loads((run_dir / "config.json").read_text())
        summary_path = run_dir / "summary.json"
        runs.append({
            "name": run_dir.name,
            "variant": re.sub(r"_s\d+$", "", run_dir.name),
            "seed": cfg["seed"],
            "config": cfg,
            "metrics": _read_jsonl(run_dir / "metrics.jsonl"),
            "eval": _read_jsonl(run_dir / "eval.jsonl"),
            "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        })
    return runs


def pooled_snr(metrics: pd.DataFrame) -> float:
    """(mean ||g_bar||^2 - mean tr Cov) / mean tr Cov over all steps."""
    var = float(metrics["grad_var_mean"].mean())
    return (float(metrics["grad_sq_norm"].mean()) - var) / var if var > 0 else float("nan")


def snr(metrics: pd.DataFrame) -> pd.Series:
    """(||g_bar||^2 - tr Cov) / tr Cov: scale-free signal-to-noise ratio of the step gradient."""
    var = metrics["grad_var_mean"].replace(0, np.nan)
    return (metrics["grad_sq_norm"] - var) / var


def subset_curve(ev: pd.DataFrame) -> pd.DataFrame:
    """Accuracy on the in-training eval subset, including the final full-test pass (accuracy_prefix)."""
    if ev.empty:
        return ev
    sub = ev[ev["split"] == "test_subset"][["step", "accuracy"]]
    fin = ev[ev["split"] == "test_final"]
    if "accuracy_prefix" in fin:
        sub = pd.concat([sub, fin[["step", "accuracy_prefix"]].rename(columns={"accuracy_prefix": "accuracy"})])
    return sub.sort_values("step")


def summarise(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m, ev, s = r["metrics"], r["eval"], r["summary"]
        if m.empty or s is None:
            continue
        curve = subset_curve(ev)
        within = m["grad_var_within"].mean() if "grad_var_within" in m else np.nan
        rows.append({
            "variant": r["variant"],
            "seed": r["seed"],
            "final_acc": s["final_accuracy"],
            "start_acc_subset": float(curve["accuracy"].iloc[0]) if len(curve) else np.nan,
            "best_acc_subset": float(curve["accuracy"].max()) if len(curve) else np.nan,
            # pooled over steps: per-step SNR estimates are dominated by noise
            "pooled_snr": pooled_snr(m),
            # the pooled ratio is sensitive to a few large-gradient steps; the median and a spike count show them
            "median_snr": float(snr(m).median()),
            "spike_steps": int((m["grad_norm_preclip"] > 2 * m["grad_norm_preclip"].median()).sum()),
            "grad_var_total": float(m["grad_var_mean"].mean()),
            "within_share": float(within / m["grad_var_mean"].mean()) if within == within else np.nan,
            "zero_var_groups": float(m["zero_variance_group_frac"].mean()),
            "active_frac": float(m["nonzero_adv_frac"].mean()),
            "gen_s": float(m["time_gen_s"].mean()),
            "train_s": float(m["time_train_s"].mean()),
            "final_len": float(m["completion_tokens_mean"].tail(10).mean()),
            "gen_Mtokens": s["gen_tokens_total"] / 1e6,
            "hours": s["wall_clock_s"] / 3600,
        })
    return pd.DataFrame(rows)


def table(per_run: pd.DataFrame) -> pd.DataFrame:
    def fmt(col):
        return lambda g: f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}" if len(g) > 1 else f"{g[col].mean():.3f}"

    out = []
    for variant, g in per_run.groupby("variant"):
        out.append({
            "variant": variant,
            "seeds": len(g),
            "final test acc": fmt("final_acc")(g),
            "start acc (subset)": f"{g['start_acc_subset'].mean():.3f}",
            "best acc (subset)": fmt("best_acc_subset")(g),
            "grad var tr Cov": f"{g['grad_var_total'].mean():.2e}",
            "within-prompt share": "–" if g["within_share"].isna().all() else f"{g['within_share'].mean():.2f}",
            "pooled grad SNR": f"{g['pooled_snr'].mean():.3f}",
            "median step SNR": f"{g['median_snr'].mean():.3f}",
            "spike steps": int(g["spike_steps"].sum()),
            "zero-variance groups": f"{g['zero_var_groups'].mean():.2f}",
            "samples with nonzero advantage": f"{g['active_frac'].mean():.2f}",
            "gen / train s per step": f"{g['gen_s'].mean():.1f} / {g['train_s'].mean():.1f}",
            "final length (tok)": f"{g['final_len'].mean():.0f}",
            "generated Mtok": f"{g['gen_Mtokens'].mean():.1f}",
            "GPU h / run": f"{g['hours'].mean():.1f}",
        })
    return pd.DataFrame(out).sort_values("variant")


# comparisons shown in the README (no multiple-comparison correction): baseline vs none, group baselines, std scaling
COMPARISONS = [("grpo", "reinforce"), ("rloo", "reinforce"), ("grpo", "rloo"), ("grpo", "grpo_nostd"),
               ("rloo", "reinforce_ema"), ("grpo", "remax")]
GROUP_BASELINES = ("grpo", "grpo_nostd", "rloo")
OTHER_BASELINES = ("reinforce", "reinforce_ema", "remax")


def tests(per_run: pd.DataFrame) -> pd.DataFrame:
    """Welch t-tests on final accuracy across seeds (each seed = one independent training run)."""
    from scipy import stats

    acc = {v: g["final_acc"].to_numpy() for v, g in per_run.groupby("variant")}
    rows = []

    def add(label, a, b):
        if len(a) < 2 or len(b) < 2:
            return
        t = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"comparison": label, "n": f"{len(a)} vs {len(b)}",
                     "difference (pp)": f"{100 * (a.mean() - b.mean()):+.1f}", "Welch p": f"{t.pvalue:.3f}"})

    for x, y in COMPARISONS:
        if x in acc and y in acc:
            add(f"{x} − {y}", acc[x], acc[y])
    group = [acc[v] for v in GROUP_BASELINES if v in acc]
    other = [acc[v] for v in OTHER_BASELINES if v in acc]
    if group and other:
        add("group baselines − others (pooled runs)", np.concatenate(group), np.concatenate(other))
    return pd.DataFrame(rows)


def plot(runs: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({r["variant"] for r in runs if "-" not in r["variant"]})  # main matrix; ablations are in the table
    panels = [
        ("Test accuracy (eval subset, greedy)", lambda r: subset_curve(r["eval"]).set_index("step")["accuracy"]),
        ("Gradient variance tr Cov (rolling mean, log)", lambda r: r["metrics"].set_index("step")["grad_var_mean"]
            .rolling(10, min_periods=1).mean()),
        ("Completion length (tokens)", lambda r: r["metrics"].set_index("step")["completion_tokens_mean"]
            .rolling(10, min_periods=1).mean()),
        ("Zero-variance groups (fraction)", lambda r: r["metrics"].set_index("step")["zero_variance_group_frac"]
            .rolling(10, min_periods=1).mean()),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, getter) in zip(axes.flat, panels):
        for v in variants:
            curves = [getter(r) for r in runs if r["variant"] == v and not r["metrics"].empty]
            curves = [c for c in curves if len(c)]
            if not curves:
                continue
            df = pd.concat(curves, axis=1)
            mean = df.mean(axis=1)
            line, = ax.plot(mean.index, mean.values, label=v)
            if df.shape[1] > 1:
                ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), alpha=0.15, color=line.get_color())
        ax.set_title(title)
        ax.set_xlabel("step")
        if "log" in title:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", default="runs")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.runs))
    per_run = summarise(runs)
    if per_run.empty:
        raise SystemExit("no finished runs found")
    per_run.to_csv(out / "per_run.csv", index=False)
    tab = table(per_run)
    (out / "summary.md").write_text(tab.to_markdown(index=False) + "\n")
    print(tab.to_markdown(index=False))
    tst = tests(per_run)
    if not tst.empty:
        (out / "tests.md").write_text(tst.to_markdown(index=False) + "\n")
        print(tst.to_markdown(index=False))
    plot(runs, out)
    print(f"wrote {out}/summary.md, tests.md, per_run.csv, curves.png")


if __name__ == "__main__":
    main()
