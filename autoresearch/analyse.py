#!/usr/bin/env python3
"""Turn the autoresearch journal into blog-ready plots and tables.

This reads the structured record of the optimization run and produces the two
artefacts a writeup needs (§10.7 of ``ale_vector_optimization.md``):

1. **Timeline plot** (``progress.png``) - environment steps-per-second (the north
   star) over the sequence of changes, one panel per ``num_envs`` in
   ``{8, 64, 256}``. The ALE trajectory is drawn as a running-best frontier with
   every attempt marked by outcome (accept / revert / gate-fail). ``gym_sync``,
   ``gym_async`` and ``envpool`` are drawn as horizontal reference lines so the
   reader sees where ALE started, how far each change moved it, and how it
   compares to the standard baselines.

2. **Metrics table** (``metrics.md`` + ``metrics.csv``) - one row per change,
   columns for the metrics: verdict, geomean speedup, SPS at each env count, the
   p99 latency and major-fault counts that *explain* the SPS move, and the
   cumulative speedup versus the pre-optimization baseline.

Inputs (all produced by the loop; see SKILL.md "records"):
  records/iterNN.json        one per iteration - the narrative + verdict metadata
  results/iterNN_bench.json  the candidate bench.py report for that iteration
  results/benchmarks.json    a one-off bench.py report with the reference
                             backends (gym_sync/gym_async/envpool) + ALE baseline

Usage
-----
    python autoresearch/analyse.py                       # writes to autoresearch/report/
    python autoresearch/analyse.py --game breakout --num-envs 8 64 256
    python autoresearch/analyse.py --out-dir /tmp/blog --show

Missing pieces degrade gracefully: a gate-fail iteration with no bench report is
shown as a marker with no SPS, reference lines are omitted if benchmarks.json is
absent, so this is safe to run at any point mid-loop.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

REF_BACKENDS = ["gym_sync", "gym_async", "envpool"]
REF_COLORS = {"gym_sync": "#888888", "gym_async": "#c9a227", "envpool": "#1f77b4"}
OUTCOME_STYLE = {
    "accept": ("#2ca02c", "o", "accepted"),
    "revert": ("#d62728", "x", "reverted"),
    "gate-fail": ("#7f7f7f", "s", "gate-fail"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class Iteration:
    iter: int
    title: str
    backlog_item: str
    outcome: str            # accept | revert | gate-fail
    verdict: str
    geomean_speedup: float | None
    hypothesis: str
    branch: str
    commit: str | None
    record: dict[str, Any]
    bench: dict[str, Any] | None  # the candidate bench.py report, if any


def _load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_iterations(records_dir: str, results_dir: str) -> list[Iteration]:
    iters: list[Iteration] = []
    for path in sorted(glob.glob(os.path.join(records_dir, "iter*.json"))):
        rec = _load_json(path)
        if rec is None:
            print(f"WARNING: could not parse {path}; skipped")
            continue
        n = int(rec.get("iter", len(iters) + 1))
        bench_ref = rec.get("bench_candidate")
        bench = None
        if bench_ref:
            # Allow either an absolute path or one relative to the repo/results.
            for cand in (bench_ref, os.path.join(results_dir, os.path.basename(bench_ref))):
                bench = _load_json(cand)
                if bench is not None:
                    break
        g = rec.get("geomean_speedup")
        iters.append(
            Iteration(
                iter=n,
                title=rec.get("title", f"iter {n}"),
                backlog_item=rec.get("backlog_item", ""),
                outcome=str(rec.get("outcome", rec.get("verdict", "revert"))).lower(),
                verdict=rec.get("verdict", ""),
                geomean_speedup=float(g) if g is not None else None,
                hypothesis=rec.get("hypothesis", ""),
                branch=rec.get("branch", ""),
                commit=rec.get("commit"),
                record=rec,
                bench=bench,
            )
        )
    iters.sort(key=lambda it: it.iter)
    return iters


# ---------------------------------------------------------------------------
# Metric extraction from a bench.py report
# ---------------------------------------------------------------------------


def _cell_label(game: str, num_envs: int, mode: str) -> str:
    return f"ale/{game}/n{num_envs}/sync/{mode}"


def _find_cell(report: dict[str, Any], label: str) -> dict[str, Any] | None:
    for cell in report.get("cells", []):
        if cell.get("label") == label:
            return cell
    return None


def _metric(report: dict[str, Any] | None, label: str, key: str) -> float | None:
    if report is None:
        return None
    cell = _find_cell(report, label)
    if cell is None or "summary" not in cell:
        return None
    return cell["summary"].get(key)


def reference_sps(
    benchmarks: dict[str, Any] | None, game: str, num_envs: int, mode: str
) -> dict[str, float]:
    """SPS of each reference backend at (game, num_envs), for horizontal lines."""
    out: dict[str, float] = {}
    if benchmarks is None:
        return out
    for backend in REF_BACKENDS + ["ale"]:
        label = f"{backend}/{game}/n{num_envs}/sync/{mode}"
        v = _metric(benchmarks, label, "sps_median")
        if v is not None:
            out[backend] = v
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def make_plot(
    iters: list[Iteration],
    benchmarks: dict[str, Any] | None,
    game: str,
    num_envs_list: list[int],
    mode: str,
    out_path: str,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(num_envs_list)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    axes = axes[0]

    for ax, n in zip(axes, num_envs_list):
        label = _cell_label(game, n, mode)
        refs = reference_sps(benchmarks, game, n, mode)

        # ALE baseline (pre-optimization) as iter 0, if present.
        ale0 = refs.get("ale")

        xs_all, ys_all, outcomes = [], [], []
        for it in iters:
            sps = _metric(it.bench, label, "sps_median")
            xs_all.append(it.iter)
            ys_all.append(sps)
            outcomes.append(it.outcome)

        # Running-best frontier: starts at the ALE baseline, only accepted
        # iterations advance it.
        frontier_x, frontier_y = [], []
        best = ale0
        if ale0 is not None:
            frontier_x.append(0)
            frontier_y.append(ale0)
        for it in iters:
            sps = _metric(it.bench, label, "sps_median")
            if it.outcome == "accept" and sps is not None:
                best = sps if best is None else max(best, sps)
            if best is not None:
                frontier_x.append(it.iter)
                frontier_y.append(best)
        if frontier_x:
            ax.step(
                frontier_x, frontier_y, where="post", color="#2ca02c",
                lw=2, label="ALE (accepted frontier)", zorder=3,
            )

        # Every attempt as a marker coloured by outcome.
        for outcome, (color, marker, _lbl) in OUTCOME_STYLE.items():
            xs = [x for x, y, o in zip(xs_all, ys_all, outcomes) if o == outcome and y is not None]
            ys = [y for y, o in zip(ys_all, outcomes) if o == outcome and y is not None]
            if xs:
                ax.scatter(xs, ys, c=color, marker=marker, s=60, zorder=4,
                           label=OUTCOME_STYLE[outcome][2])
            # gate-fail with no bench: mark at bottom so it is visible.
            miss = [x for x, y, o in zip(xs_all, ys_all, outcomes) if o == outcome and y is None]
            for x in miss:
                ax.axvline(x, color=color, ls=":", alpha=0.3, zorder=1)

        # Reference backends as horizontal dashed lines.
        for backend in REF_BACKENDS:
            if backend in refs:
                ax.axhline(refs[backend], ls="--", lw=1.3, color=REF_COLORS[backend],
                           alpha=0.9, label=f"{backend}")
        if ale0 is not None:
            ax.axhline(ale0, ls=":", lw=1.0, color="#2ca02c", alpha=0.5,
                       label="ALE baseline")

        ax.set_title(f"{game}, num_envs={n}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("steps / sec (SPS)")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"ALE vector env autoresearch - throughput over changes ({mode})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"Wrote plot -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def build_table(
    iters: list[Iteration],
    benchmarks: dict[str, Any] | None,
    game: str,
    num_envs_list: list[int],
    mode: str,
) -> tuple[list[str], list[list[str]]]:
    ale0 = {n: _metric(benchmarks, f"ale/{game}/n{n}/sync/{mode}", "sps_median")
            for n in num_envs_list}

    headers = ["iter", "change", "item", "outcome", "verdict", "geomean%"]
    for n in num_envs_list:
        headers.append(f"sps@{n}")
    headers.append(f"p99ms@{num_envs_list[-1]}")
    headers.append(f"majflt@{num_envs_list[-1]}")
    headers.append("cum×vs base")

    rows: list[list[str]] = []
    for it in iters:
        row = [
            str(it.iter),
            it.title,
            it.backlog_item,
            it.outcome,
            it.verdict,
            f"{(it.geomean_speedup - 1) * 100:+.1f}" if it.geomean_speedup else "—",
        ]
        big = num_envs_list[-1]
        for n in num_envs_list:
            sps = _metric(it.bench, _cell_label(game, n, mode), "sps_median")
            row.append(f"{sps:,.0f}" if sps is not None else "—")
        p99 = _metric(it.bench, _cell_label(game, big, mode), "latency_ms_p99_median")
        row.append(f"{p99:.2f}" if p99 is not None else "—")
        majflt = _metric(it.bench, _cell_label(game, big, mode), "majflt_median")
        row.append(f"{majflt:,.0f}" if majflt is not None else "—")
        # Cumulative speedup vs the pre-optimization baseline, geomean across sizes.
        ratios = []
        for n in num_envs_list:
            sps = _metric(it.bench, _cell_label(game, n, mode), "sps_median")
            if sps is not None and ale0.get(n):
                ratios.append(sps / ale0[n])
        if ratios and it.outcome == "accept":
            cum = math.exp(np.mean(np.log(ratios)))
            row.append(f"{cum:.3f}×")
        else:
            row.append("—")
        rows.append(row)
    return headers, rows


def write_markdown(headers: list[str], rows: list[list[str]], path: str) -> None:
    def esc(s: str) -> str:
        return s.replace("|", "\\|")
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(esc(str(c)) for c in r) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote table -> {path}")


def write_csv(headers: list[str], rows: list[list[str]], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
    print(f"Wrote table -> {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ar = os.path.join(repo_root, "autoresearch")

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--records-dir", default=os.path.join(ar, "records"))
    p.add_argument("--results-dir", default=os.path.join(ar, "results"))
    p.add_argument("--benchmarks", default=os.path.join(ar, "results", "benchmarks.json"),
                   help="bench.py report with gym_sync/gym_async/envpool + ale baseline")
    p.add_argument("--out-dir", default=os.path.join(ar, "report"))
    p.add_argument("--game", default="breakout")
    p.add_argument("--num-envs", nargs="+", type=int, default=[8, 64, 256])
    p.add_argument("--autoreset-mode", default="NextStep", choices=["NextStep", "SameStep"])
    p.add_argument("--show", action="store_true", help="also display the plot interactively")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    iters = load_iterations(args.records_dir, args.results_dir)
    if not iters:
        print(f"No iteration records found in {args.records_dir}. "
              "Run the loop first (records are written per iteration).")
        return 1
    benchmarks = _load_json(args.benchmarks)
    if benchmarks is None:
        print(f"NOTE: {args.benchmarks} not found - reference backend lines omitted. "
              "Generate it once with:\n"
              "  bench.py --backends ale gym_sync gym_async envpool "
              f"--num-envs {' '.join(map(str, args.num_envs))} --out {args.benchmarks}")

    make_plot(iters, benchmarks, args.game, args.num_envs, args.autoreset_mode,
              os.path.join(args.out_dir, "progress.png"), args.show)
    headers, rows = build_table(iters, benchmarks, args.game, args.num_envs, args.autoreset_mode)
    write_markdown(headers, rows, os.path.join(args.out_dir, "metrics.md"))
    write_csv(headers, rows, os.path.join(args.out_dir, "metrics.csv"))

    n_acc = sum(1 for it in iters if it.outcome == "accept")
    print(f"\n{len(iters)} iterations, {n_acc} accepted. Report in {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
