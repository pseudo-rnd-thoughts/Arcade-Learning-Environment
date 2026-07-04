#!/usr/bin/env python3
"""Paired accept/reject verdict for two ``bench.py`` reports.

This is the ``compare.py`` third of the autoresearch harness trio described in
``autoresearch/ale_vector_optimization.md`` (§10.2/§10.6). ``bench.py`` produces
a JSON report with per-cell, per-repeat throughput; ``compare.py`` takes a
*baseline* report and a *candidate* report and renders a single machine-readable
verdict that the autoresearch loop uses to decide **commit or revert**.

Why this exists
---------------
Most items in §5/§9 are worth 2-10%. On a machine with a 5-15% noise floor a
raw "candidate median > baseline median" comparison is a coin flip. So the
verdict is built to beat noise:

* **Per-cell effect** = ratio of medians ``median(candidate) / median(baseline)``,
  with a bootstrap 95% CI on that ratio. A cell only counts as changed if 1.0
  falls outside its CI (i.e. the effect is bigger than the measured spread).
* **Hard regression guard** (§10.2): if *any* non-holdout cell regresses beyond
  ``--regress-tol`` (default 2%) *and* that regression is significant, the whole
  candidate is REJECTED regardless of wins elsewhere. Holdout cells guard against
  overfitting to the tuned configs and must also not significantly regress.
* **Objective** = geometric mean of per-cell speedups across the (non-holdout)
  matrix, with its own bootstrap CI. ACCEPT requires the geomean CI lower bound
  to clear ``1 + --accept-gain`` so a net win must be real, not drift.

Only cells present in **both** reports and sharing a backend/config are compared;
by default only ``ale`` cells (the optimisation target) drive the verdict, the
other backends in a report are context.

Usage
-----
    python autoresearch/compare.py baseline.json candidate.json
    python autoresearch/compare.py base.json cand.json --out verdict.json
    python autoresearch/compare.py base.json cand.json --holdout \
        'ale/seaquest/n256/sync/NextStep'

Exit code: 0 = ACCEPT, 1 = REJECT, 2 = usage/data error. So the loop can branch
on ``$?`` directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Verdict thresholds (all overridable on the CLI)
# ---------------------------------------------------------------------------

DEFAULT_REGRESS_TOL = 0.02   # a cell losing >2% (significantly) is a hard reject
DEFAULT_ACCEPT_GAIN = 0.005  # geomean must clear +0.5% at its CI lower bound
DEFAULT_BOOTSTRAP = 10000
DEFAULT_CI = 0.95
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _sps_samples(cell: dict[str, Any]) -> np.ndarray:
    """Per-repeat SPS values for a cell, or empty if the cell has no data."""
    reps = cell.get("repeats") or []
    return np.array([r["sps"] for r in reps if "sps" in r], dtype=np.float64)


def _bootstrap_ratio(
    base: np.ndarray, cand: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Point estimate and CI for ``median(cand) / median(base)``.

    Base and candidate repeats are independent samples (separate bench runs), so
    they are resampled independently. Returns (ratio, lo, hi) at the given CI.
    """
    point = float(np.median(cand) / np.median(base))
    if len(base) < 2 or len(cand) < 2:
        # Not enough repeats to bootstrap; report point estimate with no width.
        return point, point, point
    bi = rng.integers(0, len(base), size=(n_boot, len(base)))
    ci_ = rng.integers(0, len(cand), size=(n_boot, len(cand)))
    med_base = np.median(base[bi], axis=1)
    med_cand = np.median(cand[ci_], axis=1)
    ratios = med_cand / med_base
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(ratios, alpha))
    hi = float(np.quantile(ratios, 1.0 - alpha))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Per-cell comparison
# ---------------------------------------------------------------------------


@dataclass
class CellVerdict:
    label: str
    backend: str
    holdout: bool
    n_base: int
    n_cand: int
    base_median_sps: float
    cand_median_sps: float
    ratio: float          # cand / base (>1 == faster)
    ratio_lo: float
    ratio_hi: float
    noise_floor: float     # max relative spread of the two cells
    significant: bool      # CI excludes 1.0
    verdict: str           # IMPROVE / REGRESS / NEUTRAL

    def as_row(self) -> str:
        pct = (self.ratio - 1.0) * 100.0
        star = "*" if self.significant else " "
        tag = " [holdout]" if self.holdout else ""
        return (
            f"{self.label:<48} {self.base_median_sps:>12,.0f} -> "
            f"{self.cand_median_sps:>12,.0f}  {pct:>+6.1f}%{star} "
            f"[{(self.ratio_lo - 1) * 100:+.1f},{(self.ratio_hi - 1) * 100:+.1f}]  "
            f"{self.verdict}{tag}"
        )


def _cell_noise(base_cell: dict, cand_cell: dict) -> float:
    def spread(c):
        s = c.get("summary") or {}
        return s.get("sps_rel_spread")
    vals = [v for v in (spread(base_cell), spread(cand_cell)) if v is not None]
    return max(vals) if vals else float("nan")


def compare_cell(
    label: str,
    base_cell: dict,
    cand_cell: dict,
    holdout: bool,
    regress_tol: float,
    improve_gain: float,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> CellVerdict | None:
    base = _sps_samples(base_cell)
    cand = _sps_samples(cand_cell)
    if len(base) == 0 or len(cand) == 0:
        return None

    ratio, lo, hi = _bootstrap_ratio(base, cand, n_boot, ci, rng)
    significant = not (lo <= 1.0 <= hi)

    if ratio < 1.0 - regress_tol and hi < 1.0:
        verdict = "REGRESS"
    elif ratio > 1.0 + improve_gain and lo > 1.0:
        verdict = "IMPROVE"
    else:
        verdict = "NEUTRAL"

    return CellVerdict(
        label=label,
        backend=(base_cell.get("config") or {}).get("backend", "?"),
        holdout=holdout,
        n_base=len(base),
        n_cand=len(cand),
        base_median_sps=float(np.median(base)),
        cand_median_sps=float(np.median(cand)),
        ratio=ratio,
        ratio_lo=lo,
        ratio_hi=hi,
        noise_floor=_cell_noise(base_cell, cand_cell),
        significant=significant,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------


@dataclass
class Report:
    verdict: str                     # ACCEPT / REJECT
    reason: str
    geomean_speedup: float
    geomean_lo: float
    geomean_hi: float
    n_cells: int
    n_regressions: int
    n_improvements: int
    regressions: list[str]
    cells: list[dict] = field(default_factory=list)
    baseline_fingerprint: dict = field(default_factory=dict)
    candidate_fingerprint: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _geomean_ci(
    cell_samples: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap CI for the geomean of per-cell median-ratios across the matrix.

    Resamples each cell's base/cand repeats independently and jointly propagates
    the uncertainty into the geomean, so the objective's CI reflects every cell's
    noise at once.
    """
    point_ratios = [np.median(c) / np.median(b) for b, c in cell_samples]
    point = float(math.exp(np.mean(np.log(point_ratios))))
    if any(len(b) < 2 or len(c) < 2 for b, c in cell_samples):
        return point, point, point

    logs = np.zeros((n_boot, len(cell_samples)), dtype=np.float64)
    for j, (b, c) in enumerate(cell_samples):
        bi = rng.integers(0, len(b), size=(n_boot, len(b)))
        ci_ = rng.integers(0, len(c), size=(n_boot, len(c)))
        logs[:, j] = np.log(np.median(c[ci_], axis=1) / np.median(b[bi], axis=1))
    geo = np.exp(logs.mean(axis=1))
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(geo, alpha)), float(np.quantile(geo, 1.0 - alpha))


def _index_cells(report: dict) -> dict[str, dict]:
    return {c["label"]: c for c in report.get("cells", [])}


def build_report(
    baseline: dict,
    candidate: dict,
    holdouts: set[str],
    backends: set[str] | None,
    regress_tol: float,
    accept_gain: float,
    improve_gain: float,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> Report:
    base_cells = _index_cells(baseline)
    cand_cells = _index_cells(candidate)
    warnings: list[str] = []

    common = [lbl for lbl in base_cells if lbl in cand_cells]
    if not common:
        return Report(
            verdict="REJECT",
            reason="no cells shared between baseline and candidate reports",
            geomean_speedup=float("nan"),
            geomean_lo=float("nan"),
            geomean_hi=float("nan"),
            n_cells=0,
            n_regressions=0,
            n_improvements=0,
            regressions=[],
            warnings=["disjoint cell sets - was the same config matrix used?"],
        )

    cell_verdicts: list[CellVerdict] = []
    objective_samples: list[tuple[np.ndarray, np.ndarray]] = []
    for lbl in common:
        bcfg = (base_cells[lbl].get("config") or {}).get("backend")
        if backends is not None and bcfg not in backends:
            continue
        cv = compare_cell(
            lbl,
            base_cells[lbl],
            cand_cells[lbl],
            holdout=lbl in holdouts,
            regress_tol=regress_tol,
            improve_gain=improve_gain,
            n_boot=n_boot,
            ci=ci,
            rng=rng,
        )
        if cv is None:
            warnings.append(f"cell {lbl!r} missing repeat data in one report; skipped")
            continue
        if not math.isnan(cv.noise_floor) and cv.noise_floor > 0.10:
            warnings.append(
                f"cell {lbl!r} noise floor {cv.noise_floor * 100:.0f}% > 10% "
                "- verdict on small effects is unreliable (§10.2 machine hygiene)"
            )
        cell_verdicts.append(cv)
        if not cv.holdout:
            objective_samples.append(
                (_sps_samples(base_cells[lbl]), _sps_samples(cand_cells[lbl]))
            )

    if not objective_samples:
        return Report(
            verdict="REJECT",
            reason="no non-holdout cells with data to form an objective",
            geomean_speedup=float("nan"),
            geomean_lo=float("nan"),
            geomean_hi=float("nan"),
            n_cells=len(cell_verdicts),
            n_regressions=0,
            n_improvements=0,
            regressions=[],
            cells=[asdict(c) for c in cell_verdicts],
            warnings=warnings,
        )

    geo, geo_lo, geo_hi = _geomean_ci(objective_samples, n_boot, ci, rng)

    regressions = [c.label for c in cell_verdicts if c.verdict == "REGRESS"]
    improvements = [c.label for c in cell_verdicts if c.verdict == "IMPROVE"]

    # Decision: hard regression guard first, then the objective must clear noise.
    if regressions:
        verdict = "REJECT"
        reason = f"{len(regressions)} cell(s) significantly regressed >{regress_tol * 100:.0f}%: " + \
                 ", ".join(regressions)
    elif geo_lo <= 1.0 + accept_gain:
        verdict = "REJECT"
        reason = (
            f"geomean speedup {(geo - 1) * 100:+.2f}% (CI lower {(geo_lo - 1) * 100:+.2f}%) "
            f"does not clear the +{accept_gain * 100:.1f}% acceptance bar"
        )
    else:
        verdict = "ACCEPT"
        reason = (
            f"geomean speedup {(geo - 1) * 100:+.2f}% "
            f"(CI [{(geo_lo - 1) * 100:+.2f}%, {(geo_hi - 1) * 100:+.2f}%]) clears "
            f"+{accept_gain * 100:.1f}% with no significant regressions"
        )

    return Report(
        verdict=verdict,
        reason=reason,
        geomean_speedup=geo,
        geomean_lo=geo_lo,
        geomean_hi=geo_hi,
        n_cells=len(cell_verdicts),
        n_regressions=len(regressions),
        n_improvements=len(improvements),
        regressions=regressions,
        cells=[asdict(c) for c in cell_verdicts],
        baseline_fingerprint=baseline.get("fingerprint", {}),
        candidate_fingerprint=candidate.get("fingerprint", {}),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def print_report(report: Report) -> None:
    print("\n" + "=" * 100)
    print(f"{'cell':<48} {'baseline':>12}    {'candidate':>12}   {'Δ%':>7}   {'95% CI':>14}   verdict")
    print("-" * 100)
    order = {"REGRESS": 0, "IMPROVE": 1, "NEUTRAL": 2}
    for c in sorted(report.cells, key=lambda d: (order.get(d["verdict"], 3), d["label"])):
        cv = CellVerdict(**c)
        print(cv.as_row())
    print("=" * 100)
    print(f"cells compared: {report.n_cells}   improvements: {report.n_improvements}   "
          f"regressions: {report.n_regressions}")
    print(f"geomean speedup: {(report.geomean_speedup - 1) * 100:+.2f}%  "
          f"CI [{(report.geomean_lo - 1) * 100:+.2f}%, {(report.geomean_hi - 1) * 100:+.2f}%]")

    bfp = report.baseline_fingerprint or {}
    cfp = report.candidate_fingerprint or {}
    if bfp.get("git_sha") or cfp.get("git_sha"):
        print(f"baseline  @ {str(bfp.get('git_sha'))[:12]} ({bfp.get('git_branch')})"
              f"  dirty={bfp.get('git_dirty')}")
        print(f"candidate @ {str(cfp.get('git_sha'))[:12]} ({cfp.get('git_branch')})"
              f"  dirty={cfp.get('git_dirty')}")
    if bfp.get("cpu_model") and cfp.get("cpu_model") and bfp["cpu_model"] != cfp["cpu_model"]:
        report.warnings.append(
            "baseline and candidate ran on different CPUs - comparison is invalid"
        )
    for w in report.warnings:
        print(f"  WARNING: {w}")

    print("\n" + "*" * 100)
    print(f"VERDICT: {report.verdict}")
    print(f"  {report.reason}")
    print("*" * 100)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("baseline", help="bench.py JSON report for the reference build")
    p.add_argument("candidate", help="bench.py JSON report for the candidate build")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["ale"],
        help="only these backends drive the verdict (default: ale). "
        "Pass 'all' to compare every backend present.",
    )
    p.add_argument(
        "--holdout",
        nargs="*",
        default=[],
        help="cell label(s) excluded from the objective geomean but still "
        "regression-guarded (§10.2 hold-out).",
    )
    p.add_argument("--regress-tol", type=float, default=DEFAULT_REGRESS_TOL,
                   help="a significant loss beyond this fraction hard-rejects (default 0.02)")
    p.add_argument("--accept-gain", type=float, default=DEFAULT_ACCEPT_GAIN,
                   help="geomean CI lower bound must exceed 1+this to ACCEPT (default 0.005)")
    p.add_argument("--improve-gain", type=float, default=DEFAULT_ACCEPT_GAIN,
                   help="per-cell ratio above 1+this (and significant) counts as IMPROVE")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--ci", type=float, default=DEFAULT_CI)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=None, help="write the machine-readable verdict JSON here")
    args = p.parse_args(argv)

    try:
        baseline = _load(args.baseline)
        candidate = _load(args.candidate)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not load reports: {exc}", file=sys.stderr)
        return 2

    backends: set[str] | None = None if "all" in args.backends else set(args.backends)
    rng = np.random.default_rng(args.seed)

    report = build_report(
        baseline,
        candidate,
        holdouts=set(args.holdout),
        backends=backends,
        regress_tol=args.regress_tol,
        accept_gain=args.accept_gain,
        improve_gain=args.improve_gain,
        n_boot=args.bootstrap,
        ci=args.ci,
        rng=rng,
    )

    print_report(report)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(asdict(report), fh, indent=2)
        print(f"\nWrote verdict -> {args.out}")

    if report.verdict == "ACCEPT":
        return 0
    if report.n_cells == 0:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
