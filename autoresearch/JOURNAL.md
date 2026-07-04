# ALE Vector Autoresearch — Journal

Running log of the optimization loop (§10.6 of `ale_vector_optimization.md`). One
entry per iteration, **appended whether the change is accepted or reverted** —
the reverts are the ablation data. Newest entries at the top of the log.

The seed journal is `ale_vector_optimization.md` itself: §5/§9 are the prioritized
backlog, §10 is the loop contract. Work the §9 priority table top-down until it is
exhausted, then switch to profile-driven exploration.

## Contract (do not violate)

- **Editable surface**: only `src/ale/vector/*` and `src/ale/python/*vector*`.
  Everything else — emulator core, game logic, single-env path, top-level build,
  tests — is off-limits. Enforced by `check-protected.sh` (git allowlist vs the
  base commit).
- **Hash-pinned harness** (never edit; `protected.sha256`): `verify.py`,
  `compare.py`, `bench.py`, `analyse.py`, `goldens/`. A *legitimate* semantics
  change makes the goldens fail by construction — that routes to a human for
  deliberate golden regeneration, never a reason to touch the oracle.
- **Gates before timing**: `verify.py all` must PASS before any benchmark number
  counts. Perf on a broken build is auto-reject.
- **One hypothesis per iteration**, on a branch, small diff, greedy hill-climb
  with revert. `compare.py` exit code is the verdict.
- **Record every iteration** (accept and revert): a `JOURNAL.md` entry + a
  `records/iterNN.json` (schema in SKILL.md) so `analyse.py` can build the writeup.

## Benchmark config

- `num_envs ∈ {8, 64, 256}`, game breakout, NextStep, sync.
- **North star = SPS** (steps/sec). Accept/reject on SPS geomean + per-cell
  regression guard only. Latency p50/p99, page-faults, RSS are recorded for
  *understanding*, not for the verdict.

## Baseline

- Goldens generated @ `bfb82d80` (config: breakout, 8 envs, 60 calls, seed 12345,
  max_frames=120 forcing a mid-window reset; 8 resets captured in window).
- Harness: `bench.py` (matrix + JSON + fingerprint), `verify.py` (sync/async/
  differential gates), `compare.py` (bootstrap ratio-of-medians verdict),
  `analyse.py` (progress plot + metrics table vs gym/envpool references).

## Backlog (from §9 priority table — pick the top not-yet-done item)

| # | Change | § | Kind | Status |
|---|--------|---|------|--------|
| 1 | Worker-error deadlock fix | 3.1 | correctness | todo |
| 2 | Stale terminal observation fix | 3.2 | correctness | todo (NOTE: memory says §3.2 stale terminal obs is intentional Gym-matching — confirm before touching) |
| 3 | `step_into` external outputs | 5.2 | alloc+copy | todo |
| 4 | Refcount buffer pool for `recv()` | 4.2 | allocation | todo |
| 5 | `cv::setNumThreads(0)` | 8.1 | oversubscription | todo |
| 6 | Thread defaults: physical−1, main-thread placement | 5.5 | scheduling | todo |
| 10 | PGO / LTO / `-march` | 8.2 | build | todo |
| 11 | Chunked dequeue | 5.3 | queue | todo |
| 12 | 2-memcpy linearization + SameStep 2× | 5.4 | copy | todo |
| 13 | Opt-out info dict, `nb::ndarray` args, fused step | 5.7 | python | todo |
| 14 | False-sharing padding | 5.6 | cache | todo |

(§7 training-loop items and §5.2 API signatures are human-owned per §10.4.)

---

## Log

<!-- Template — copy for each iteration:

### YYYY-MM-DD  iter NN — <one-line hypothesis>  [ACCEPT|REVERT|GATE-FAIL]
- **Backlog item:** §x.y <name>
- **Branch:** autoresearch/iter-NN-<slug>   **Diff:** <files touched, ~N lines>
- **Hypothesis:** <what bottleneck, from which profile signal>
- **Gates:** sync/async/differential = PASS|FAIL (<detail if fail>)
- **Bench:** baseline vs candidate — geomean <±%>, CI [<lo>,<hi>], regressions: <list>
- **Verdict:** <compare.py verdict + reason>
- **Outcome:** committed <sha> | reverted
- **Notes / next:** <what the numbers taught us; what to try next>

-->

_(no iterations yet — run `/ale-vector-autoresearch` to begin)_