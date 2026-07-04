---
name: ale-vector-autoresearch
description: >-
  Run the ALE vector-environment optimization loop: profile-guided, one
  hypothesis at a time, gated by a bit-exact correctness oracle and a
  noise-aware benchmark verdict, with automatic commit-or-revert, a structured
  journal, and blog-ready plots/tables. Use when asked to optimize / speed up
  the C++ vector env (src/ale/vector/), run the autoresearch loop, or do an
  autonomous performance-tuning run. A bare invocation runs ONE iteration
  in-session; "autonomous N" launches N unattended headless iterations via
  run_loop.sh.
---

# ALE Vector Autoresearch Loop

Executes the optimization loop specified in
`autoresearch/ale_vector_optimization.md` §10. It turns "make the vector env
faster" into a disciplined, reversible, one-hypothesis-per-step search that
cannot silently break correctness or accept noise, and it records every step so
the run can be written up afterwards.

The seed document is the master spec: §5/§9 are the backlog, §10 is the loop
contract, §10.7 defines the records + analysis. `autoresearch/JOURNAL.md` is the
human-readable running log — read it first, every time.

## Benchmark configuration (fixed)

- **Env counts:** `num_envs ∈ {8, 64, 256}` — small (Python/binding-overhead
  dominated), mid, and large (emulation + bandwidth dominated).
- **North-star metric:** environment **steps per second (SPS)**. Accept/reject is
  decided on SPS alone (geomean across the three env counts, hard per-cell
  regression guard). Every other metric — fps, p50/p99 step latency, page-faults
  (`minflt`/`majflt`), RSS — is recorded to *explain* an SPS move, never to
  decide it.
- Default game `breakout`, `NextStep` autoreset, sync (`batch_size == num_envs`).

## Invariants (never violate)

1. **Editable surface = `src/ale/vector/*` and `src/ale/python/*vector*` only.**
   The loop may not touch the emulator core, game logic, the single-env path,
   top-level build files, tests, or anything else. `check-protected.sh` enforces
   this (git allowlist against the base commit) and also **hash-pins the harness**
   (`verify.py`, `bench.py`, `compare.py`, `analyse.py`, `goldens/`) — the
   reward-hacking surface (§10.5). Run it at the start and end of every iteration.
   If a legitimate semantics change makes the goldens fail, that is *by design* —
   stop and surface it for deliberate human golden regeneration; never "fix" it by
   editing the oracle.
2. **Gates before timing.** `verify.py all` must PASS before any benchmark number
   is collected. A perf win on a broken build is an auto-revert.
3. **One hypothesis per iteration.** Small diff, on a branch. Greedy hill-climb
   with revert — `compare.py`'s exit code is the verdict, not your judgement.
4. **Record every iteration** — accepted *and* reverted. Write both the
   human `JOURNAL.md` entry and the machine `records/iterNN.json` (schema below).
   The reverts are the ablation data and the blog's honesty.

## Commands (this repo, `.venv312`)

```bash
VENV=.venv312
GUARD=.claude/skills/ale-vector-autoresearch/check-protected.sh
BASE=$(git rev-parse <base-branch>)                        # the pre-iteration HEAD

$GUARD "$BASE"                                             # protection guard
$VENV/bin/pip install .                                    # build (~1-3 min)
$VENV/bin/python autoresearch/verify.py all \              # HARD GATE
    --goldens autoresearch/goldens/goldens.npz
$VENV/bin/python autoresearch/bench.py --backends ale \    # measure candidate
    --num-envs 8 64 256 --repeats 8 \
    --out autoresearch/results/iter<NN>_bench.json
$VENV/bin/python autoresearch/compare.py \                 # VERDICT (exit 0=accept)
    autoresearch/results/baseline.json \
    autoresearch/results/iter<NN>_bench.json \
    --out autoresearch/results/iter<NN>_verdict.json
$VENV/bin/python autoresearch/analyse.py                   # rebuild plot + tables
```

`--repeats 8` minimum for real verdicts. `bench.py --quick` (2 repeats) is a smoke
test only — its noise floor swamps any change worth making, so `compare.py` will
correctly reject on quick data.

## Two ways to run

### A. One iteration, in-session (supervised)
Invoked as `/ale-vector-autoresearch`. Do exactly one pass of the procedure below,
then stop and report. Good for building trust or debugging a specific item.

### B. Autonomous, many iterations (unattended)
Invoked as `/ale-vector-autoresearch autonomous <N>`. Launch the headless driver
(fresh context per iteration is intentional — it re-reads the journal and a fresh
profile each time) and tell the user how to watch it:

```bash
.claude/skills/ale-vector-autoresearch/run_loop.sh --iterations <N>
# monitor: tail -f autoresearch/JOURNAL.md ; open autoresearch/report/progress.png
```

The runner owns the outer guards (protection before/after each iteration, a clean
base, stall-stop after `MAX_STALL` consecutive non-improvements), generates the
reference `benchmarks.json` once, calls `claude -p` per iteration, and builds the
`analyse.py` report at the end. Do **not** loop N times inside one session — that
defeats the fresh-context design and exhausts the context window.

## Per-iteration procedure

0. **Integrity + orientation.** `check-protected.sh "$BASE"` (abort on failure).
   Read `JOURNAL.md` and the §9 backlog. Note the base branch/SHA.
1. **Ensure a baseline.** If `autoresearch/results/baseline.json` is missing or
   its `fingerprint.git_sha` ≠ current HEAD, regenerate it (`bench.py --backends
   ale --num-envs 8 64 256 --repeats 8 --out results/baseline.json`). Note the
   **noise floor**: if any cell's `sps_rel_spread` > ~10%, record it — sub-5%
   verdicts are untrustworthy until the machine is quieted (§10.2).
2. **Pick ONE hypothesis.** Highest-priority not-done backlog item, ideally
   pointed to by a fresh profile signal (perf top-N, off-CPU, page-faults — §2).
   Correctness fixes (§3.1) first. Re §3.2: the "stale terminal obs" behavior is
   intentional Gym-matching (see memory) — confirm before changing it.
3. **Branch + implement.** `git checkout -b autoresearch/iter-<NN>-<slug>`.
   Smallest diff that tests the hypothesis. Touch **only** `src/ale/vector/*` and
   `src/ale/python/*vector*`.
4. **Build.** `$VENV/bin/pip install .`. Compile error → fix in scope or revert.
5. **GATE.** `verify.py all`. Any FAIL/ERROR → record the first-divergence line,
   revert the branch, write the record with `outcome:"gate-fail"`, end
   `RESULT=GATE-FAIL`. Never proceed to timing.
6. **Benchmark.** `bench.py --backends ale --num-envs 8 64 256 --repeats 8
   --out results/iter<NN>_bench.json`.
7. **Verdict.** `compare.py baseline.json results/iter<NN>_bench.json
   --out results/iter<NN>_verdict.json`. Exit 0 = ACCEPT, 1 = REJECT.
8. **Commit or revert.**
   - **ACCEPT:** commit on the branch, fast-forward base, then refresh
     `baseline.json` (`cp results/iter<NN>_bench.json results/baseline.json`).
   - **REJECT:** revert the branch.
9. **Record (both forms).**
   - Append a `JOURNAL.md` entry (template at the bottom of that file).
   - Write `autoresearch/records/iter<NN>.json` (schema below).
10. **Integrity + signal.** `check-protected.sh "$BASE"` again. Optionally
    `analyse.py` to refresh the report. End the reply with a single final line:
    `RESULT=ACCEPT` | `RESULT=REVERT` | `RESULT=GATE-FAIL` | `RESULT=STOP`.

## Record schema — `autoresearch/records/iter<NN>.json`

`analyse.py` consumes these. Keep keys stable; unknown keys are ignored, missing
optional keys degrade gracefully.

```json
{
  "iter": 3,
  "timestamp": "2026-07-04T16:40:00",
  "title": "Disable OpenCV internal threading",
  "backlog_item": "§8.1",
  "hypothesis": "cv::resize spawns a TBB pool that fights env workers; off-CPU profile showed futex churn",
  "branch": "autoresearch/iter-03-opencv-threads",
  "base_commit": "bfb82d80",
  "commit": "def456",                 // null when reverted / gate-fail
  "outcome": "accept",                // accept | revert | gate-fail
  "verdict": "ACCEPT",                // compare.py verdict (empty on gate-fail)
  "verdict_reason": "geomean +3.1% (CI [+1.2,+4.9]) clears +0.5% with no regressions",
  "geomean_speedup": 1.031,           // null on gate-fail
  "geomean_ci": [1.012, 1.049],
  "diff_stat": "1 file, +2 -0",
  "gates": {"sync": true, "async": true, "differential": true},
  "profile_note": "off-CPU futex time 12% -> 1%",  // the evidence the SPS move is real
  "bench_candidate": "results/iter03_bench.json",  // path analyse.py reads metrics from
  "verdict_json": "results/iter03_verdict.json",
  "notes": "next: thread affinity §5.5"
}
```

## Records & analysis (blog artefacts)

- `records/iterNN.json` + `results/iterNN_bench.json` + `results/iterNN_verdict.json`
  — the full, machine-readable trail of every attempt.
- `results/benchmarks.json` — one-off `bench.py` report with `gym_sync`,
  `gym_async`, `envpool` and the ALE baseline (the runner generates it; regenerate
  by hand with `bench.py --backends ale gym_sync gym_async envpool --num-envs 8 64
  256 --repeats 8 --out autoresearch/results/benchmarks.json`).
- `analyse.py` → `report/progress.png` (SPS over iterations, one panel per env
  count, ALE accepted-frontier + attempt markers + gym/envpool reference lines)
  and `report/metrics.{md,csv}` (one row per change, metric columns). These are
  the figure and table for the writeup.

## When to stop / escalate

- Goldens fail on a change you believe is *correct* → STOP; a human regenerates
  goldens from the new reference and re-pins `protected.sha256`.
- API-signature / training-loop work (§5.2 `step_into`/`out=`, §7, frame-stack
  dedup contracts) is human-owned per §10.4 — implement only against a written
  spec, else STOP and ask. (Note: §8.2 build-flag work touches top-level CMake,
  which is **outside** the editable surface — human-owned.)
- Noise floor too high to resolve the effect → STOP and flag machine hygiene.

## Files

| Path | Role |
|------|------|
| `autoresearch/ale_vector_optimization.md` | master spec / seed journal (§5,§9 backlog; §10 contract) |
| `autoresearch/JOURNAL.md` | human running log — read first, append every iteration |
| `autoresearch/verify.py` | **hash-pinned** correctness oracle (sync/async/differential) |
| `autoresearch/bench.py` | **hash-pinned** benchmark (matrix + JSON + fingerprint) |
| `autoresearch/compare.py` | **hash-pinned** paired verdict (bootstrap; exit 0=accept) |
| `autoresearch/analyse.py` | **hash-pinned** plot + table generator |
| `autoresearch/goldens/` | **hash-pinned** golden blobs |
| `autoresearch/protected.sha256` | integrity manifest for the harness |
| `autoresearch/records/` | per-iteration record JSON (blog trail) |
| `autoresearch/results/` | baseline.json, benchmarks.json, per-iter bench/verdict |
| `autoresearch/report/` | analyse.py output: progress.png, metrics.md/.csv |
| `.claude/skills/ale-vector-autoresearch/check-protected.sh` | allowlist + hash-pin guard |
| `.claude/skills/ale-vector-autoresearch/run_loop.sh` | autonomous headless driver |
