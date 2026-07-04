#!/usr/bin/env bash
# Autonomous driver for the ALE vector autoresearch loop (§10.6).
#
# Runs N *independent* headless iterations. Each iteration is a fresh `claude -p`
# invocation so it starts with clean context and re-reads the journal + a fresh
# profile — this is the AIDE/AlphaEvolve pattern, and fresh-context-per-iteration
# is deliberate (a 20-iteration in-session loop would blow the context window and
# let stale reasoning accumulate).
#
# The runner owns only the *outer* guards; the per-iteration work (pick → diff →
# build → gates → bench → compare → commit/revert → journal) lives in SKILL.md,
# which the headless agent reads each time.
#
#   Outer guards enforced here:
#     * protected tree integrity before AND after every iteration (§10.5)
#     * a clean base to branch from
#     * stall detection: stop after too many consecutive non-improvements
#
# Usage:
#   .claude/skills/ale-vector-autoresearch/run_loop.sh --iterations 20
#   ITER=10 ./run_loop.sh                 # env-var form
#
# Env knobs:
#   VENV        python venv to build/run in            (default .venv312)
#   CLAUDE_BIN  claude executable                       (default claude)
#   CLAUDE_PERM permission mode for headless runs       (default acceptEdits)
#   CLAUDE_MODEL model override                          (default: unset)
#   MAX_STALL   stop after this many consecutive REVERT/GATE-FAIL (default 5)
set -euo pipefail

ITERATIONS="${ITER:-10}"
VENV="${VENV:-.venv312}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_PERM="${CLAUDE_PERM:-acceptEdits}"
MAX_STALL="${MAX_STALL:-5}"
EXTRA_BENCH_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --max-stall)  MAX_STALL="$2"; shift 2 ;;
    --venv)       VENV="$2"; shift 2 ;;
    --bench-args) EXTRA_BENCH_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
GUARD="$SKILL_DIR/check-protected.sh"

log() { printf '\n\033[1;36m[loop %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }

# --- preflight ------------------------------------------------------------
if [[ -n "$(git status --porcelain -- src ':!src/ale/vector' ':!src/ale/python/*vector*')" ]]; then
  echo "aborting: source outside the optimisation surface is uncommitted. Commit the" >&2
  echo "harness + any baseline work first so each iteration branches from a clean base" >&2
  echo "and the guard can diff the agent's changes against it." >&2
  exit 1
fi
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
"$GUARD" "$(git rev-parse HEAD)" || { echo "aborting: protection guard failed at start" >&2; exit 1; }
log "base branch = $BASE_BRANCH @ $(git rev-parse --short HEAD)"

# Reference benchmarks (gym_sync/gym_async/envpool + ALE baseline) — generated
# once so analyse.py can draw them as fixed lines. Skip if already present.
BENCHMARKS="autoresearch/results/benchmarks.json"
if [[ ! -f "$BENCHMARKS" ]]; then
  log "generating reference benchmarks (gym_sync/gym_async/envpool/ale) -> $BENCHMARKS"
  mkdir -p autoresearch/results
  "$VENV/bin/python" autoresearch/bench.py \
      --backends ale gym_sync gym_async envpool \
      --num-envs 8 64 256 --repeats 8 --out "$BENCHMARKS" \
    || log "WARNING: reference benchmark run failed (missing gym/envpool?) — plots will omit lines"
fi

stall=0
for ((i = 1; i <= ITERATIONS; i++)); do
  START_SHA="$(git rev-parse "$BASE_BRANCH")"
  "$GUARD" "$START_SHA" || { echo "aborting: protection guard failed before iteration $i" >&2; exit 1; }

  read -r -d '' PROMPT <<EOF || true
Run exactly ONE iteration of the ALE vector autoresearch loop defined in
.claude/skills/ale-vector-autoresearch/SKILL.md (iteration $i of $ITERATIONS).

Follow the skill's per-iteration procedure end to end: read autoresearch/JOURNAL.md,
pick the single highest-priority not-done backlog item, implement it as a small diff
on a fresh branch named autoresearch/iter-<n>-<slug>, build with '$VENV/bin/pip install .',
run 'verify.py all' as a hard gate, and only if it passes benchmark with bench.py
$EXTRA_BENCH_ARGS and judge with compare.py against autoresearch/results/baseline.json.
Commit on ACCEPT (and refresh the baseline), otherwise revert. Append one entry to
autoresearch/JOURNAL.md either way. Never edit any file under the protected manifest.

Finish your reply with a single final line, exactly one of:
  RESULT=ACCEPT   RESULT=REVERT   RESULT=GATE-FAIL   RESULT=STOP
Use RESULT=STOP only if the backlog is exhausted or you cannot proceed safely.
EOF

  log "iteration $i/$ITERATIONS — launching headless agent"
  CLAUDE_ARGS=(-p "$PROMPT" --permission-mode "$CLAUDE_PERM")
  [[ -n "${CLAUDE_MODEL:-}" ]] && CLAUDE_ARGS+=(--model "$CLAUDE_MODEL")

  OUT="$("$CLAUDE_BIN" "${CLAUDE_ARGS[@]}" 2>&1)" || true
  echo "$OUT" | tail -n 40
  RESULT="$(printf '%s\n' "$OUT" | grep -oE 'RESULT=(ACCEPT|REVERT|GATE-FAIL|STOP)' | tail -n1 | cut -d= -f2)"
  RESULT="${RESULT:-UNKNOWN}"
  log "iteration $i result: $RESULT"

  # Post-iteration integrity: the agent must not have touched anything outside
  # the optimisation surface, nor the hash-pinned harness/goldens.
  if ! "$GUARD" "$START_SHA"; then
    echo "PROTECTION violated during iteration $i — stopping for human review." >&2
    echo "Inspect branch $(git rev-parse --abbrev-ref HEAD); the harness is hash-pinned" >&2
    echo "so any oracle/bench edit is recoverable from autoresearch/protected.sha256." >&2
    exit 1
  fi

  # Return to a clean base branch for the next iteration.
  git checkout "$BASE_BRANCH" >/dev/null 2>&1 || true

  case "$RESULT" in
    ACCEPT)          stall=0 ;;
    REVERT|GATE-FAIL) stall=$((stall + 1)) ;;
    STOP)            log "agent signalled STOP — backlog exhausted or blocked"; break ;;
    *)               stall=$((stall + 1)); log "no RESULT line parsed — counting as stall" ;;
  esac

  if (( stall >= MAX_STALL )); then
    log "stopping: $stall consecutive non-improvements (>= MAX_STALL=$MAX_STALL)"
    break
  fi
done

# --- blog-ready report ----------------------------------------------------
log "building analysis report (plot + metrics table)"
"$VENV/bin/python" autoresearch/analyse.py --num-envs 8 64 256 \
  || log "WARNING: analyse.py failed (no records yet?)"

log "loop finished."
log "  journal:  autoresearch/JOURNAL.md"
log "  report:   autoresearch/report/progress.png + metrics.md + metrics.csv"
log "  accepted commits:"
git log --oneline "$BASE_BRANCH" | grep -i autoresearch | head -n 20 || true