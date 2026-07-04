#!/usr/bin/env bash
# Protection guard for the autoresearch loop (§10.5). Two layers:
#
#   1. ALLOWLIST — the loop may only modify source under:
#          src/ale/vector/*          (the C++ vectorizer + its CMake)
#          src/ale/python/*vector*   (the nanobind vector + XLA interfaces, vector_env.py)
#      Any *other* tracked file that differs from the base is a violation. This is
#      the primary guard: it keeps the agent inside the optimisation surface and
#      out of the emulator core, game logic, the single-env path, build config, etc.
#
#   2. HASH-PIN — the harness itself (oracle, benchmark, verdict, analysis, goldens)
#      is untracked data that git's allowlist can't see, so it is pinned in
#      protected.sha256. These files are the reward-hacking surface: an agent that
#      can edit verify.py/bench.py/compare.py/goldens can fake correctness or game
#      the timer. They must never change during the loop.
#
# Append-only records (autoresearch/records|results|report) are the loop's own
# output and are allowed to appear/grow.
#
# Usage:  check-protected.sh [BASE_REF]
#   BASE_REF given  -> also diff tracked files against it (used by run_loop.sh with
#                      the base-branch SHA, so a committed candidate branch is checked).
#   BASE_REF absent -> allowlist checks uncommitted changes vs HEAD only.
#
# Exit: 0 clean · 1 violation · 2 manifest missing.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
BASE_REF="${1:-${AUTORESEARCH_BASE:-HEAD}}"

rc=0

# --- Layer 1: allowlist on tracked source ---------------------------------
is_allowed() {
  case "$1" in
    src/ale/vector/*)        return 0 ;;
    src/ale/python/*vector*) return 0 ;;
    autoresearch/records/*)  return 0 ;;
    autoresearch/results/*)  return 0 ;;
    autoresearch/report/*)   return 0 ;;
    autoresearch/JOURNAL.md) return 0 ;;
    *)                       return 1 ;;
  esac
}

violations=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  is_allowed "$path" || violations+=("$path")
done < <(git diff --name-only "$BASE_REF" -- 2>/dev/null || true)

# New untracked files under src/ must fall inside the allowed surface — a
# reward-hacking shim has to live under src/ to be compiled into the module.
# (Scratch dirs, build artefacts and the autoresearch data tree are irrelevant
# to what gets built and are deliberately not scanned; the harness itself is
# covered by the hash-pin below.)
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  is_allowed "$path" || violations+=("$path")
done < <(git ls-files --others --exclude-standard -- src 2>/dev/null || true)

if ((${#violations[@]})); then
  echo "PROTECTED: allowlist VIOLATION — files changed outside the optimisation surface" >&2
  printf '  %s\n' "${violations[@]}" >&2
  echo "The loop may only modify src/ale/vector/* and src/ale/python/*vector*." >&2
  rc=1
fi

# --- Layer 2: hash-pin on the harness -------------------------------------
MANIFEST="autoresearch/protected.sha256"
if [[ ! -f "$MANIFEST" ]]; then
  echo "PROTECTED: manifest $MANIFEST missing — cannot verify harness integrity" >&2
  exit 2
fi
if ! sha256sum --quiet --check "$MANIFEST" >/dev/null 2>&1; then
  echo "PROTECTED: harness INTEGRITY failure — a hash-pinned file was modified:" >&2
  sha256sum --check "$MANIFEST" 2>&1 | grep -v ': OK$' >&2 || true
  echo "verify.py / bench.py / compare.py / analyse.py / goldens must never be edited." >&2
  echo "Legitimate golden change? A human re-pins from the repo root:" >&2
  echo "  sha256sum autoresearch/{verify,compare,bench,analyse}.py autoresearch/goldens/goldens.npz > autoresearch/protected.sha256" >&2
  rc=1
fi

if ((rc == 0)); then
  echo "PROTECTED: surface allowlist + harness integrity OK (base=$BASE_REF)"
fi
exit $rc