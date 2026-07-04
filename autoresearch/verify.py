#!/usr/bin/env python3
"""Correctness oracle for the ALE vector environment (Python surface).

This is the ``verify.py`` half of the autoresearch harness trio described in
``autoresearch/ale_vector_optimization.md`` (§10.3). It defines the contract
where the library is actually consumed - the Gymnasium ``AtariVectorEnv`` API
(and the XLA ``xla_step`` function) - **not** the C++ interface. That single
choice makes ``EnvVectorizer`` / ``BatchResult`` / ``ResultStaging`` / the queue
/ the nanobind plumbing free-refactor territory: the oracle constrains only
Python-visible behaviour, so any internal change that preserves observations,
rewards, terminations, truncations and info arrays passes.

Determinism is what makes a bit-exact oracle possible: ALE with
``repeat_action_probability=0`` is deterministic given a seed, and the
INTER_AREA resize + uint8 pipeline are deterministic, so per-step observation
hashes match exactly across builds.

Checks
------
* ``generate``      - produce goldens from the *current* (reference) build.
* ``sync``          - ordered/sync run must reproduce the goldens bit-exactly.
* ``async``         - metamorphic: with ``batch_size < num_envs`` and a policy
                      keyed by ``(env_id, per-env step index)``, each env's
                      observation *sequence* must equal its sync golden, even
                      though batch composition is scheduling-dependent (§10.3).
* ``differential``  - the numpy ``step()`` path and the XLA ``xla_step`` path
                      must agree bit-exactly for identical seeds and actions
                      (pins the FFI layer independently; needs ``jax``).

Every check reports the **first divergence** as ``(step/call, env_id, field)``
and, for observation mismatches, dumps the offending frames to ``.npy`` so the
failure says *where*, not just *broken* - which is what keeps agent iterations
cheap.

Usage
-----
    python autoresearch/verify.py generate --goldens goldens.npz
    python autoresearch/verify.py sync  --goldens goldens.npz
    python autoresearch/verify.py async --goldens goldens.npz --batch-size 4
    python autoresearch/verify.py differential
    python autoresearch/verify.py all   --goldens goldens.npz   # gate: all of the above

Exit code is non-zero on any divergence, so this doubles as a CI/agent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Default oracle configuration.
#
# max_frames is deliberately small so a truncation + autoreset happens *inside*
# the golden window - otherwise the oracle would never exercise terminal /
# reset handling (§3.2 lives here). num_calls must comfortably exceed
# max_frames / frame_skip so the reset is captured.
# ---------------------------------------------------------------------------

DEFAULT_GAME = "breakout"
DEFAULT_NUM_ENVS = 8
DEFAULT_NUM_CALLS = 60
DEFAULT_SEED = 12345
DEFAULT_MAX_FRAMES = 120  # ~30 calls at frame_skip=4 -> forces a mid-window reset


@dataclass(frozen=True)
class OracleConfig:
    game: str = DEFAULT_GAME
    num_envs: int = DEFAULT_NUM_ENVS
    num_calls: int = DEFAULT_NUM_CALLS
    seed: int = DEFAULT_SEED
    max_frames: int = DEFAULT_MAX_FRAMES
    frame_skip: int = 4
    grayscale: bool = True
    stack_num: int = 4
    repeat_action_probability: float = 0.0
    autoreset_mode: str = "NextStep"


# ---------------------------------------------------------------------------
# Hashing + policy
# ---------------------------------------------------------------------------


def hash_obs(frame: np.ndarray) -> np.uint64:
    """8-byte BLAKE2 digest of one env's stacked observation as a uint64."""
    h = hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8)
    return np.uint64(int.from_bytes(h.digest(), "little"))


def policy_table(cfg: OracleConfig, num_actions: int) -> np.ndarray:
    """Deterministic action for every (env_id, call index).

    Keying the policy on ``(env_id, call_index)`` rather than on the batch is
    what makes the async oracle work: an env receives the identical action
    sequence regardless of how batches are scheduled, so its trajectory is a
    pure function of its own history (§10.3).
    """
    rng = np.random.default_rng(cfg.seed)
    return rng.integers(
        0, num_actions, size=(cfg.num_envs, cfg.num_calls), dtype=np.int64
    )


def _make_env(cfg: OracleConfig, batch_size: int):
    from ale_py import AtariVectorEnv

    return AtariVectorEnv(
        cfg.game,
        cfg.num_envs,
        batch_size=batch_size,
        autoreset_mode=cfg.autoreset_mode,
        max_num_frames_per_episode=cfg.max_frames,
        repeat_action_probability=cfg.repeat_action_probability,
        grayscale=cfg.grayscale,
        stack_num=cfg.stack_num,
        frameskip=cfg.frame_skip,
    )


def _num_actions(cfg: OracleConfig) -> int:
    env = _make_env(cfg, cfg.num_envs)
    n = int(env.single_action_space.n)
    env.close()
    return n


# ---------------------------------------------------------------------------
# Golden data structure
#
# Per-env sequences indexed by "number of actions applied":
#   obs_hash[e, k]     hash after k actions (k=0 is the reset observation)
#   lives[e, k]        lives after k actions
#   reward[e, k]       reward from action k   (k in 0..num_calls-1)
#   terminated[e, k]   terminated after action k
#   truncated[e, k]    truncated after action k
# ---------------------------------------------------------------------------


@dataclass
class Golden:
    config: dict[str, Any]
    num_actions: int
    policy: np.ndarray
    obs_hash: np.ndarray
    lives: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            config=json.dumps(self.config),
            num_actions=self.num_actions,
            policy=self.policy,
            obs_hash=self.obs_hash,
            lives=self.lives,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @staticmethod
    def load(path: str) -> "Golden":
        d = np.load(path, allow_pickle=False)
        return Golden(
            config=json.loads(str(d["config"])),
            num_actions=int(d["num_actions"]),
            policy=d["policy"],
            obs_hash=d["obs_hash"],
            lives=d["lives"],
            reward=d["reward"],
            terminated=d["terminated"],
            truncated=d["truncated"],
        )


def generate_golden(cfg: OracleConfig) -> tuple[Golden, np.ndarray]:
    """Run a sync (ordered) trajectory and record per-env golden sequences.

    Also returns the raw stacked observations tensor so a checker can dump the
    diverging frame for debugging.
    """
    num_actions = _num_actions(cfg)
    policy = policy_table(cfg, num_actions)
    n, T = cfg.num_envs, cfg.num_calls

    obs_hash = np.zeros((n, T + 1), dtype=np.uint64)
    lives = np.zeros((n, T + 1), dtype=np.int64)
    reward = np.zeros((n, T), dtype=np.int64)
    terminated = np.zeros((n, T), dtype=bool)
    truncated = np.zeros((n, T), dtype=bool)
    raw_obs = np.zeros((n, T + 1) + _obs_shape(cfg), dtype=np.uint8)

    env = _make_env(cfg, cfg.num_envs)  # sync => ordered, obs[e] is env e
    obs, info = env.reset(seed=cfg.seed)
    assert np.array_equal(info["env_id"], np.arange(n)), "sync recv must be ordered"
    for e in range(n):
        obs_hash[e, 0] = hash_obs(obs[e])
        lives[e, 0] = info["lives"][e]
        raw_obs[e, 0] = obs[e]

    for t in range(T):
        actions = policy[:, t]
        obs, rew, term, trunc, info = env.step(actions)
        assert np.array_equal(info["env_id"], np.arange(n))
        for e in range(n):
            obs_hash[e, t + 1] = hash_obs(obs[e])
            lives[e, t + 1] = info["lives"][e]
            raw_obs[e, t + 1] = obs[e]
        reward[:, t] = rew
        terminated[:, t] = term
        truncated[:, t] = trunc
    env.close()

    golden = Golden(
        config=asdict(cfg),
        num_actions=num_actions,
        policy=policy,
        obs_hash=obs_hash,
        lives=lives,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
    )
    return golden, raw_obs


def _obs_shape(cfg: OracleConfig) -> tuple[int, ...]:
    if cfg.grayscale:
        return (cfg.stack_num, 84, 84)
    return (cfg.stack_num, 84, 84, 3)


# ---------------------------------------------------------------------------
# Divergence reporting
# ---------------------------------------------------------------------------


class Divergence(Exception):
    def __init__(self, call: int, env_id: int, field: str, expected, got, frames=None):
        self.call = call
        self.env_id = env_id
        self.field = field
        self.expected = expected
        self.got = got
        self.frames = frames  # (expected_frame, got_frame) for obs mismatches
        super().__init__(
            f"first divergence @ call={call} env_id={env_id} field={field}: "
            f"expected {expected!r}, got {got!r}"
        )

    def dump_frames(self, outdir: str) -> str | None:
        if self.frames is None:
            return None
        os.makedirs(outdir, exist_ok=True)
        exp, got = self.frames
        base = os.path.join(outdir, f"divergence_call{self.call}_env{self.env_id}")
        np.save(base + "_expected.npy", exp)
        np.save(base + "_got.npy", got)
        return base


# ---------------------------------------------------------------------------
# Sync (ordered) check
# ---------------------------------------------------------------------------


def check_sync(cfg: OracleConfig, golden: Golden) -> None:
    n, T = cfg.num_envs, cfg.num_calls
    env = _make_env(cfg, cfg.num_envs)
    obs, info = env.reset(seed=cfg.seed)
    try:
        for e in range(n):
            h = hash_obs(obs[e])
            if h != golden.obs_hash[e, 0]:
                raise Divergence(0, e, "obs(reset)", int(golden.obs_hash[e, 0]), int(h),
                                 frames=(None, obs[e].copy()))
            if info["lives"][e] != golden.lives[e, 0]:
                raise Divergence(0, e, "lives(reset)", int(golden.lives[e, 0]),
                                 int(info["lives"][e]))
        for t in range(T):
            obs, rew, term, trunc, info = env.step(golden.policy[:, t])
            for e in range(n):
                h = hash_obs(obs[e])
                if h != golden.obs_hash[e, t + 1]:
                    raise Divergence(t + 1, e, "obs", int(golden.obs_hash[e, t + 1]),
                                     int(h), frames=(None, obs[e].copy()))
                if rew[e] != golden.reward[e, t]:
                    raise Divergence(t + 1, e, "reward", int(golden.reward[e, t]), int(rew[e]))
                if bool(term[e]) != bool(golden.terminated[e, t]):
                    raise Divergence(t + 1, e, "terminated",
                                     bool(golden.terminated[e, t]), bool(term[e]))
                if bool(trunc[e]) != bool(golden.truncated[e, t]):
                    raise Divergence(t + 1, e, "truncated",
                                     bool(golden.truncated[e, t]), bool(trunc[e]))
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Async (unordered) metamorphic check
# ---------------------------------------------------------------------------


def check_async(cfg: OracleConfig, golden: Golden, batch_size: int) -> None:
    """Each env's observation/reward sequence in async mode must equal its sync
    golden. Bookkeeping keys results by ``info['env_id']`` and a per-env count
    of actions applied, so scheduling-dependent batch composition is irrelevant.
    """
    n = cfg.num_envs
    if not (0 < batch_size < n):
        raise ValueError(f"async batch_size must be in (0, {n}), got {batch_size}")

    env = _make_env(cfg, batch_size)
    # sent[e] = number of actions applied to env e so far.
    sent = np.zeros(n, dtype=np.int64)
    reset_seen = np.zeros(n, dtype=bool)
    verified = np.zeros(n, dtype=np.int64)

    def action_for(e: int) -> int:
        j = int(sent[e])
        # Beyond the golden horizon we cannot verify; repeat last known action.
        return int(golden.policy[e, min(j, cfg.num_calls - 1)])

    try:
        obs, info = env.reset(seed=cfg.seed)
        prev_ids = np.asarray(info["env_id"]).tolist()
        for pos, e in enumerate(prev_ids):
            _cmp_obs(obs[pos], golden.obs_hash[e, 0], 0, e, "obs(reset)")
            reset_seen[e] = True

        # Run enough step() calls for every env to be verified num_calls times.
        # Generous headroom: async scheduling is uneven, so the slowest env needs
        # well beyond the num_calls*n/batch_size average before it reaches the
        # horizon. The loop breaks as soon as verified.min() hits num_calls, so
        # the cap only bounds the pathological (starvation) case.
        max_steps = 4 * ((cfg.num_calls * n) // batch_size + n)
        for _ in range(max_steps):
            actions = np.array([action_for(e) for e in prev_ids], dtype=np.int64)
            obs, rew, term, trunc, info = env.step(actions)
            # Each action we just sent is now applied to that env.
            for e in prev_ids:
                sent[e] += 1
            new_ids = np.asarray(info["env_id"]).tolist()
            for pos, e in enumerate(new_ids):
                k = int(sent[e])
                if not reset_seen[e]:
                    # Leftover reset from the initial reset() enqueue.
                    _cmp_obs(obs[pos], golden.obs_hash[e, 0], 0, e, "obs(reset)")
                    reset_seen[e] = True
                    continue
                if k > cfg.num_calls:
                    continue  # beyond golden horizon
                _cmp_obs(obs[pos], golden.obs_hash[e, k], k, e, "obs")
                if rew[pos] != golden.reward[e, k - 1]:
                    raise Divergence(k, e, "reward", int(golden.reward[e, k - 1]), int(rew[pos]))
                if bool(term[pos]) != bool(golden.terminated[e, k - 1]):
                    raise Divergence(k, e, "terminated",
                                     bool(golden.terminated[e, k - 1]), bool(term[pos]))
                if bool(trunc[pos]) != bool(golden.truncated[e, k - 1]):
                    raise Divergence(k, e, "truncated",
                                     bool(golden.truncated[e, k - 1]), bool(trunc[pos]))
                verified[e] = k
            prev_ids = new_ids
            if verified.min() >= cfg.num_calls:
                break
    finally:
        env.close()

    # Every env must have been verified across the full golden horizon. If the
    # loop exhausted max_steps without reaching that, a schedule starved some
    # env (or the env-id bookkeeping is wrong) - mirror test_batch_size_async's
    # per-env progress guarantee rather than passing on partial coverage.
    if verified.min() < cfg.num_calls:
        raise RuntimeError(
            f"async check did not fully verify every env "
            f"(verified={verified.tolist()}, expected all >= {cfg.num_calls})"
        )


def _cmp_obs(frame, golden_hash, call, env_id, field):
    h = hash_obs(frame)
    if h != golden_hash:
        raise Divergence(call, env_id, field, int(golden_hash), int(h),
                         frames=(None, np.asarray(frame).copy()))


# ---------------------------------------------------------------------------
# Cross-interface differential: numpy step() vs XLA xla_step()
# ---------------------------------------------------------------------------


def check_differential(cfg: OracleConfig) -> None:
    try:
        import jax
        import jax.numpy as jnp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jax not installed; skip differential (install ale_py[xla])") from exc

    num_actions = _num_actions(cfg)
    policy = policy_table(cfg, num_actions)
    n, T = cfg.num_envs, cfg.num_calls

    # --- numpy reference path ---
    env = _make_env(cfg, cfg.num_envs)
    ref_obs, ref = [], []
    obs, info = env.reset(seed=cfg.seed)
    ref_reset = obs.copy()
    for t in range(T):
        obs, rew, term, trunc, info = env.step(policy[:, t])
        ref_obs.append(obs.copy())
        ref.append((rew.copy(), term.copy(), trunc.copy()))
    env.close()

    # --- XLA path ---
    xenv = _make_env(cfg, cfg.num_envs)
    handle, xla_reset, xla_step = xenv.xla()
    handle, (xobs, _info) = xla_reset(handle, seed=cfg.seed)
    xobs = np.asarray(xobs)
    if not np.array_equal(xobs, ref_reset):
        _first_obs_diff(0, ref_reset, xobs)
    for t in range(T):
        handle, (xobs, xrew, xterm, xtrunc, _info) = xla_step(handle, policy[:, t])
        xobs = np.asarray(xobs)
        r_obs = ref_obs[t]
        if not np.array_equal(xobs, r_obs):
            _first_obs_diff(t + 1, r_obs, xobs)
        rew, term, trunc = ref[t]
        if not np.array_equal(np.asarray(xrew), rew):
            e = int(np.argmax(np.asarray(xrew) != rew))
            raise Divergence(t + 1, e, "reward(xla)", int(rew[e]), int(np.asarray(xrew)[e]))
        if not np.array_equal(np.asarray(xterm), term):
            e = int(np.argmax(np.asarray(xterm) != term))
            raise Divergence(t + 1, e, "terminated(xla)", bool(term[e]), bool(np.asarray(xterm)[e]))
        if not np.array_equal(np.asarray(xtrunc), trunc):
            e = int(np.argmax(np.asarray(xtrunc) != trunc))
            raise Divergence(t + 1, e, "truncated(xla)", bool(trunc[e]), bool(np.asarray(xtrunc)[e]))
    xenv.close()


def _first_obs_diff(call: int, ref: np.ndarray, got: np.ndarray) -> None:
    diff = np.any(ref != got, axis=tuple(range(1, ref.ndim)))
    e = int(np.argmax(diff))
    raise Divergence(call, e, "obs(xla)", "<ref frame>", "<xla frame>",
                     frames=(ref[e].copy(), got[e].copy()))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cfg_from_args(args: argparse.Namespace) -> OracleConfig:
    return OracleConfig(
        game=args.game,
        num_envs=args.num_envs,
        num_calls=args.num_calls,
        seed=args.seed,
        max_frames=args.max_frames,
        autoreset_mode=args.autoreset_mode,
    )


def _run_check(name: str, fn, dump_dir: str) -> bool:
    print(f"[{name}] ...", end=" ", flush=True)
    try:
        fn()
    except Divergence as d:
        print("FAIL")
        print(f"    {d}")
        base = d.dump_frames(dump_dir)
        if base:
            print(f"    dumped frames: {base}_*.npy")
        return False
    except Exception as exc:
        print("ERROR")
        print(f"    {type(exc).__name__}: {exc}")
        return False
    print("PASS")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "sync", "async", "differential", "all"])
    p.add_argument("--goldens", default="autoresearch/goldens.npz")
    p.add_argument("--game", default=DEFAULT_GAME)
    p.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    p.add_argument("--num-calls", type=int, default=DEFAULT_NUM_CALLS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument("--autoreset-mode", default="NextStep", choices=["NextStep", "SameStep"])
    p.add_argument("--batch-size", type=int, default=0, help="async batch size (default num_envs//2)")
    p.add_argument("--dump-dir", default="autoresearch/divergences")
    args = p.parse_args(argv)

    cfg = _cfg_from_args(args)

    if args.command == "generate":
        golden, _raw = generate_golden(cfg)
        os.makedirs(os.path.dirname(os.path.abspath(args.goldens)), exist_ok=True)
        golden.save(args.goldens)
        print(f"Wrote goldens -> {args.goldens}")
        print(f"  config: {cfg}")
        print(f"  obs_hash shape {golden.obs_hash.shape}, "
              f"resets in window: {int(golden.truncated.sum() + golden.terminated.sum())}")
        return 0

    if args.command == "differential":
        ok = _run_check("differential (numpy vs xla)", lambda: check_differential(cfg), args.dump_dir)
        return 0 if ok else 1

    # sync/async/all need goldens; verify the golden's config matches.
    if not os.path.exists(args.goldens):
        print(f"ERROR: goldens not found: {args.goldens}. Run 'generate' first.")
        return 2
    golden = Golden.load(args.goldens)
    gcfg = OracleConfig(**golden.config)
    if gcfg != cfg:
        # The golden pins the reference config; use it so a stray CLI flag can't
        # silently compare against the wrong contract.
        print(f"NOTE: using golden's pinned config {gcfg}")
        cfg = gcfg

    bs = args.batch_size or max(1, cfg.num_envs // 2)
    results: dict[str, bool] = {}
    if args.command in ("sync", "all"):
        results["sync"] = _run_check("sync", lambda: check_sync(cfg, golden), args.dump_dir)
    if args.command in ("async", "all"):
        results["async"] = _run_check(
            f"async (batch_size={bs})", lambda: check_async(cfg, golden, bs), args.dump_dir
        )
    if args.command == "all":
        results["differential"] = _run_check(
            "differential (numpy vs xla)", lambda: check_differential(cfg), args.dump_dir
        )

    passed = all(results.values())
    print("\n" + ("ALL GATES PASSED" if passed else "GATE FAILURES: " +
                   ", ".join(k for k, v in results.items() if not v)))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
