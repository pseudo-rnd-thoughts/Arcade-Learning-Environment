#!/usr/bin/env python3
"""Rigorous throughput benchmark for the ALE vector environment.

This is the ``bench.py`` half of the autoresearch harness trio described in
``autoresearch/ale_vector_optimization.md`` (§10.2). Its job is to produce a
*machine-readable* measurement that beats its own noise floor, so that a later
``compare.py`` can render an accept/reject verdict on a candidate change.

What it measures
----------------
For each (backend, config) cell it runs a pure random-action loop (no learner,
no GPU inference) so the env pipeline is isolated, and records **per repeat**:

* ``sps``          - environment steps per second (the headline throughput number)
* ``fps``          - emulator frames per second (``sps * frame_skip``)
* latency percentiles ``p50/p90/p99`` of a single ``step()`` call, in milliseconds
* ``minflt`` / ``majflt`` - soft/hard page faults during the measured window
  (the fingerprint of the per-batch allocation problem, §2.1/§5.1)
* ``rss_mb``       - peak resident set size delta

Repeats give the **noise floor**: run the identical binary N times and look at
the relative spread. Most items in §5/§9 are worth 2-10% and are invisible if
the machine is noisy, so every JSON record carries its own spread.

Backends (``--backends``)
-------------------------
* ``ale``        - ``ale_py.AtariVectorEnv`` (the target of optimisation)
* ``gym_sync``   - ``gymnasium.vector.SyncVectorEnv`` + ``AtariPreprocessing``
* ``gym_async``  - ``gymnasium.vector.AsyncVectorEnv`` (subprocess workers)
* ``envpool``    - ``envpool`` C++ vector env (the throughput reference point)

All backends are configured to emit identical ``(N, 4, 84, 84)`` uint8 grayscale
observations with frame-skip 4 and noop-max 30, so their SPS is comparable.

Usage
-----
    python autoresearch/bench.py --quick                 # fast smoke test
    python autoresearch/bench.py --matrix --out bench.json
    python autoresearch/bench.py --backends ale envpool --num-envs 128 256

The output JSON is the artefact; the human-readable table printed to stdout is
just a convenience.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Game name helpers. `ale` uses snake_case, envpool/gymnasium use CamelCase.
# ---------------------------------------------------------------------------

# Representative "cheap" (simple TIA, short frames) and "expensive" games.
DEFAULT_GAMES = ["breakout"]
CHEAP_GAME = "pong"
EXPENSIVE_GAME = "seaquest"


def snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


# ---------------------------------------------------------------------------
# Runner abstraction: normalise every backend to reset()/sample()/step()/close.
# ---------------------------------------------------------------------------


class Runner:
    """Uniform interface over the different vector-env backends.

    ``steps_per_call`` is the number of individual environment advances produced
    by one ``step()`` call. For synchronous backends this equals ``num_envs``;
    for ALE async mode it equals ``batch_size``. Throughput is
    ``steps_per_call * n_calls / wall_time``.
    """

    backend: str
    num_envs: int
    batch_size: int
    steps_per_call: int

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def sample_actions(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, actions: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ALERunner(Runner):
    backend = "ale"

    def __init__(
        self,
        game: str,
        num_envs: int,
        batch_size: int,
        num_threads: int,
        autoreset_mode: str,
        thread_affinity_offset: int,
    ):
        from ale_py import AtariVectorEnv

        self.num_envs = num_envs
        self.batch_size = batch_size if batch_size > 0 else num_envs
        self.steps_per_call = self.batch_size
        self._env = AtariVectorEnv(
            game,
            num_envs,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
            autoreset_mode=autoreset_mode,
        )
        self._action_space = self._env.action_space

    def reset(self) -> None:
        self._env.reset(seed=0)

    def sample_actions(self):
        return self._action_space.sample()

    def step(self, actions) -> None:
        self._env.step(actions)

    def close(self) -> None:
        self._env.close()


def _make_gym_thunk(game: str) -> Callable[[], Any]:
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    gym.register_envs(ale_py)
    env_id = f"ALE/{snake_to_camel(game)}-v5"

    def thunk():
        # frameskip=1 at the base env; AtariPreprocessing owns the skip/maxpool.
        env = gym.make(env_id, frameskip=1)
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            grayscale_obs=True,
            screen_size=84,
            noop_max=30,
            scale_obs=False,
        )
        env = FrameStackObservation(env, stack_size=4)
        return env

    return thunk


class GymRunner(Runner):
    def __init__(self, game: str, num_envs: int, mode: str):
        import gymnasium as gym

        self.backend = "gym_async" if mode == "async" else "gym_sync"
        self.num_envs = num_envs
        self.batch_size = num_envs
        self.steps_per_call = num_envs
        thunk = _make_gym_thunk(game)
        thunks = [thunk for _ in range(num_envs)]
        if mode == "async":
            # 'spawn' rather than the default 'fork': the benchmark process has
            # imported JAX/OpenCV (multithreaded), and forking a multithreaded
            # process risks deadlock in the child workers.
            self._env = gym.vector.AsyncVectorEnv(
                thunks, autoreset_mode="NextStep", context="spawn"
            )
        else:
            self._env = gym.vector.SyncVectorEnv(thunks, autoreset_mode="NextStep")
        self._action_space = self._env.action_space

    def reset(self) -> None:
        self._env.reset(seed=0)

    def sample_actions(self):
        return self._action_space.sample()

    def step(self, actions) -> None:
        self._env.step(actions)

    def close(self) -> None:
        self._env.close()


class EnvpoolRunner(Runner):
    backend = "envpool"

    def __init__(self, game: str, num_envs: int):
        import envpool

        self.num_envs = num_envs
        self.batch_size = num_envs
        self.steps_per_call = num_envs
        task_id = f"{snake_to_camel(game)}-v5"
        self._env = envpool.make(
            task_id,
            env_type="gymnasium",
            num_envs=num_envs,
            stack_num=4,
            frame_skip=4,
            gray_scale=True,
            img_height=84,
            img_width=84,
            noop_max=30,
        )
        self._num_actions = int(self._env.action_space.n)

    def reset(self) -> None:
        self._env.reset()

    def sample_actions(self):
        return np.random.randint(0, self._num_actions, size=self.num_envs, dtype=np.int32)

    def step(self, actions) -> None:
        self._env.step(actions)

    def close(self) -> None:
        self._env.close()


def build_runner(backend: str, cfg: "Config") -> Runner:
    if backend == "ale":
        return ALERunner(
            cfg.game,
            cfg.num_envs,
            cfg.batch_size,
            cfg.num_threads,
            cfg.autoreset_mode,
            cfg.thread_affinity_offset,
        )
    if backend == "gym_sync":
        return GymRunner(cfg.game, cfg.num_envs, "sync")
    if backend == "gym_async":
        return GymRunner(cfg.game, cfg.num_envs, "async")
    if backend == "envpool":
        return EnvpoolRunner(cfg.game, cfg.num_envs)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Config + measurement
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Config:
    backend: str
    game: str
    num_envs: int
    batch_size: int = 0  # 0 => sync (== num_envs); only meaningful for `ale`
    num_threads: int = 0
    autoreset_mode: str = "NextStep"
    thread_affinity_offset: int = -1
    frame_skip: int = 4

    def label(self) -> str:
        sync = "sync" if self.batch_size in (0, self.num_envs) else f"bs{self.batch_size}"
        return (
            f"{self.backend}/{self.game}/n{self.num_envs}/{sync}/{self.autoreset_mode}"
        )

    def key(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RepeatResult:
    sps: float
    fps: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    minflt: int
    majflt: int
    rss_mb: float
    n_calls: int
    wall_time: float


def _percentiles(latencies_ms: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
        float(np.percentile(arr, 99)),
    )


def measure_once(
    runner: Runner,
    warmup_steps: int,
    measure_steps: int,
    frame_skip: int,
) -> RepeatResult:
    """Run one measured window on an already-constructed runner."""
    import psutil

    proc = psutil.Process()

    runner.reset()

    # Warmup: JIT/first-touch faults, page-in, steady-state thread scheduling.
    for _ in range(warmup_steps):
        runner.step(runner.sample_actions())

    # Pre-sample actions so RNG cost is out of the timed path.
    action_scripts = [runner.sample_actions() for _ in range(measure_steps)]

    gc.collect()
    gc.disable()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    rss0 = proc.memory_info().rss

    latencies_ms: list[float] = []
    t_start = time.perf_counter()
    for i in range(measure_steps):
        s = time.perf_counter()
        runner.step(action_scripts[i])
        latencies_ms.append((time.perf_counter() - s) * 1e3)
    wall = time.perf_counter() - t_start

    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rss1 = proc.memory_info().rss
    gc.enable()

    env_steps = runner.steps_per_call * measure_steps
    sps = env_steps / wall
    p50, p90, p99 = _percentiles(latencies_ms)

    return RepeatResult(
        sps=sps,
        fps=sps * frame_skip,
        latency_ms_p50=p50,
        latency_ms_p90=p90,
        latency_ms_p99=p99,
        minflt=ru1.ru_minflt - ru0.ru_minflt,
        majflt=ru1.ru_majflt - ru0.ru_majflt,
        rss_mb=(rss1 - rss0) / 1e6,
        n_calls=measure_steps,
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Environment fingerprint (§10.2: git SHA, build flags, CPU model, governor)
# ---------------------------------------------------------------------------


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _governor() -> str | None:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _smt_enabled() -> bool | None:
    try:
        with open("/sys/devices/system/cpu/smt/active") as fh:
            return fh.read().strip() == "1"
    except OSError:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _jax_devices() -> list[str] | None:
    try:
        import jax

        return [str(d) for d in jax.devices()]
    except Exception:
        return None


def environment_fingerprint(repo_root: str) -> dict[str, Any]:
    git = lambda *a: _sh(["git", "-C", repo_root, *a])  # noqa: E731
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "governor": _governor(),
        "smt_active": _smt_enabled(),
        "python": platform.python_version(),
        "numpy": _pkg_version("numpy"),
        "ale_py": _pkg_version("ale_py"),
        "gymnasium": _pkg_version("gymnasium"),
        "envpool": _pkg_version("envpool"),
        "jax_devices": _jax_devices(),
        "env_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENCV_FOR_THREADS_NUM", "MKL_NUM_THREADS")
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(repeats: list[RepeatResult]) -> dict[str, Any]:
    sps = [r.sps for r in repeats]
    median = statistics.median(sps)
    # Relative spread = the noise floor for this cell. If it is large, no verdict
    # on a small optimisation is trustworthy (§10.2).
    spread = (max(sps) - min(sps)) / median if median else float("nan")
    stdev = statistics.pstdev(sps) if len(sps) > 1 else 0.0
    return {
        "sps_median": median,
        "sps_mean": statistics.mean(sps),
        "sps_min": min(sps),
        "sps_max": max(sps),
        "sps_stdev": stdev,
        "sps_rel_spread": spread,
        "sps_rel_stdev": (stdev / median) if median else float("nan"),
        "fps_median": statistics.median(r.fps for r in repeats),
        "latency_ms_p50_median": statistics.median(r.latency_ms_p50 for r in repeats),
        "latency_ms_p99_median": statistics.median(r.latency_ms_p99 for r in repeats),
        "minflt_median": statistics.median(r.minflt for r in repeats),
        "majflt_median": statistics.median(r.majflt for r in repeats),
        "rss_mb_median": statistics.median(r.rss_mb for r in repeats),
    }


# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------


def build_configs(args: argparse.Namespace) -> list[Config]:
    configs: list[Config] = []
    for backend in args.backends:
        for game in args.games:
            for n in args.num_envs:
                for mode in args.autoreset_modes:
                    # Only `ale` supports batch_size != num_envs (async barrier).
                    if backend == "ale":
                        for frac in args.batch_fracs:
                            bs = 0 if frac >= 1.0 else max(1, int(n * frac))
                            configs.append(
                                Config(
                                    backend=backend,
                                    game=game,
                                    num_envs=n,
                                    batch_size=bs,
                                    num_threads=args.num_threads,
                                    autoreset_mode=mode,
                                    thread_affinity_offset=args.thread_affinity_offset,
                                )
                            )
                    else:
                        # gymnasium/envpool: sync only, NextStep semantics only.
                        if mode != "NextStep":
                            continue
                        configs.append(
                            Config(
                                backend=backend,
                                game=game,
                                num_envs=n,
                                batch_size=0,
                                autoreset_mode=mode,
                            )
                        )
    # De-duplicate while preserving order.
    seen: set = set()
    unique: list[Config] = []
    for c in configs:
        k = tuple(sorted(c.key().items()))
        if k not in seen:
            seen.add(k)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = build_configs(args)

    print(f"Configurations: {len(configs)}  x  repeats: {args.repeats}")
    fingerprint = environment_fingerprint(repo_root)
    if fingerprint["git_dirty"]:
        print("WARNING: working tree is dirty; results are not reproducible from git SHA")
    if fingerprint["governor"] not in (None, "performance"):
        print(
            f"WARNING: cpufreq governor is {fingerprint['governor']!r}, not 'performance' "
            "- expect thermal/frequency noise (§10.2 machine hygiene)"
        )

    # Interleaved schedule (A/B/A/B ...) to cancel thermal drift across repeats.
    # Each element is (repeat_index, config). Within a repeat the config order is
    # shuffled so no cell systematically runs while the machine is cold/hot.
    schedule: list[tuple[int, Config]] = []
    for r in range(args.repeats):
        order = list(configs)
        if args.interleave:
            random.Random(1234 + r).shuffle(order)
        schedule.extend((r, c) for c in order)

    # Collect per-config repeat lists.
    results: dict[str, list[RepeatResult]] = {c.label(): [] for c in configs}
    labels_to_cfg = {c.label(): c for c in configs}
    failures: dict[str, str] = {}

    for idx, (rep, cfg) in enumerate(schedule):
        label = cfg.label()
        if label in failures:
            continue
        print(f"[{idx + 1}/{len(schedule)}] repeat {rep + 1}: {label} ...", end="", flush=True)
        try:
            runner = build_runner(cfg.backend, cfg)
        except Exception as exc:  # backend missing / config unsupported
            failures[label] = f"{type(exc).__name__}: {exc}"
            print(f" SKIP ({failures[label]})")
            continue
        try:
            res = measure_once(runner, args.warmup, args.steps, cfg.frame_skip)
            results[label].append(res)
            print(f" {res.sps:,.0f} sps  p99={res.latency_ms_p99:.2f}ms  majflt={res.majflt}")
        except Exception as exc:
            failures[label] = f"{type(exc).__name__}: {exc}"
            print(f" FAIL ({failures[label]})")
        finally:
            try:
                runner.close()
            except Exception:
                pass
            del runner
            gc.collect()
            time.sleep(args.cooldown)

    cells = []
    for label, cfg in labels_to_cfg.items():
        reps = results[label]
        cell: dict[str, Any] = {
            "label": label,
            "config": cfg.key(),
            "repeats": [dataclasses.asdict(r) for r in reps],
        }
        if reps:
            cell["summary"] = summarise(reps)
        if label in failures:
            cell["error"] = failures[label]
        cells.append(cell)

    return {
        "fingerprint": fingerprint,
        "params": {
            "repeats": args.repeats,
            "warmup_steps": args.warmup,
            "measure_steps": args.steps,
            "interleave": args.interleave,
            "cooldown_s": args.cooldown,
        },
        "cells": cells,
    }


def print_table(report: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    header = f"{'config':<52} {'sps(median)':>14} {'±spread':>9} {'p99 ms':>8} {'majflt':>8}"
    print(header)
    print("-" * 100)
    for cell in report["cells"]:
        if "summary" not in cell:
            print(f"{cell['label']:<52} {'--- ' + cell.get('error', 'no data'):>50}")
            continue
        s = cell["summary"]
        print(
            f"{cell['label']:<52} {s['sps_median']:>14,.0f} "
            f"{s['sps_rel_spread'] * 100:>8.1f}% {s['latency_ms_p99_median']:>8.2f} "
            f"{s['majflt_median']:>8.0f}"
        )
    print("=" * 100)

    # Speedup of every backend relative to ale, per (game, num_envs) sync cell.
    ale_ref: dict[tuple[str, int], float] = {}
    for cell in report["cells"]:
        c = cell["config"]
        if c["backend"] == "ale" and c["batch_size"] in (0, c["num_envs"]) and "summary" in cell:
            ale_ref[(c["game"], c["num_envs"])] = cell["summary"]["sps_median"]
    if ale_ref:
        print("\nThroughput relative to ale (sync, same game/num_envs):")
        for cell in report["cells"]:
            c = cell["config"]
            ref = ale_ref.get((c["game"], c["num_envs"]))
            if ref and "summary" in cell:
                rel = cell["summary"]["sps_median"] / ref
                print(f"  {cell['label']:<52} {rel:>6.2f}x")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--backends",
        nargs="+",
        default=["ale", "gym_sync", "gym_async", "envpool"],
        choices=["ale", "gym_sync", "gym_async", "envpool"],
    )
    p.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    p.add_argument("--num-envs", nargs="+", type=int, default=[8, 64, 256])
    p.add_argument(
        "--batch-fracs",
        nargs="+",
        type=float,
        default=[1.0],
        help="ale-only: batch_size as a fraction of num_envs (1.0=sync). e.g. 0.25 0.5",
    )
    p.add_argument("--autoreset-modes", nargs="+", default=["NextStep"], choices=["NextStep", "SameStep"])
    p.add_argument("--num-threads", type=int, default=0, help="ale worker threads (0=auto)")
    p.add_argument("--thread-affinity-offset", type=int, default=-1)
    p.add_argument("--repeats", type=int, default=5, help="repeats per cell (noise floor)")
    p.add_argument("--warmup", type=int, default=200, help="warmup step() calls before timing")
    p.add_argument("--steps", type=int, default=2000, help="measured step() calls")
    p.add_argument("--cooldown", type=float, default=0.5, help="seconds between runs")
    p.add_argument("--no-interleave", dest="interleave", action="store_false")
    p.add_argument("--out", type=str, default=None, help="write JSON report to this path")
    p.add_argument("--quick", action="store_true", help="fast smoke: tiny matrix + few steps")
    p.add_argument(
        "--matrix",
        action="store_true",
        help="full matrix: cheap+expensive game, 8/32/128/256 envs, sync+async, both autoreset modes",
    )

    args = p.parse_args(argv)

    if args.quick:
        args.num_envs = [8]
        args.repeats = 2
        args.warmup = 50
        args.steps = 300
    if args.matrix:
        args.games = [CHEAP_GAME, EXPENSIVE_GAME]
        args.num_envs = [8, 64, 256]
        args.autoreset_modes = ["NextStep", "SameStep"]
        args.batch_fracs = [1.0, 0.5]

    report = run(args)
    print_table(report)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
