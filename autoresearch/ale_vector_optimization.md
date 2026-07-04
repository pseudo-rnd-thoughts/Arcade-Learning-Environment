# ALE Vector Environment — Performance Analysis & Optimization Plan

Scope: the C++ vectorizer (`EnvVectorizer`, `PreprocessedEnv`, `ActionQueue`, `ResultStaging`, `BatchResult`), the nanobind numpy interface, the XLA FFI interface (CPU + GPU handlers), and `AtariVectorEnv` in `vector_env.py`. Focus is throughput in **sync mode** (`batch_size == num_envs`, ordered staging), with end-to-end PPO training throughput as the ultimate metric.

---

## 1. Mental model for sync-mode throughput

Each `send()`/`recv()` round is a fork–join barrier. Per-step wall time decomposes as:

```
t_step ≈ max_over_envs(t_env) + t_fixed
```

where `t_env` is one environment's emulator + preprocessing time and `t_fixed` is the per-step overhead that does not parallelize (batch allocation, queue operations, GIL/numpy work, the recv→send turnaround on the main thread). Throughput is `batch_size / t_step`.

Two consequences drive everything below. First, the **slowest env sets the pace**: the p99/median ratio of per-env step latency directly bounds achievable efficiency, and env resets (fire + up to `noop_max` no-op emulator frames, ~34 frames vs. ~4 for a normal step) are a recurring ~8× straggler at every episode boundary. With 128 envs and ~1000-step episodes, roughly 12% of batches contain at least one reset. Second, **fixed per-step costs are paid at batch frequency**, so a few hundred microseconds of allocation or Python overhead per step is a large fraction of a ~1 ms batch.

Profile these four quantities separately: (a) per-env work time distribution, (b) time lost at the barrier (worker idle waiting for stragglers), (c) fixed per-step costs, (d) worker blocked-vs-running ratio.

---

## 2. Profiling methodology

Benchmark harness: a pure random-action loop from Python (no learner, no GPU) so the env pipeline is isolated. Then repeat through the training loop to measure end-to-end effects.

### 2.1 On-CPU profiling

```bash
perf record -g --call-graph dwarf -- python bench_random_policy.py
perf report --no-children
```

This gives the split between `ALEInterface::act` (Stella emulation), `applyPaletteGrayscale`/`applyPaletteRGB`, `maxpool_frames`, `cv::resize`, and `memcpy` in `write_to`/staging. Follow with:

```bash
perf stat -d -- python bench_random_policy.py   # IPC, LLC misses, and page-faults
perf c2c record -- ... && perf c2c report        # confirm/deny false sharing (§5.6)
```

Watch the **page-faults** counter specifically — it is the fingerprint of the per-batch allocation problem (§5.1): fresh multi-MB buffers are mmap'd and every byte written by workers takes a soft fault on first touch.

### 2.2 Off-CPU profiling

This is where sync-mode problems live and it is the step people skip. Use BCC/bpftrace `offcputime` (or `perf sched timehist`) to see how long workers block on `items_available_` / `slots_available_` and how long the main thread blocks in `wait_for_batch`:

```bash
offcputime-bpfcc -p <pid> 10 > offcpu_stacks.txt
```

If workers are idle 40% of the time, the problem is stragglers/scheduling and optimizing `cv::resize` will not help. OpenCV's internal thread pool, if enabled, shows up here as futex churn (§8.1).

### 2.3 Timeline tracing

For a lockstep system, a per-thread timeline beats aggregate profiles. Instrument Tracy zones (or `rdtsc` timestamps into per-thread arrays) around `execute_env`, `stage_result`, `enqueue_bulk`, and `recv`. The barrier effect, load imbalance, and reset stragglers become visually obvious. Dump per-env step-latency histograms; track p50/p99 over time.

### 2.4 Layer-by-layer baselines

Measure in this order so the ceiling at each layer is known:

1. Single `ALEInterface::act` + full preprocessing in a bare C++ loop → theoretical per-core FPS.
2. `EnvVectorizer` driven from C++ (no Python) → threading/staging overhead.
3. Through nanobind from Python → binding + GIL overhead per step.
4. Scaling curves over `num_envs ∈ {1, 8, 32, 128, 256}` × `num_threads`, physical cores only vs. with SMT, and (dual-socket) with/without NUMA binding. Sanity-check against envpool's published Atari numbers on comparable hardware.

### 2.5 Allocator experiments (cheap hypothesis tests)

```bash
strace -c -f -e trace=mmap,munmap,madvise -p <pid>       # syscalls per N steps
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so python bench_random_policy.py
```

If the tcmalloc (or jemalloc) preload alone raises FPS materially, the per-batch allocation cost (§5.1) is confirmed before writing any code.

---

## 3. Correctness flaws — fix before optimizing

### 3.1 Deadlock on worker error in sync mode

If a worker throws, `worker_loop` calls `set_error` and continues — but that env's result is never staged, so `staged_count_` never reaches `batch_size_`, `batch_ready_` is never signalled, and `recv()` blocks forever in `wait_for_batch()` *before* it can reach `check_error()`. The error is recorded but unreachable. Fix: `set_error` must also signal `batch_ready_`, and `wait_for_batch` should re-check the error flag on wake (then `recv` rethrows).

### 3.2 Stale observation on early termination

In `PreprocessedEnv::step`, the frame-skip loop `break`s on `game_over_` / truncation / life-loss **before** the `skip_id <= 2` screen capture. If the game ends on frame 1 or 2 of a 4-skip step, `raw_frames_` still holds the *previous* step's screens, so the terminal observation (and `final_obs` in SameStep mode) predates the action that ended the episode. Fix: capture the screen after the break when terminating.

### 3.3 `ActionQueue` is only single-producer safe

`enqueue_bulk` does `alloc_idx_.fetch_add`, then writes slots, then signals. With two concurrent producers, producer B's signal can release consumers into producer A's not-yet-written slots. Currently only the main thread and the destructor enqueue, so it holds — but it is an undocumented invariant one refactor away from a heisenbug. Document it, assert it, and add a capacity check (currently bounded only implicitly by the send/recv protocol).

### 3.4 Unenforced API protocol

`reset(env_ids, …)` writes `pending_seed_` on env objects from the main thread; nothing stops a caller doing this while those envs are mid-step after a `send()` — a data race. Calling `send` twice without `recv` overflows the queue/staging accounting. Add cheap state-machine guards so misuse is loud instead of corrupting. (The existing `first_batch_` guard is the right pattern; extend it.)

### 3.5 GPU FFI handler: async-copy lifetime landmine

In `XLAStepGPUImpl`, the trailing `cudaMemcpyAsync` calls copy from `result` (a stack-scoped `BatchResult`) and from local `host_terminations`/`host_truncations` vectors, then the function returns without a final stream sync. This is *accidentally* safe today only because async copies from **pageable** host memory block until the data is staged. The moment the source becomes pinned memory (recommended in §6.3), these become use-after-scope bugs. Either add a `cudaStreamSynchronize` before the temporaries go out of scope, or (better) use persistent pinned staging buffers owned by the vectorizer.

---

## 4. Buffer ownership: pooling semantics and rollout-buffer safety

Background: `BatchResult` heap-allocates every buffer per batch; the nanobind layer transfers ownership to numpy via a capsule whose deleter is `delete[]`. `vector_env.py` adds no copies — the arrays go straight to user code.

### 4.1 Current semantics (baseline)

Every `recv()` returns arrays that exclusively own fresh memory. `rollout_obs[t] = obs` (numpy slice assignment, `torch_buf[t] = ...`, `jnp .at[t].set(...)`) **copies element-wise** into the rollout's own storage; the entry is decoupled forever. Only storing the array *object* (`obs_list.append(obs)`, or the CleanRL-style `next_obs` held across a loop iteration for bootstrapping) is a reference — and even that is safe today because nothing ever mutates a returned buffer.

### 4.2 Refcount-aware buffer pool (safe design)

Replace the capsule deleter's `delete[]` with a return-to-free-list; `release_batch()` pops from the pool instead of `new[]`. **Safety invariant: a buffer re-enters the pool only when its capsule deleter fires**, and the deleter fires only when the last Python reference dies — including numpy views (which hold `.base`), `torch.from_numpy` bridges, and DLPack consumers. The C++ side never observes a buffer between handout and deleter, so it can never write into memory any live array can see. Consequences:

- Semantics are **identical** to today. `rollout[t] = obs` still copies; a retained reference (e.g. `next_obs`) simply keeps that one buffer out of the pool for a step.
- Graceful degradation: if a user retains every observation (list-append rollouts), the pool never refills and behavior falls back to fresh allocation — worse throughput, never corruption.
- Steady state for the copy-into-rollout pattern is 2–3 warm, faulted-in buffers, which is the point: no mmap/munmap churn, no first-touch faults in the worker write path.

Implementation notes: mutex-protect the free-list (deleters run under the GIL in CPython, so contention is nil — but do not assume which thread); cap the pool (~4 buffers) and `delete[]` beyond the cap; over-align to 64 B; make the pool a leaked process-lifetime object so capsule deleters firing after vectorizer destruction or during interpreter teardown cannot touch a dead pool. To route the deleter (which receives only `void*`) back to the pool, use a header block:

```cpp
struct alignas(64) BufHeader { BufferPool* pool; };
// allocate 64 + nbytes; hand out (base + 64); payload stays 64B-aligned

// capsule deleter:
nb::capsule owner(payload, [](void* p) noexcept {
    auto* h = reinterpret_cast<BufHeader*>(static_cast<char*>(p) - 64);
    h->pool->release(h);   // returns whole block to free-list, or frees past cap
});
```

Optionally `cudaHostRegister` pooled buffers once at first use so torch `non_blocking=True` H2D transfers genuinely overlap (§7.3).

### 4.3 The anti-pattern to avoid (or gate behind a flag)

A **fixed ring reused on a schedule** — K buffers rotated every K steps regardless of refcounts, with a "valid until next recv" contract — is the design that produces the classic failure: `rollout[t] = obs` happens to be safe in a tight lockstep loop, but the `next_obs` bootstrap, async logging, render queues, or any deferred read silently observes torn data. If this mode is ever wanted for the last few percent, it must be an explicit opt-in with a documented contract — the precedent being `copy=False` on Gymnasium's `SyncVectorEnv`.

---

## 5. Performance issues in the C++ pipeline

### 5.1 Per-batch heap allocation — likely the #1 fixed cost

`release_batch()` news up a complete `BatchResult` every step. At 256 envs × 4×84×84 grayscale that is ~7.2 MB of observations, above glibc's mmap threshold, so each step likely pays `mmap`/`munmap` plus a soft page fault on **every byte written** — inside the workers' write path. This is forced by the capsule-ownership handoff. Fixes, in order of preference: external outputs (§5.2) where the caller provides destination memory, or the refcount pool (§4.2) for the plain `recv()` path. Validate first with the tcmalloc preload and the page-fault counter (§2.5, §2.1).

### 5.2 `step_into` — write directly into caller-provided buffers

The strongest single change, and it pays on both interfaces:

- **XLA CPU path**: `XLAStepImpl` currently allocates a `BatchResult`, workers write ~7 MB into it, then the handler `memcpy`s the entire batch *again* into XLA's output buffers. Every observation byte is written twice, plus the allocation. XLA hands over the destination pointers **before** `send()`, so nothing prevents workers writing straight into them.
- **numpy/PPO path**: expose `envs.step(actions, out=rollout_obs[t])`. Observations land directly in rollout storage — removing the C++ allocation *and* the Python-side copy. Requirements: C-contiguous uint8, pointer extracted under the GIL, binding stashes an `nb::object` reference until the call returns (numpy arrays do not relocate, so the data pointer is stable).

Design sketch — `ResultStaging` holds a per-round pointer set instead of an owned `BatchResult`; `stage_result` builds `OutputSlot`s from it exactly as today (ordered mode: slot = env_id):

```cpp
struct ExternalOutputs {
    uint8_t* obs;
    int* rewards; bool* terminated; bool* truncated;
    int* env_ids; int* lives;
    int* frame_numbers; int* episode_frame_numbers;
    uint8_t* final_obs;   // nullable; SameStep only
};

// EnvVectorizer:
void step_into(const std::vector<Action>& actions, const ExternalOutputs& out);
// = register targets in staging → enqueue → wait_for_batch → check_error
```

Caller guarantees pointer validity until the call returns (trivially true for the XLA handler scope; enforced by the held `nb::object` on the numpy path). Roughly a 40-line refactor. With this in place, pooling is only needed for the convenience `recv()` API.

### 5.3 Queue overhead

Every step performs `batch_size` semaphore signal/wait pairs and 2×`batch_size` atomic RMWs on two shared cache lines (`alloc_idx_`, `dequeue_idx_`), all serialized through the same lines. `LightweightSemaphore` spin-waits, so it is not catastrophic — but it is pure overhead in lockstep. Options, in increasing invasiveness:

1. **Chunked dequeue** — workers grab K actions at a time (K ≈ 4–8) via `waitMany(K)` + a single `fetch_add(n)`, cutting contention by K while preserving dynamic load balancing. You *want* dynamic balancing: Atari step time varies by game state and resets are 8× stragglers, and the sync barrier means the slowest worker sets batch time.
2. For pure sync mode: replace the queue with an epoch counter + static partitioning with work stealing.

Try (1) only if the queue actually shows in the profile; expect it to be modest next to §5.1.

### 5.4 Observation copy chain

Per env per step: palette conversion → maxpool → `cv::resize` into the circular stack → `write_to` linearizes with `stack_num` memcpys into the batch buffer. Cheap wins:

- Linearize in **at most 2 memcpys** instead of `stack_num` — the circular buffer has at most two contiguous runs.
- SameStep terminal handling currently performs **three** full linearizations (`write_obs_to` + `write_to` twice); restructure to two by writing metadata directly instead of round-tripping through the slot.
- With `frame_skip == 1`, `maxpool_frames` runs every step against a zero-filled `raw_frames_[1]` — dead work; skip maxpool when only one frame was captured.

A deeper option — resizing only the newest frame directly into the output slot and copying the older `stack_num−1` frames from the previous output — eliminates linearization entirely but couples env state to batch buffers; only pursue if the profile says the ~28 KB/env memcpy matters next to `act` + `resize`. (Frame-stack deduplication, §7.2, makes this moot anyway.)

### 5.5 Thread count and placement

The default `num_threads = min(batch_size, hardware_concurrency())` uses every logical CPU, so workers fight the Python main thread and each other via SMT. The main thread getting descheduled delays the recv→send turnaround, which is dead time for *all* workers in sync mode. Benchmark `physical_cores − 1` as the default; when affinity is enabled, keep the main thread off the worker cores. On dual-socket machines, bind env frame buffers and the batch/output buffer to the same NUMA node as the workers touching them (`numactl --cpunodebind` as a first test; first-touch placement thereafter).

### 5.6 Metadata false sharing

Workers write adjacent elements of `rewards_`, `terminations_`, etc. — 16 ints or 64 bools per 64 B cache line. Real but small next to the observation traffic; confirm with `perf c2c` before acting. If it shows, use a per-slot padded staging area copied once at `release_batch` (or accept it — external outputs in §5.2 do not change this either way).

### 5.7 Python-boundary overhead

Each `recv` builds 8 numpy arrays, 8 capsules, and a dict; each `send` converts two Python lists into vectors. Amortizes fine at 128+ envs; can dominate at small `num_envs`. Trims: a fused C++ `step()` (one GIL round-trip instead of send + recv); accept actions as `nb::ndarray` instead of `std::vector` conversions; make the info dict **opt-out** (lives, frame numbers, and env_ids-in-ordered-mode are ignored by a PPO inner loop — skipping them removes four array/capsule constructions and a dict per step); intern the dict keys.

### 5.8 Minor

The `std::function` in `stage_result` fits SBO, but a template parameter costs nothing. `send()` duplicates action state (`set_action` on the env *and* the same fields in the queued `Action`) — pick one channel to remove a class of divergence bugs.

---

## 6. XLA/JAX path specifics

### 6.1 CPU: eliminate the double write

Covered in §5.2 — the FFI step handler should pass XLA's output pointers into `step_into` rather than memcpying a `BatchResult`. The bool→PRED element-wise loop can also become a memcpy (both are 1 byte on mainstream ABIs), though it is trivial either way.

### 6.2 GPU: constant-handle round trip

`XLAStepGPUImpl` performs a device→host copy **plus a full `cudaStreamSynchronize` at the top of every step** just to read the 8-byte vectorizer pointer, whose value is constant for the vectorizer's lifetime. Pass the pointer as an FFI **attribute** (`.Attr<int64_t>("handle")`, baked at trace time from `self.ale.handle()`), and keep the threaded handle buffer purely as an ordering token so `scan` sequencing is preserved.

### 6.3 GPU: pinned staging

H2D copies from pageable memory run at roughly half bandwidth and serialize with the stream. Give the vectorizer a persistent pinned staging buffer (`cudaHostAlloc` once, or `cudaHostRegister` on pooled buffers) and copy observations through it. This is exactly where the lifetime bug in §3.5 must be fixed first: pinned `cudaMemcpyAsync` is truly asynchronous.

---

## 7. Training-loop-level throughput (often the biggest wins)

### 7.1 Double-buffered sampling — overlap env stepping with inference

In sync PPO the envs idle during the policy forward pass and the accelerator idles during stepping. Split the envs into two groups — two vectorizers, or the existing async machinery with `batch_size = num_envs/2` — and while group A's actions are computed, group B steps. This is the sample-factory / SEED-RL pattern and is usually the largest *end-to-end* win available, larger than anything inside the env pipeline. The `send`/`recv` split already makes the API async-capable; it is mostly a training-loop change. It is easiest via the numpy or explicit send/recv interface; under `jit` + `scan` on the XLA path it is awkward.

### 7.2 Frame-stack deduplication

With `stack_num = 4`, every frame is written, copied, and transferred four times across consecutive steps. Run the envs with `stack_num = 1`, store rollouts as `[T+3, B, 84, 84]`, and gather 4-frame windows by indexing at minibatch time (rlpyt's frame-buffer trick). 4× less observation bandwidth through *every* layer — worker writes, XLA copies, H2D, rollout memory — at the cost of episode-boundary masking in the gather, which is controllable in a bespoke JAX library.

### 7.3 uint8 end-to-end

Keep observations uint8 through the rollout buffer and the H2D transfer; normalize (`/255.`) on-device after transfer. A host-side `astype(float32)` quadruples transfer volume and adds a large CPU pass. For torch, combine with pinned/registered pooled buffers so `.to(device, non_blocking=True)` genuinely overlaps with the next env step.

---

## 8. Environment/build-level

### 8.1 Disable OpenCV internal threading

Call `cv::setNumThreads(0)` in the vectorizer constructor. OpenCV builds with TBB/OpenMP can spawn their own pool inside `cv::resize`, fighting the env workers — a classic silent throughput killer, visible in the off-CPU profile as futex churn.

### 8.2 Compiler flags

`-O3 -march=native` (or `x86-64-v3` for distributable wheels), LTO, and ideally PGO over a stepping workload. Stella's interpreter loop is branch-heavy; PGO typically buys 5–15% on exactly the code that dominates the on-CPU profile.

### 8.3 Reset stragglers (known, hard to remove)

A NextStep-mode reset costs fire + up to `noop_max` no-op emulator frames (~34 frames vs. ~4 for a step). Within sync semantics there is no clean fix — the post-reset observation must be returned that step — but it will be visible as the p99 spike in the latency histogram and it is part of the motivation for double-buffered sampling (§7.1), which hides individual stragglers behind the other group's compute.

---

## 9. Priority summary

| # | Change | Kind | Expected sync-mode impact | Effort |
|---|--------|------|---------------------------|--------|
| 1 | Worker-error deadlock fix (§3.1) | Correctness | — (mandatory) | S |
| 2 | Stale terminal observation fix (§3.2) | Correctness | — (mandatory) | S |
| 3 | `step_into` external outputs (§5.2) | Alloc + copy | High — removes alloc, faults, and the XLA double write | M |
| 4 | Refcount buffer pool for `recv()` (§4.2) | Allocation | High if page faults confirmed; superseded by #3 where `out=` is used | M |
| 5 | `cv::setNumThreads(0)` (§8.1) | Oversubscription | Potentially large if OpenCV pool active; else nil | XS |
| 6 | Thread defaults: physical−1, main-thread placement (§5.5) | Scheduling | Low–mid | XS |
| 7 | Double-buffered sampling (§7.1) | Training loop | Large end-to-end | M–L |
| 8 | Frame-stack dedup, `stack_num=1` + gather (§7.2) | Bandwidth | Mid–high end-to-end | M |
| 9 | GPU: handle-as-attribute + pinned staging (§6.2–6.3) | XLA GPU | Mid on GPU path | S–M |
| 10 | PGO / LTO / `-march` (§8.2) | Build | 5–15% | S |
| 11 | Chunked dequeue (§5.3) | Queue | Low–mid; profile first | S |
| 12 | 2-memcpy linearization + SameStep 2× (§5.4) | Copy | Low | S |
| 13 | Opt-out info dict, `nb::ndarray` args, fused step (§5.7) | Python | Low unless small `num_envs` | S |
| 14 | False-sharing padding (§5.6) | Cache | Low; `perf c2c` first | S |
| 15 | Harness trio + agent loop (§10) | Infra | Multiplier — enables validated iteration on everything above | M |

---

## 10. Agentic optimization (autoresearch loop)

Given a rigorous benchmark script and a correctness oracle, the optimization process itself can be partially automated: an agent proposes a change → build → correctness gates → benchmark → accept/revert → journal → repeat (the AlphaEvolve / FunSearch / AIDE pattern; practically, Claude Code in headless mode or the Agent SDK inside the existing GCP + WandB setup). The agent is the easy part — **the evaluator is ~90% of the engineering** — and this codebase is unusually well suited because ALE with `repeat_action_probability=0` is deterministic given a seed, which enables a bit-exact oracle.

### 10.1 Tier 1: parameter autotuning (no LLM)

A chunk of the optimization surface is pure configuration: `num_threads`, `thread_affinity_offset`, dequeue chunk size K, pool cap, `cv::setNumThreads`, NUMA policy, compiler flags and PGO on/off. That is a classical autotuning problem — Optuna or nevergrad over the knob space with benchmark FPS as the objective. Zero correctness risk, a few hours of compute, and it gives the agent tier a properly tuned baseline so it does not burn iterations rediscovering thread counts. Run this first.

### 10.2 Benchmark rigor (`bench.py`)

The benchmark must beat its own noise floor, or every verdict is a coin flip:

- **Measure the noise floor first**: run the identical binary ~20 times. An unhygienic machine shows ±5–15% variance; most items in §5/§9 are worth 2–10% and would be invisible.
- **Machine hygiene**: dedicated box or sole-tenant VM. Shared cloud vCPUs (e.g. Cloud Run runners) contribute ±15% from noisy neighbours alone; sole tenancy also allows `perf_event_paranoid` low enough for the agent to run profilers itself. Pin the frequency governor, make an explicit SMT decision, isolate cores, exclude warmup steps.
- **Interleave** baseline/candidate runs (A/B/A/B) to cancel thermal drift; use paired statistics (bootstrap on medians or Mann–Whitney) with an acceptance threshold tied to the measured noise.
- **Config matrix**: `num_envs ∈ {8, 32, 128, 256}` × cheap/expensive game × NextStep/SameStep × numpy/XLA (× sync/async once the async oracle below exists). Objective = geomean speedup; hard constraint = no cell regresses >2%; keep a **hold-out cell** evaluated only at acceptance time to catch overfitting to the tuned configs.
- **Machine-readable output**: JSON with per-repeat FPS, latency percentiles, page-faults, RSS, plus an environment fingerprint (git SHA, build flags, CPU model, governor state).

> **As operated.** The loop uses `num_envs ∈ {8, 64, 256}` — small (Python/binding-overhead dominated), mid, and large (emulation + bandwidth dominated) — as the standing set, with the full matrix reserved for `bench.py --matrix`. The **north-star metric is SPS** (env steps/sec): accept/reject is decided on SPS geomean across the three env counts plus the per-cell regression guard, and **nothing else**. FPS, p50/p99 step latency, page-faults (`minflt`/`majflt`) and RSS are recorded per repeat to *explain* an SPS move (e.g. a page-fault drop confirming an allocation win) but never enter the verdict. `compare.py` implements the objective as a bootstrap CI on the geomean of per-cell median-SPS ratios; ACCEPT requires the geomean CI lower bound to clear `1 + accept_gain` with no significant per-cell regression.

### 10.3 Correctness oracle at the Python surface (`verify.py`)

Define the contract where the library is actually consumed: the Gymnasium `AtariVectorEnv` API and the XLA `xla_reset`/`xla_step` functions — **not** the C++ interface, which has essentially no external consumers. This single decision reclassifies most "interface-changing" work as agent-safe: `EnvVectorizer`, `BatchResult`, `ResultStaging`, the queue, and the nanobind plumbing become free-refactor territory, because the oracle constrains only Python-visible behavior. A Python-surface oracle also exercises the binding layer itself — capsule ownership, lifetime, and GIL mistakes surface as corrupted hashes or ASAN hits — which a C++-level oracle would miss.

Recipes per mode:

- **Sync / ordered**: fixed seeds + fixed action script; per-step BLAKE2/xxHash over observations, rewards, terminations, truncations, and info arrays; bit-exact match against goldens generated once from a pinned reference commit. (INTER_AREA resize and the uint8 pipeline are deterministic, so this holds.)
- **Async / unordered**: batch composition is scheduling-dependent, so batch-level hashing cannot work. Key the action script by `(env_id, per-env step index)`, accumulate results by `info["env_id"]`, and require each env's *sequence* to match its deterministic single-env golden.
- **Cross-interface differential**: numpy path ≡ XLA path for identical seeds/actions — cheap, and it pins the FFI layer independently.

Gates beyond goldens — all strict pass/fail *before* any timing counts, all under hard timeouts (§3.1 is a live deadlock path): TSAN (races do not show up in FPS numbers), ASAN/UBSAN, SameStep `final_obs` and async env-id-mapping tests, and error injection (worker throws → `recv` must rethrow within the timeout). `verify.py` should report the **first divergence** (step, env_id, field) and dump the offending frames — a Python-level mismatch says "broken", not "where", so localization support is what keeps agent iterations cheap. Keep component-level C++ tests as *diagnostics* the agent may edit during internal refactors, never as acceptance gates.

Escalation property: legitimate semantic changes (e.g. altering RNG consumption order in reset) fail the goldens *by construction*. That is a feature — such diffs route to a human for a deliberate golden regeneration instead of being silently accepted.

### 10.4 API-changing work under a Python-surface oracle

- **`out=` / `step_into` (§5.2)** — additive at the Python level. The default `step(actions)` path is validated against the existing goldens unchanged; the new path gets a differential test: `step(a, out=buf)` must be byte-identical to `step(a)` under the same seed. With the signature spec written down, this is agent-implementable end to end.
- **Frame-stack dedup (§7.2)** — a metamorphic oracle: `env(stack_num=1)` composed with the reference gather (zero-padding at episode starts; SameStep final-stack assembly from frame history) must be **bit-exact** against the `stack_num=4` goldens. Per-frame preprocessing and RNG consumption are independent of `stack_num`, so the equivalence is exact and becomes the acceptance criterion the agent iterates against.
- **Double-buffered sampling (§7.1)** — a training-loop change; the env-level oracle is untouched (the async per-env recipe already covers `batch_size = num_envs/2`). Acceptance is end-to-end — PPO steps/sec plus a learning-curve sanity check — and stays human-owned.

Under this framing, the human-only set shrinks to the API design decisions themselves (signatures and contracts, written as specs the agent implements) and training-loop restructuring. The right mental model remains: an optimization intern with infinite patience and a strict test suite, not an architect.

### 10.5 Protecting the oracle

Reward hacking is the documented failure mode of "make it faster" agents: caching outputs, weakening semantics, or gaming the timer is often faster than emulating Atari. Defences: goldens plus `verify.py`/`compare.py` live **outside the agent-writable tree**, hash-pinned, generated from a pinned reference commit; gates are strict pass/fail before any timing is even measured. The Python-surface choice makes this easier — the protected surface is one small, stable script plus golden blobs, rather than in-tree C++ tests the agent legitimately needs to touch while refactoring internals.

> **As built — two-layer guard** (`.claude/skills/ale-vector-autoresearch/check-protected.sh`, run before *and* after every iteration by `run_loop.sh`):
>
> 1. **Editable-surface allowlist.** The loop may only modify `src/ale/vector/*` and `src/ale/python/*vector*` (the C++ vectorizer + CMake, and the nanobind vector + XLA interfaces + `vector_env.py`). Enforced by diffing tracked files against the iteration's base commit and rejecting any change outside those globs; new untracked files under `src/` outside the globs are also rejected (a reward-hacking shim must live under `src/` to be compiled in). This keeps the agent out of the emulator core, game logic, the single-env path, top-level build flags, and the tests — and, crucially, means a change that "wins" can only have won by making the *vector pipeline itself* faster.
> 2. **Harness hash-pin.** `verify.py`, `bench.py`, `compare.py`, `analyse.py` and `goldens/` are untracked data the git allowlist cannot see, so they are pinned in `autoresearch/protected.sha256`. Any drift aborts the iteration. Re-pinning is a deliberate human act (the message tells you the exact `sha256sum … > protected.sha256` command), which is the only sanctioned path for a golden regeneration.
>
> Note the build-flag work (§8.2 PGO/LTO/`-march`) touches the *top-level* CMake, which is outside the editable surface by design — it is human-owned, not agent territory. `src/ale/vector/CMakeLists.txt` is inside the surface, so vector-local build tweaks remain in scope.

### 10.6 Loop mechanics and budget

Per iteration: read journal + fresh profiler output (perf top-N, off-CPU summary, page-fault counts) → pick **one** hypothesis → small diff on a branch → ccache/ninja build → gates → bench matrix → `compare.py` verdict → WandB log → commit or revert; journal updated either way. Feeding the profile into context each iteration is what separates this from blind mutation — proposals target the measured bottleneck, which is the evaluator-feedback trick that makes AlphaEvolve-style systems work.

This document is the seed journal: run phase 1 not as open-ended search but as a **prioritized ablation of §5/§9** — implement item, measure, record — then switch to exploration once the list is exhausted. Far higher yield per token, and the ablation data has standalone value for a writeup. Cost per iteration ≈ 1–3 min build (ccache) + ~2 min gates + 5–10 min bench matrix → roughly 30–50 iterations/day on one machine; greedy hill-climb with revert is sufficient at this scale — no population needed.

Prerequisite regardless of whether an agent ever touches the loop: the harness trio — `bench.py` (config matrix + JSON + fingerprint), `verify.py` (golden generation + gates + first-divergence reporting), and `compare.py` (paired stats, accept/reject verdict) — plus `analyse.py` (§10.7) for the writeup.

### 10.7 Records and analysis (the writeup falls out of the loop)

The loop is instrumented so a blog post is a by-product, not a separate effort. Every iteration — **accepted or reverted** — leaves a durable, machine-readable trail; nothing interesting is lost to a reverted branch.

Per iteration the loop writes:

- **`records/iterNN.json`** — the narrative + verdict metadata: `title`, `backlog_item` (§ reference), `hypothesis`, `branch`, `commit` (null on revert), `outcome` (`accept`/`revert`/`gate-fail`), `verdict_reason`, `geomean_speedup` + CI, `gates` pass/fail, a `diff_stat`, a `profile_note` (the evidence the SPS move is real, e.g. "off-CPU futex 12%→1%"), and pointers to the bench/verdict JSONs. The reverted attempts are the ablation — they are what make the post credible ("we tried chunked dequeue, it was within noise, here is the number").
- **`results/iterNN_bench.json`** — the full `bench.py` report (per-repeat SPS, latency percentiles, page-faults, RSS, fingerprint).
- **`results/iterNN_verdict.json`** — the `compare.py` verdict.

Plus a one-off **`results/benchmarks.json`**: a `bench.py` run of the reference backends (`gym_sync`, `gym_async`, `envpool`) and the ALE baseline, so every plot can show where ALE started and how it compares to the standard vector envs.

`analyse.py` turns that trail into the two artefacts a writeup needs:

- **`report/progress.png`** — SPS over the sequence of changes, one panel per `num_envs ∈ {8, 64, 256}`. The ALE trajectory is a running-best *accepted frontier* (only accepted iterations advance it), every attempt is a marker coloured by outcome (accepted / reverted / gate-fail), and `gym_sync`/`gym_async`/`envpool` are horizontal reference lines. The reader sees the starting point, each step's contribution, the dead ends, and the standing of ALE against the field — in one figure.
- **`report/metrics.{md,csv}`** — one row per change, columns for verdict, geomean speedup, SPS at each env count, the p99-latency and major-fault numbers that *explain* the SPS move, and the cumulative speedup versus the pre-optimization baseline. The markdown table drops straight into the post.

This replaces the WandB coupling in §10.6 for the single-machine case: the records are the log, `analyse.py` is the dashboard, and both are plain files in the repo. (WandB remains the right choice if the loop runs across many machines.) The design keeps the *reason* for every accept/revert attached to its number, which is exactly the material a post-mortem or blog post is made of.

---

## 11. Order of attack

1. Fix §3.1 and §3.2 — cheap, and they will otherwise fire mid-training-run.
2. Run the tcmalloc preload + a Tracy/off-CPU capture to confirm the two hypotheses: allocation/fault cost per step, and worker idle time at the barrier tail.
3. Implement `step_into` (XLA first, then the numpy `out=` API); add the pool only for the plain `recv()` path if it still matters after that.
4. One-liners with measurement: `cv::setNumThreads(0)`, thread-count defaults, build flags.
5. Queue chunking, linearization, false-sharing padding — only where the profile points.
6. Training-loop layer: double-buffering, frame-stack dedup, uint8-to-device pipeline.
7. Optionally automate: stand up the harness trio and the agent loop (§10) — Tier-1 autotune the knobs, then run the agent as a prioritized ablation of §5/§9 against the Python-surface oracle, keeping API specs (§10.4) and training-loop changes human-owned.
