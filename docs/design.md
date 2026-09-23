# Design

## Cost model

A GPU that compiles shape `L` and then runs `k` inferences at that shape pays

```
compilation_us(R, L) + k * inference_us(S, L)
```

`R` and `S` are the 1024-token probe (`compile_us` and `infer_us_1024`). Per-bucket cost uses modifiers in `fullFold/data/reference_timings.csv` (`Inference_modifier`, `Compilation_modifier`; both `1.0` at 1024) through bucket **5216**:

```
inference_us(S, L)     = round(S * Inference_modifier[L])
compilation_us(R, L)   = round(R * Compilation_modifier[L])
```

A lookup at a size missing from the table snaps **up** to the next CSV bucket. Above 5216, inference uses the tokamax cubic and compile uses the last CSV compile modifier (5216 → 1.40297111) so compile does not follow the inference curve:

```
T_predicted(gpu, b) = m_gpu * f(b/1024) * T_REF     # inference only, b > 5216
f(u) = 1 + 0.33(u - 1) + 0.42(u² - 1) + 0.27(u³ - 1)
T_REF = 59.423 s   # A100 @ bucket 1024
m_A100 = 1
```

`k` is the number of *inferences*, not jobs: a 5-seed job is five times the GPU work of a 1-seed job. Compilation is charged **once per distinct compiled shape per GPU** at that shape's `compilation_us`, not a flat 1024 `R`. A long-lived `ModelRunner` (`jax.jit` on the instance) pays nothing the second time it sees a shape this run. The on-disk JAX cache is not inspected.

Each GPU's multiplier comes from its 1024-token probe: `m_gpu = S_1024 / T_REF`. An A100 whose probe matches `T_REF` has `m_gpu = 1`.

`fullFold/scheduling.py` is stdlib-only (`csv` + `pathlib`). It does not import AlphaFold 3 or JAX.

Inner DP (`plan_gpu`): choose which candidate shapes to compile. Every job runs at the smallest compiled shape at or above its token count. `O(m^2)` in the number of candidate shapes.

Outer DP (`plan_multi_gpu`): assign **contiguous** blocks of the sorted job list to GPUs, minimising makespan (`max` over GPUs). Jobs stay atomic: seeds of one job never split across GPUs. The most expensive single job is a hard floor, recorded as `floor_us` / `floor_job` in `plan.json`.

Above `exact_split_threshold` (default 512 jobs) split points coarsen to bucket boundaries plus `G` even cuts per bucket. After that, `rebalance_contiguous` moves each GPU–GPU split (binary search + local descent) so predicted per-GPU times differ by about one job, even when a boundary bucket is huge.

Each GPU's manifest then appends the other GPUs' jobs **last-job-first** (`share_tails`), but only if the thief would **finish that job sooner** than the original GPU. Thief finish is leftover primary cost plus already-accepted steal, plus `compilation_us` if the bucket is new on that GPU this run, plus the job's inference. Original finish is the prefix cost on the owner through that job. Atomic `O_CREAT|O_EXCL` claims skip work another GPU already took. Stolen groups are marked `shared` and are not part of the predicted makespan.

Integer microseconds throughout. Ties broken by lowest index.

`--bucket-mode=free` (default) compiles at CSV `Bucket_size` values: each job rounds up to the next CSV bucket, and intervening CSV sizes stay available so the inner DP can merge. Above 5216 it still rounds **up to a multiple of 8**. `--bucket-mode=ladder` intersects AF3's bucket list with the CSV set, then adds CSV-only extras the same way (next CSV, not a raw token count, unless the job is larger than 5216). Oversized jobs always get a formula extra (`free` → ceil 8, `ladder` → exact count).

If a probe is contaminated (seeds 3–4 disagree, so S is unusable), the planner falls back to round-robin. That still groups by shape. Tokamax compile on seeds 1 and 2, or a warm JAX cache (R ≈ 0), is not contamination: the contiguous DP still uses each GPU's throughput.

## Benchmark arithmetic

One 1024-token randomised protein, 32-sequence synthetic MSA, **4 model seeds**, `buckets=[1024]`. Only `ModelRunner.run_inference` is timed, `t1..t4` in milliseconds.

```
S = mean(t3, t4)          # raw inference ms at 1024 tokens
R = t1 + t2 - 2 S         # tokamax compile on seeds 1 and 2; featurisation excluded
compile_us   = round(R_ms * 1000)
infer_us_1024 = round(m_gpu * T_REF * 1e6)
```

Job cost at bucket `b` is `n_seeds * inference_us(infer_us_1024, b)`, not `tokens * us_per_token`. `compile_overhead_ms(bench, shape)` and `compilation_us` scale probe `R` by the CSV compile modifier (last-row modifier above 5216).

Cache key: `<host_id>__<pci_bus_id>|<device_kind>|<jax_version>|<af3_version>.json` under `~/.cache/fullFold/bench/`. `host_id` is `/etc/machine-id` or a hostname hash. Cache misses are probed concurrently, one subprocess per GPU, so the parent process never initialises CUDA.

Workers and probes always enable the JAX persistent compilation cache. The directory is `<cache_dir>/jax/<host_id>__<sanitised device_kind>/` unless `--jax-compilation-cache-dir` overrides the root. `device_kind` is the nvidia-smi GPU name, matching JAX's persistent-cache topology (a GPU-name string, not compute capability). Two cards with the same name on one host share the directory; A100 40GB vs 80GB, or A100 vs A30, do not, because JAX will not reuse those executables. The probe JSON stays per PCI.

## Pipeline unit

`featurise_input` materialises one batch per seed before returning. A 400-seed job cannot call it with all seeds at once. The worker calls it with `replace(fold_input, rng_seeds=[seed])` and pipelines `(job, seed)` through a `queue.Queue(maxsize=prefetch)`. Resident batches stay at about `prefetch + 1` within a job, plus at most one first-seed batch of the next job.

Background extract (on by default; `--no-background-extract` disables it) runs `extract_*` plus seed/final writes on a one-slot thread so they overlap the next inference. The backlog drains before a shape that has not been compiled yet, because a pending raw result sits in GPU memory through that compile. `done.json` is written after the drain, so it can lag by a job within a bucket. Resident batches rise to about `prefetch + 2`. Stolen groups of an already-compiled shape do not drain.

Logging is one function, `log_event(fp, **kw)`, appending a JSON line to `<out>/_af3sched/gpu<slot>.jsonl`. Event types: `group_start`, `job_prep` (CPU claim + first-seed featurise starts), `job_start` (inference starts), `seed_done`, `job_done`, `job_failed`, `job_skip` (already done, failed, or locked by another GPU). Overlap is visible as `job_prep` for job B while job A has `job_start` but not yet `job_done`.

Worker stdout/stderr, including JAX/XLA compile warnings, is redirected to `<out>/_af3sched/gpu<slot>.log` so a run's terminal stays on scheduler progress. Uncaught worker exceptions still surface after that redirect is released.

Outputs are streamed the same way: each seed's sample / embedding / distogram dirs are written as that seed finishes; only ranking tuples and the current best `InferenceResult` are kept. File writes go through `post_processing.write_output`.

`process_item` rebuilds its RNG from the seed alone, so one seed at a time is bit-identical to the corresponding element of a batched call.

Inputs must already be featurisation-ready. `fullFold` does not run the AlphaFold 3 data pipeline; MSA / template fields have to be present (use `""` for MSA-free). `processing` here means claim + `write_fold_input_json` + `featurise_input`, not MSA/template search.

## Determinism

Scan order is `(bucket, tokens, sanitised_name, sha256)`, never filesystem mtime. Manifests are `json.dumps(..., sort_keys=True)`. Given the same inputs, config, and benchmark cache, `plan.json` and `manifest-gpu*.json` are byte-identical across reruns.

## Where to change things

| Want | Edit |
|---|---|
| Default buckets, prefetch, thresholds | `Config` in `config.py` |
| AF3 ModelRunner / write_fold_input_json | `runner.py` (installed `alphafold3` package) |
| Per-bucket compile / inference modifiers | `data/reference_timings.csv`; `compilation_us` / `inference_us` in `scheduling.py` |
| T_REF (A100 @ 1024) | `T_REF_S` in `benchmark.py` |
| Cost model / assignment | `scheduling.py` (keep it pure) |
| AF3 output layout | `write_seed_outputs` / `write_job_final` in `worker.py` |
| How SMILES vs sequences are classified | `read_records` in `templates.py` (extension / column / `--type` only) |
