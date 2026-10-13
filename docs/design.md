# Design

## Cost model

A GPU that compiles shape `L` and then runs `k` inferences at that shape pays

```
compilation_us(R, L) + k * inference_us(S, L)
```

`R` and `S` are the 1024-token probe (`compile_us` and `infer_us_1024`). Per-bucket cost uses modifiers (`Inference_modifier`, `Compilation_modifier`; both `1.0` at 1024) through bucket **5216**. `--reference standard` (default) reads `fullFold/data/reference_timings.csv`. `--reference fast` reads `fullFold/data/reference_timing_fast.csv`, the Anthropic fast-mode compile and inference curve, and the scheduler optimises against that curve (candidate buckets in `free` mode are that file's `Bucket_size` values):

```
inference_us(S, L)     = round(S * Inference_modifier[L])
compilation_us(R, L)   = round(R * Compilation_modifier[L])
```

A lookup at a size missing from the active table snaps **up** to the next CSV bucket. Above 5216, inference uses the tokamax cubic and compile uses the last compile modifier of the active table (standard 5216 → 1.40297111; fast 5216 → 1.242444415) so compile does not follow the inference curve:

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

Default planner (`--policy search`, `fullFold/planner.py`, stdlib-only like `scheduling.py`). Same cost model, same `Group` output, so manifests, `share_tails` and the worker are unchanged.

*One GPU.* The inner DP runs on **aggregated** counts: inference units per rounded shape, so the cost is `O(K^2)` in the number of occupied shapes and independent of the job count (500,000 jobs in about 0.6 s; `plan_gpu` needs about 20 s). Between two occupied shapes only unoccupied table buckets that could replace the occupied one below are candidates (cheaper compile or cheaper inference, Pareto-minimal): any other bucket is dominated, so the optimum equals the DP over every bucket. The timing tables are not monotone, so a few unoccupied buckets do win occasionally (compile of 76 is 9% cheaper than 72 in the standard table).

*Several GPUs.* The model of `multi-gpu-bucket-scheduling`: every GPU opens its own buckets, jobs are not constrained to contiguous blocks. The state is `x[gpu][type]` with a type = (rounded shape, n_seeds), and a GPU's cost is the exact single-GPU DP of its types (cached), so each move is scored with the true compile plus inference cost and the plan does not depend on the order the GPUs are listed in. Seeds are optimal contiguous partitions for up to four GPU orders (bisection on the makespan; the cost of a job range never decreases when the range grows, so each GPU takes the longest range that fits, with earlier cuts bracketing later searches: about 100 DP calls at any job count) plus a speed-proportional split. A descent on the sorted load vector then moves batches of one type, whole bucket groups (all types served by one compiled shape) and swaps from the critical GPU to the least loaded ones; candidates are ranked by a fixed-bucket estimate and the best few are scored exactly. Iterated local search with a seeded RNG follows. Budgets are counted in DP cells, never seconds, so `plan.json` and the manifests stay byte-identical for identical inputs. Up to 12 jobs the plan is the absolute optimum (subset DP, `O(G 3^n)`). The result is never worse than the best contiguous seed, and `plan.json` records a valid lower bound (`planner.lower_bound_us`: hardest single job over all GPUs and shapes, and a fluid bound that relaxes atomic jobs and duplicated compiles). The most expensive single job at its own shape is also recorded as `floor_us` / `floor_job`. A GPU whose probe is unusable gets no jobs.

`--plan-effort` sets what follows the seeds, in DP cells (`BUDGET_CELLS`): `0` (3M cells) runs only the descent from the two best seeds, each limited to a third of the budget (no exact stage, no iterated search); `1` (default, 20M cells) adds the exact stage for up to 12 jobs and the iterated local search, which stops after 60 rounds without improvement and is skipped for large instances (more than 150 types or 5,000 jobs), where it gains at most 0.03%; `2` (80M cells) keeps the search going for 300 idle rounds, and also runs it on large instances with 15. Higher effort continues the same seeded search, so it is usually better but not guaranteed to be.

On 324 instances of 6-14 jobs the search alone (exact stage off) reached the optimum in 99.4% of cases against 49% for the contiguous-block planner of release 0.1.1, since removed (mean gap 0.003% against 9%); at 20-2,000 jobs it is on average 0.02-0.1% from the best plan found by the earlier prototypes, in 0.8-2.3 s instead of 4-15 s. Gains over contiguous blocks concentrate at roughly 3-100 jobs per GPU and vanish above about 1,000.

The derivation (aggregated DP, dominated-shape argument, exact elimination of the bucket variables, lower bound, the moves of the search) and the numerical comparison are written up in `docs/note/Multi-GPU-Planner-Note.pdf`, in the style of the `multi-gpu-bucket-scheduling` note; `Multi-GPU-Planner-Note.tex` is the LaTeX source.

Each GPU's manifest then appends the other GPUs' jobs **last-job-first** (`share_tails`), but only if the thief would **finish that job sooner** than the original GPU. Thief finish is leftover primary cost plus already-accepted steal, plus `compilation_us` if the bucket is new on that GPU this run, plus the job's inference. Original finish is the prefix cost on the owner through that job. Atomic `O_CREAT|O_EXCL` claims skip work another GPU already took. Stolen groups are marked `shared` and are not part of the predicted makespan.

Integer microseconds throughout. Ties broken by lowest index.

`--bucket-mode=free` (default) compiles at the active reference CSV's `Bucket_size` values: each job rounds up to the next CSV bucket, and intervening CSV sizes stay available so the inner DP can merge. Above 5216 it rounds **up to a multiple of 64**. `--bucket-mode=ladder` intersects AF3's bucket list with that CSV set, then adds CSV-only extras the same way (next CSV, not a raw token count, unless the job is larger than 5216). Oversized jobs always get a formula extra (the next multiple of 64, in both modes). `--reference=fast` selects the Anthropic fast-mode table for both the modifiers and those buckets.

If a probe is contaminated (seeds 3–4 disagree, so S is unusable), the planner falls back to round-robin. That still groups by shape. Tokamax compile on seeds 1 and 2, or a warm JAX cache (R ≈ 0), is not contamination: the planner still uses each GPU's throughput.

## Benchmark arithmetic

One 1024-token randomised protein, 32-sequence synthetic MSA, **4 model seeds**, `buckets=[1024]`. Only `ModelRunner.run_inference` is timed, `t1..t4` in milliseconds.

```
S = mean(t3, t4)          # raw inference ms at 1024 tokens
R = t1 + t2 - 2 S         # tokamax compile on seeds 1 and 2; featurisation excluded
compile_us   = round(R_ms * 1000)
infer_us_1024 = round(m_gpu * T_REF * 1e6)
```

Job cost at bucket `b` is `n_seeds * inference_us(infer_us_1024, b)`, not `tokens * us_per_token`. `compile_overhead_ms(bench, shape)` and `compilation_us` scale probe `R` by the CSV compile modifier (last-row modifier above 5216).

Cache key: `<host_id>__<pci_bus_id>|<device_kind>|<jax_version>|<af3_version>|<mode>|<model>|h=<hoists>.json` under `~/.cache/fullFold/bench/`. `host_id` is `/etc/machine-id` or a hostname hash. The versions come from package metadata (`jax+jaxlib`, `alphafold3[+fasterX]`), so the parent still never imports JAX, and a new software stack never reuses an old measurement. `<hoists>` also lists explicitly disabled fast-mode levers (`no-sampler-bf16`, `no-hoist-logits`). A contaminated probe (S unusable) is used for that run and not written, so one noisy measurement cannot pin a GPU to round-robin; an unreadable cache file is a miss. `--mode fast`, `--mode default`, and `--mode off` are distinct files, as are different active hoist sets under `default`. A cache hit on one PCI is reused for every selected GPU of that `device_kind` so identical cards keep the same `m_gpu`. Live probes run when `fullfold benchmark` or `--force-benchmark` is used, or when two or more `device_kind` values are selected and some kind still has no sample. Otherwise the planner uses unit costs `(compile_us, infer_us_1024) = (1e6, 1e6)` and `plan.json` records `probed: false`; `report` then prints relative modifier cost (1.0 = one 1024-token inference) instead of estimated seconds. Unit benches are not written to the cache. Required probes of missing cards still run concurrently, one subprocess per target, so the parent process never initialises CUDA.

Workers and probes always enable the JAX persistent compilation cache. The directory is `<cache_dir>/jax/<host_id>__<sanitised device_kind>/` unless `--jax-compilation-cache-dir` overrides the root. `device_kind` is the nvidia-smi GPU name, matching JAX's persistent-cache topology (a GPU-name string, not compute capability). Two cards with the same name on one host share the directory; A100 40GB vs 80GB, or A100 vs A30, do not, because JAX will not reuse those executables. The probe JSON stays per PCI.

## Pipeline unit

`featurise_input` materialises one batch per seed before returning. A 400-seed job cannot call it with all seeds at once. The worker calls it with `replace(fold_input, rng_seeds=[seed])` and pipelines `(job, seed)` through a `queue.Queue(maxsize=prefetch)`. Resident batches stay at about `prefetch + 1` within a job, plus at most one first-seed batch of the next job.

Background extract (on by default; `--no-background-extract` disables it) runs `extract_*` plus seed/final writes on a one-slot thread so they overlap the next inference. The backlog drains before a shape that has not been compiled yet, because a pending raw result sits in GPU memory through that compile. A job's `done.json` / `failed.json` is queued on the same thread right behind its last write, without making the main thread wait, so it lags the job by at most the extract in flight (not by the bucket). Resident batches rise to about `prefetch + 2`. Stolen groups of an already-compiled shape do not drain.

Locks: a claim is `O_CREAT|O_EXCL` on `.af3sched/lock` (host, pid, start time). A daemon thread in the claiming process refreshes the mtime of every lock it holds every `min(60, stale_lock_seconds / 4)` s, so `--stale-lock-seconds` means "no heartbeat", not "job is old". A lock is reclaimable when its pid is dead on this host or its mtime is older than that window; an unreadable lock is judged by age only (it may be mid-write). Reclaiming renames the lock away (one contender wins), checks it is the file that was judged stale, and puts it back if another claimant got there first. A worker that is interrupted or hits an error outside a job releases the claims it still holds instead of writing `failed.json` for jobs that never ran.

Inputs are keyed by `sanitise(name)`. `scan` rejects an input whose name sanitises to nothing usable, to `.`/`..`/`_af3sched`, or to the same id as an earlier file in sorted order (compared case-insensitively, because AlphaFold 3 lower-cases names and common filesystems do not distinguish case).

The parsed input and its CCD are kept for the running job and the next one (`worker._FOLDS`), so a many-seed job is parsed once, not once per seed.

Logging is one function, `log_event(fp, **kw)`, appending a JSON line to `<out>/_af3sched/gpu<slot>.jsonl`. Event types: `group_start`, `job_prep` (CPU claim + first-seed featurise starts), `job_start` (inference starts), `seed_done`, `job_done`, `job_failed`, `job_skip` (already done, failed, or locked by another GPU). Overlap is visible as `job_prep` for job B while job A has `job_start` but not yet `job_done`.

Worker stdout/stderr, including JAX/XLA compile warnings, is redirected to `<out>/_af3sched/gpu<slot>.log` so a run's terminal stays on scheduler progress. Uncaught worker exceptions still surface after that redirect is released.

Outputs are streamed the same way: each seed's sample / embedding / distogram dirs are written as that seed finishes; only ranking tuples and the current best `InferenceResult` are kept. File writes go through `post_processing.write_output`.

`process_item` rebuilds its RNG from the seed alone, so one seed at a time is bit-identical to the corresponding element of a batched call.

Inputs must already be featurisation-ready. `fullFold` does not run the AlphaFold 3 data pipeline; MSA / template fields have to be present (use `""` for MSA-free). `processing` here means claim + `write_fold_input_json` + `featurise_input`, not MSA/template search.

## Inference modes

`--mode default` monkeypatches three exact-math hoists (noise sharing, atom-pair table, token-pair conditioning) when the installed DeepMind AF3 source hashes match `hoists/fingerprints.json` (generated from v3.0.4). `ast.dump` differs between Python versions, so each function is recorded under several digests: a canonical serialisation (`canonical_dump`, identical on every Python) and the `ast.dump` forms. A function matches if any current digest is listed. `tools/update_hoist_fingerprints.py` writes all of them; a file written before that holds only an `ast.dump` digest, which matches on the Python that wrote it (or 3.12-format on 3.13+). A mismatch disables that hoist and prints a notice; `atom_cond_hoist` also requires `diffusion_hoist`. ColabFold ≥ 3.1.7 already does the three in native code; `--mode default` then only adds af3-faster's pair-logit hoist when that package is present. `--mode off` leaves AF3 untouched. `--mode fast` is af3-faster's full kernel set.

The hoists are the same ops as stock, not bitwise identical. `src/tests/test_hoist_equivalence.py` checks them on CPU when AlphaFold 3 is installed. The GPU checks are a manual procedure, run on AlphaFold 3 v3.0.4 with real weights:

1. One 1024-token protein and one ligand job, same seeds, `--mode default` vs `--mode off`. Compare coordinate RMSD, ranking score, pLDDT and PAE.
2. Memory at bucket 5120 with a ligand-heavy input at the automatic `XLA_CLIENT_MEM_FRACTION` (total minus 640 MiB), on a large card and on a card of 24 GB or less. If it OOMs, skip `atom_cond_hoist` above an atom threshold.
3. A timing probe for default vs off, and the same on ColabFold.

Workers set `XLA_CLIENT_MEM_FRACTION` to `(nvidia-smi memory.total − 640 MiB) / total` unless `--xla-mem-fraction` is passed.

## Determinism

Scan order is `(bucket, tokens, sanitised_name, sha256)`, never filesystem mtime. Manifests are `json.dumps(..., sort_keys=True)`. Given the same inputs, config, and benchmark cache, `plan.json` and `manifest-gpu*.json` are byte-identical across reruns.

## Where to change things

| Want | Edit |
|---|---|
| Default buckets, prefetch, thresholds | `Config` in `config.py` |
| AF3 ModelRunner / write_fold_input_json | `runner.py` (installed `alphafold3` package) |
| af3-faster kernels, `--model`, `run_alphafold.py` flags | `af3args.py`, `runner.py`, `cli.py`. Omitted `--mode` is `fast` when af3-faster is installed, else `default`. `openfold3` / `openbind0` are ColabFold-only choices |
| Exact-math diffusion hoists | `hoists/` (target DeepMind AF3 v3.0.4). Fingerprints in `hoists/fingerprints.json`. Regenerated by `tools/update_hoist_fingerprints.py` |
| XLA memory fraction | `mem_fraction` in `config.py`: `(memory.total − 640 MiB) / memory.total` |
| Per-bucket compile / inference modifiers | `data/reference_timings.csv` (`--reference standard`) or `data/reference_timing_fast.csv` (`--reference fast`); `compilation_us` / `inference_us` in `scheduling.py` |
| T_REF (A100 @ 1024) | `T_REF_S` in `benchmark.py` |
| Cost model / assignment | `scheduling.py` (keep it pure) |
| Multi-GPU assignment, search budgets | `planner.py` (`BUDGET_CELLS`, `EXACT_MAX_JOBS`, `plan_effort` in `config.py`) |
| AF3 output layout | `write_seed_outputs` / `write_job_final` in `worker.py` |
| How SMILES vs sequences are classified | `read_records` in `templates.py` (extension / column / `--type` only) |
