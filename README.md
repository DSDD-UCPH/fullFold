# fullFold: unlocking the full speed of AF3!

**fullFold** is a lightweight inference and scheduling layer for running AlphaFold 3 efficiently, with a particular focus on high-throughput workloads on one or more GPUs. It can boost throughput by up to **3-fold** by (1) moving CPU-side work to background processes to keep GPU utilization near full, and (2) intelligently select and allocate bucket sizes while balancing compilation overhead against inference throughput, and as a bonus it will enable compilation caching by default. The more jobs you run, the greater the impact of fullFold. However, even for a single or a few jobs, fullFold is faster than native AlphaFold 3 in approximately 95% of cases, while in the remaining cases it performs on par with native AF3.

On multi-GPU systems, fullFold orchestrates prediction jobs across all available GPUs, including heterogeneous systems with GPUs of different performance. It dynamically distributes workloads while reducing unnecessary scheduling overhead and fully preserving the underlying AF3 inference implementation. In our benchmarks, fullFold has shown near-linear scaling with both the number and performance of available GPUs**, allowing throughput to scale efficiently across multi-GPU systems.

fullFold is deliberately non-invasive: it does not modify the AlphaFold 3 source code or model, making it straightforward to use alongside an existing AlphaFold 3 installation. It operates on fully prepared AlphaFold 3 inputs and focuses exclusively on the inference stage, not MSA generation or other parts of the data pipeline.

Inputs must therefore already contain the required MSA and template fields, including `""` where appropriate for MSA-free predictions. **fullFold does not run or replace the AlphaFold 3 data pipeline**; feature generation using the standard AlphaFold 3 pipeline must be completed before running fullFold.

See [docs/design.md](docs/design.md) for the cost model and [examples/demo.sh](examples/demo.sh)
for a template → scan → dry-run → kill → resume walkthrough (although just a run can be enough).

## Install

Requires Python 3.12+ and DeepMind’s `alphafold3>=3.0.2` already installed in the same environment. 

```bash
# when installing from this GitHub repo
pip install .
# or directly via pip (SOON, not yet available):
pip install fullfold
```

Then run `fullfold` / `fullFold` (same command) or `python -m fullFold` from any directory.

## Quickstart

The main command that does everything (it will run a few minute benchmark the first time):

```bash
fullfold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```



## Other commands

Dry-run the same plan without launching workers:

```bash
fullfold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models --dry-run
```

Generate a ligand screen from a SMILES file, then schedule it:

```bash
fullfold template --template receptor.json --records ligands.smi --output-dir jobs/
fullfold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```



## Helper commands - setting up batches

For setting up small-molecule screens or MSA-free peptide / de novo protein screens, you can use the template command below. For peptides and proteins it will write empty MSAs and
skip templates. 

It takes an existing json input template with your configuration of interest, but without the other peptide/protein/small molecule. In our cases, this was often a single protein entry with all MSAs prepped. The records in the SMI/CSV/FASTA file are added as a new chain; existing receptor chains are left untouched):

Example 1 (SMILES input):

```bash
fullfold template --template receptor.json --records ligands.smi --output-dir jobs/
fullfold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```

Example 2 (fasta input):

```bash
fullfold template --template receptor.json --records binders.fasta --type protein --output-dir jobs/
fullfold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```

Resume is the same command: completed jobs (matching `done.json` hash) are skipped.

## Subcommands


| Command     | What it does                                                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `run`       | All-in-one command: will run scan the AF3 json, plan them, and then execute. `--dry-run` stops after plan                                      |
| `scan`      | Verify and calculate the tokens for every `*.json` in `--input-dir` (writes `ledger.jsonl`)                                                    |
| `benchmark` | benchmark of the available GPUs for a 1024-token 4-seed probe per GPU; cached under `~/.cache/fullFold/bench/`                                 |
| `plan`      | Scan + benchmark (on cache miss) + write per-GPU manifests                                                                                     |
| `template`  | Quickly setting up batch screens, from a template JSON file combined with each record in a FASTA / CSV / SMI file (one output JSON per record) |




## GPU scheduling

`CUDA_VISIBLE_DEVICES` selects GPUs (indices or UUIDs). Unset or empty means every GPU reported by `nvidia-smi`, ordered by PCI bus ID. `--gpus` overrides.

`--policy contiguous` (default) assigns jobs with DP, then rebalances and steals tails. `--policy roundrobin` is the fallback (also used if a GPU probe is contaminated).

`--bucket-mode free` (default) compiles at the next `reference_timings.csv` bucket (ceil-8 only above 5216 tokens). `--bucket-mode ladder` intersects AlphaFold 3's default compile buckets with that CSV.

Prefetch and background-extract are on by default. Disable with `--no-prefetch` and `--no-background-extract`.

## Config fields

All tunables live in `config.py` (`Config`). Defaults:


| Field                         | Default                   | Meaning                                                                                                                                                          |
| ----------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_dir`                   | `.`                       | Folder of AlphaFold 3 JSON jobs                                                                                                                                  |
| `output_dir`                  | `.`                       | Per-job output trees and `_af3sched/`                                                                                                                            |
| `model_dir`                   | `~/models`                | AlphaFold 3 model weights                                                                                                                                        |
| `gpus`                        | `()`                      | Override GPU indices/UUIDs; empty uses `CUDA_VISIBLE_DEVICES` or all                                                                                             |
| `dry_run`                     | `False`                   | Write manifests and stop                                                                                                                                         |
| `prefetch`                    | `1`                       | Seed queue depth; also prepares the next job's first seed during the current job's inference. `--no-prefetch` sets `0` and disables all overlap                  |
| `background_extract`          | `True`                    | Overlap one extract+write with the next same-bucket inference. Drains before compiling a new shape. `--no-background-extract` disables it                        |
| `policy`                      | `contiguous`              | `contiguous` DP (then rebalance + steal tails), or `roundrobin` fallback                                                                                         |
| `bucket_mode`                 | `free`                    | `free` = next CSV bucket (ceil-8 above 5216); `ladder` = AF3 buckets ∩ CSV                                                                                       |
| `buckets`                     | AF3 128..5120             | Candidate compile shapes                                                                                                                                         |
| `bucket_margin`               | `0.05`                    | Near-boundary token counts escalate to exact count                                                                                                               |
| `stale_lock_seconds`          | `900`                     | Reclaim a lock whose mtime is older than this                                                                                                                    |
| `retry_failed`                | `False`                   | Re-run jobs with `failed.json`                                                                                                                                   |
| `exact_split_threshold`       | `512`                     | Above this, coarsen multi-GPU split points                                                                                                                       |
| `force_benchmark`             | `False`                   | Ignore the probe cache                                                                                                                                           |
| `bench_seed`                  | `42`                      | RNG seed for the 1024-token probe protein                                                                                                                        |
| `cache_dir`                   | `~/.cache/fullFold/bench` | Per-host, per-GPU probe cache                                                                                                                                    |
| `jax_compilation_cache_dir`   | `None`                    | Root for the per-host, per-GPU-name JAX compile cache (default `<cache_dir>/jax/<host>__<name>/`; reuse follows JAX's GPU-name topology, not compute capability) |
| `xla_mem_fraction`            | `0.97`                    | `XLA_CLIENT_MEM_FRACTION` in workers                                                                                                                             |
| `xla_preallocate`             | `True`                    | `XLA_PYTHON_CLIENT_PREALLOCATE`                                                                                                                                  |
| `exact_tokens`                | `False`                   | Always use the AF3 tokenizer (slow, CCD load)                                                                                                                    |
| `template`                    | `None`                    | Base JSON for `template`                                                                                                                                         |
| `records`                     | `None`                    | FASTA / CSV / SMI for `template`                                                                                                                                 |
| `record_type`                 | `protein`                 | Default kind for FASTA / CSV `sequence` column                                                                                                                   |
| `save_embeddings`             | `False`                   | Write per-seed embeddings                                                                                                                                        |
| `save_distogram`              | `False`                   | Write per-seed distograms                                                                                                                                        |
| `compress_large_output_files` | `False`                   | Passed to `post_processing.write_output`                                                                                                                         |
| `save_terms_of_use`           | `True`                    | Include AF3 output terms of use                                                                                                                                  |
| `num_recycles`                | `10`                      | Model recycle count                                                                                                                                              |
| `num_diffusion_samples`       | `5`                       | Diffusion samples per seed                                                                                                                                       |
| `flash_attention`             | `triton`                  | Flash-attention implementation                                                                                                                                   |




## Markers

Per job, under `<output-dir>/<sanitised-name>/.af3sched/`:

- `lock` — `O_CREAT|O_EXCL`, reclaimed if this host's pid is dead or mtime is stale
- `done.json` — written atomically; skip on resume if `sha256` matches
- `failed.json` — skip on resume unless `--retry-failed`
- `af3.log` — AlphaFold 3 stdout/stderr for that job

Under `<output-dir>/_af3sched/`:

- `gpu{N}.jsonl` — per-GPU scheduler events (`job_prep`, `job_skip`, …)
- `gpu{N}.log` — worker stdout/stderr, including JAX/XLA compile warnings (not printed to the terminal during `run`)



## Templates

SMILES are always ligands. Kind comes from the file extension (`.smi` / `.smiles`), a CSV `smiles` column, or `--type`. The same string `CCCC` is a ligand from `.smi` and a protein from FASTA. `--type ligand` on a FASTA is refused.

Added proteins get `"unpairedMsa": ""`, `"pairedMsa": ""`, `"templates": []` (MSA-free). That is how a de novo or peptide screen skips the AlphaFold 3 data pipeline. Existing template chains are not touched. `modelSeeds` is copied verbatim and never synthesised.

## Citing fullFold

A preprint or publication describing **fullFold** is not yet available. In the meantime, if you use fullFold in your work, please cite this GitHub repository and the specific version used:

> Verhellen, J. & Kooistra, A. J. **fullFold: Unlocking the Full Speed of AlphaFold 3.** Version `<version>`. GitHub: `https://github.com/DSDD-UCPH/fullFold`.

For reproducibility, please replace `<version>` with the fullFold release used in your analysis (for example, `v0.1.0`). If you used an unreleased version, please cite the corresponding Git commit hash in addition to the repository URL.

Once a preprint or publication becomes available, the recommended citation will be updated here.