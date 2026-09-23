# fullFold

Wrapper-only multi-GPU scheduler for AlphaFold 3. It does not modify any
AlphaFold 3 source. Inputs must already be featurisation-ready (MSA / template
fields set, including `""` for MSA-free). This package does not run the
AlphaFold 3 data pipeline.

See [docs/design.md](docs/design.md) for the cost model and [examples/demo.sh](examples/demo.sh)
for a template → scan → dry-run → kill → resume walkthrough.

## Install

Requires Python 3.12+ and `alphafold3>=3.0.2` in the same environment. The
AlphaFold 3 git checkout is not needed as the working directory.

```bash
pip install .
# or, once published:
pip install fullFold
```

Then run `fullFold` or `python -m fullFold` from any directory.

## Quickstart

With GPUs visible and model weights at `--model-dir`:

```bash
fullFold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```

Dry-run the same plan without launching workers:

```bash
fullFold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models --dry-run
```

Generate a ligand screen from a SMILES file, then schedule it:

```bash
fullFold template --template receptor.json --records ligands.smi --output-dir jobs/
fullFold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```

Generate an MSA-free peptide / de novo protein screen (empty MSAs and
templates are written for each added chain; existing receptor chains are
left untouched):

```bash
fullFold template --template receptor.json --records binders.fasta --type protein --output-dir jobs/
fullFold run --input-dir jobs/ --output-dir results/ --model-dir /path/to/models
```

Resume is the same command: completed jobs (matching `done.json` hash) are skipped.

## Subcommands

| Command | What it does |
|---|---|
| `scan` | Token-estimate every `*.json` in `--input-dir`, write `ledger.jsonl` |
| `benchmark` | 1024-token 4-seed probe per GPU; cached under `~/.cache/fullFold/bench/` |
| `plan` | Scan + benchmark (on cache miss) + write per-GPU manifests |
| `run` | Plan then execute. `--dry-run` stops after manifests |
| `template` | One output JSON per record in a FASTA / CSV / SMI file |

## GPU scheduling

`CUDA_VISIBLE_DEVICES` selects GPUs (indices or UUIDs). Unset or empty means every GPU reported by `nvidia-smi`, ordered by PCI bus ID. `--gpus` overrides.

`--policy contiguous` (default) assigns jobs with DP, then rebalances and steals tails. `--policy roundrobin` is the fallback (also used if a GPU probe is contaminated).

`--bucket-mode free` (default) compiles at the next `reference_timings.csv` bucket (ceil-8 only above 5216 tokens). `--bucket-mode ladder` intersects AlphaFold 3's default compile buckets with that CSV.

Prefetch and background-extract are on by default. Disable with `--no-prefetch` and `--no-background-extract`.

## Config fields

All tunables live in `config.py` (`Config`). Defaults:

| Field | Default | Meaning |
|---|---|---|
| `input_dir` | `.` | Folder of AlphaFold 3 JSON jobs |
| `output_dir` | `.` | Per-job output trees and `_af3sched/` |
| `model_dir` | `~/models` | AlphaFold 3 model weights |
| `gpus` | `()` | Override GPU indices/UUIDs; empty uses `CUDA_VISIBLE_DEVICES` or all |
| `dry_run` | `False` | Write manifests and stop |
| `prefetch` | `1` | Seed queue depth; also prepares the next job's first seed during the current job's inference. `--no-prefetch` sets `0` and disables all overlap |
| `background_extract` | `True` | Overlap one extract+write with the next same-bucket inference. Drains before compiling a new shape. `--no-background-extract` disables it |
| `policy` | `contiguous` | `contiguous` DP (then rebalance + steal tails), or `roundrobin` fallback |
| `bucket_mode` | `free` | `free` = next CSV bucket (ceil-8 above 5216); `ladder` = AF3 buckets ∩ CSV |
| `buckets` | AF3 128..5120 | Candidate compile shapes |
| `bucket_margin` | `0.05` | Near-boundary token counts escalate to exact count |
| `stale_lock_seconds` | `900` | Reclaim a lock whose mtime is older than this |
| `retry_failed` | `False` | Re-run jobs with `failed.json` |
| `exact_split_threshold` | `512` | Above this, coarsen multi-GPU split points |
| `force_benchmark` | `False` | Ignore the probe cache |
| `bench_seed` | `42` | RNG seed for the 1024-token probe protein |
| `cache_dir` | `~/.cache/fullFold/bench` | Per-host, per-GPU probe cache |
| `jax_compilation_cache_dir` | `None` | Root for the per-host, per-GPU-name JAX compile cache (default `<cache_dir>/jax/<host>__<name>/`; reuse follows JAX's GPU-name topology, not compute capability) |
| `xla_mem_fraction` | `0.97` | `XLA_CLIENT_MEM_FRACTION` in workers |
| `xla_preallocate` | `True` | `XLA_PYTHON_CLIENT_PREALLOCATE` |
| `exact_tokens` | `False` | Always use the AF3 tokenizer (slow, CCD load) |
| `template` | `None` | Base JSON for `template` |
| `records` | `None` | FASTA / CSV / SMI for `template` |
| `record_type` | `protein` | Default kind for FASTA / CSV `sequence` column |
| `save_embeddings` | `False` | Write per-seed embeddings |
| `save_distogram` | `False` | Write per-seed distograms |
| `compress_large_output_files` | `False` | Passed to `post_processing.write_output` |
| `save_terms_of_use` | `True` | Include AF3 output terms of use |
| `num_recycles` | `10` | Model recycle count |
| `num_diffusion_samples` | `5` | Diffusion samples per seed |
| `flash_attention` | `triton` | Flash-attention implementation |

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
