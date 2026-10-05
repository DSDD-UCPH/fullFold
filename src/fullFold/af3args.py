"""run_alphafold.py flag surface for the fullFold CLI.

Stdlib only. Detects a ColabFold tree from the installed package files so
the parent process does not import AlphaFold or JAX. DeepMind orig never
sees openfold3 or openbind0 choices.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ORIG_MODELS = ('alphafold3',)
COLAB_MODELS = ('alphafold3', 'openfold3', 'openbind0', 'of3', 'openbind')

_TRUE = {'1', 'true', 't', 'yes', 'y', 'on'}
_FALSE = {'0', 'false', 'f', 'no', 'n', 'off'}

# Explicitly set data-pipeline options. Omitted flags do not warn.
_PIPELINE_VALUES = (
    'db_dir',
    'small_bfd_database_path', 'small_bfd_z_value',
    'mgnify_database_path', 'mgnify_z_value',
    'uniprot_cluster_annot_database_path', 'uniprot_cluster_annot_z_value',
    'uniref90_database_path', 'uniref90_z_value',
    'ntrna_database_path', 'ntrna_z_value',
    'rfam_database_path', 'rfam_z_value',
    'rna_central_database_path', 'rna_central_z_value',
    'pdb_database_path', 'seqres_database_path',
    'jackhmmer_binary_path', 'nhmmer_binary_path', 'hmmalign_binary_path',
    'hmmsearch_binary_path', 'hmmbuild_binary_path',
    'jackhmmer_n_cpu', 'jackhmmer_max_parallel_shards',
    'nhmmer_n_cpu', 'nhmmer_max_parallel_shards',
    'msa_server_url', 'msa_server_user_agent',
)

_PIPELINE_WARN = (
    'fullFold: not running genetic or template search; input JSON must already '
    'contain MSAs and templates. Ignoring data-pipeline flags.'
)


def _module_exists(name: str) -> bool:
    """True when ``name`` can be found. A missing parent is absence, not an error."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def colabfold_installed() -> bool:
    """True when this environment's alphafold3 is the ColabFold tree.

    ``decoded_ccd.py`` and ``model_registry.py`` are the ColabFold markers.
    The check uses the package location only, so the CLI does not import
    AlphaFold or JAX.
    """
    try:
        spec = importlib.util.find_spec('alphafold3')
    except (ImportError, ValueError, AttributeError):
        return False
    locations = getattr(spec, 'submodule_search_locations', None) if spec else None
    if not locations:
        return False
    root = Path(next(iter(locations)))
    return (
        (root / 'constants' / 'decoded_ccd.py').is_file()
        and (root / 'model' / 'model_registry.py').is_file()
    )


def af3_faster_installed() -> bool:
    return _module_exists('af3_faster')


def model_choices(colabfold: bool) -> tuple[str, ...]:
    return COLAB_MODELS if colabfold else ORIG_MODELS


def flag_set(argv: list[str], name: str) -> bool:
    prefix = f'--{name}'
    return any(tok == prefix or tok.startswith(prefix + '=') for tok in argv)


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {value!r}')


class AbslBool(argparse.Action):
    """``--flag``, ``--flag=true``, and ``--flag=false``."""

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        super().__init__(
            option_strings=option_strings, dest=dest, nargs='?', const=True,
            default=default, required=required, help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        else:
            setattr(namespace, self.dest, parse_bool(values))


def add_run_alphafold_flags(p: argparse.ArgumentParser, *, colabfold: bool) -> None:
    """Flags from run_alphafold.py that fullFold did not already define."""
    p.add_argument(
        '--mode', choices=('off', 'fast'), default='off',
        help='af3-faster kernels. off (default) leaves AlphaFold unchanged. '
             'fast requires the af3-faster package.',
    )
    p.add_argument(
        '--model', choices=model_choices(colabfold), default='alphafold3',
        help='Model family whose weights live in --model-dir.',
    )
    p.add_argument('--num-seeds', '--num_seeds', type=int, default=None)
    p.add_argument('--gpu_device', type=int, default=None)
    p.add_argument('--jax_backend', default='gpu')
    p.add_argument('--run_inference', action=AbslBool, default=True)
    p.add_argument('--save-terms-of-use', '--save_terms_of_use',
                   action=AbslBool, default=True)
    p.add_argument('--force-output-dir', '--force_output_dir',
                   action=AbslBool, default=False)
    p.add_argument('--conformer-max-iterations', '--conformer_max_iterations',
                   type=int, default=None)
    p.add_argument('--resolve-msa-overlaps', '--resolve_msa_overlaps',
                   action=AbslBool, default=True)
    p.add_argument('--max-template-date', '--max_template_date',
                   default='2021-09-30')
    p.add_argument('--fix-standalone-glycans', '--fix_standalone_glycans',
                   action=AbslBool, default=False)
    p.add_argument('--download-weights', '--download_weights',
                   action=AbslBool, default=True)
    p.add_argument('--weights-precision', '--weights_precision', default='int8',
                   choices=('fp32', 'fp16', 'int8'))
    p.add_argument('--sampler-bf16', '--sampler_bf16', action=AbslBool, default=None)
    p.add_argument('--no-sampler-bf16', dest='sampler_bf16',
                   action='store_const', const=False)
    p.add_argument('--hoist-logits', '--hoist_logits', action=AbslBool, default=None)
    p.add_argument('--no-hoist-logits', dest='hoist_logits',
                   action='store_const', const=False)
    # Underscore --cache_dir is ColabFold's tokamax cache, not --cache-dir.
    p.add_argument('--cache_dir', dest='colab_cache_dir', default=None)
    p.add_argument('--lowercache_dir', default=None)
    p.add_argument('--run_data_pipeline', action=AbslBool, default=None)
    for name in _PIPELINE_VALUES:
        if name == 'db_dir':
            p.add_argument('--db_dir', action='append', default=None)
        else:
            p.add_argument(f'--{name}', default=None)
    if colabfold:
        p.add_argument('--of3_weights', action=AbslBool, default=None)
        p.add_argument('--of3_checkpoint', default=None)
        p.add_argument('--nojit', action=AbslBool, default=False)
        p.add_argument('--precompile', default='')
        p.add_argument('--use_msa_server', action=AbslBool, default=False)
        p.add_argument('--use_esm_embeddings', action=AbslBool, default=False)
        p.add_argument('--cyclic', default='')
        p.add_argument('--dropout', action=AbslBool, default=False)
        p.add_argument('--featurise_off', default='')


def finalize_args(args: argparse.Namespace, *, colabfold: bool) -> str | None:
    """Map accepted flags onto Config fields. Return an error, or None."""
    if args.cmd in ('scan', 'plan', 'run'):
        if args.json_path and args.input_dir:
            return 'fullFold: pass only one of --json_path and --input-dir'
        if not args.json_path and not args.input_dir:
            return 'fullFold: pass --input-dir or --json_path'
    if not getattr(args, 'run_inference', True):
        return 'fullFold: only runs inference; --run_inference=false is not supported'
    backend = str(getattr(args, 'jax_backend', 'gpu') or 'gpu').lower()
    if backend != 'gpu':
        return f'fullFold: only runs on GPU; --jax_backend={backend} is not supported'
    gpu = getattr(args, 'gpu_device', None)
    if gpu is not None and not str(getattr(args, 'gpus', '') or '').strip():
        args.gpus = str(gpu)
    if colabfold and getattr(args, 'of3_checkpoint', None):
        return (
            'fullFold: --of3_checkpoint is not supported; convert weights '
            'offline and pass --model openfold3 --model_dir DIR'
        )
    if colabfold and getattr(args, 'of3_weights', None):
        args.model = 'openfold3'
    if colabfold and getattr(args, 'nojit', False):
        return 'fullFold: --nojit is not supported'
    if colabfold and str(getattr(args, 'precompile', '') or '').strip():
        return 'fullFold: --precompile is not supported'
    if colabfold and getattr(args, 'use_msa_server', False):
        return 'fullFold: --use_msa_server is not supported; JSON must already contain MSAs'
    if colabfold and getattr(args, 'use_esm_embeddings', False):
        return 'fullFold: --use_esm_embeddings is not supported'
    if colabfold and str(getattr(args, 'cyclic', '') or '').strip():
        return 'fullFold: --cyclic is not supported'
    if colabfold and getattr(args, 'dropout', False):
        return 'fullFold: --dropout is not supported'
    if colabfold and str(getattr(args, 'featurise_off', '') or '').strip():
        return 'fullFold: --featurise_off is not supported'
    return None


def argument_warnings(args: argparse.Namespace) -> list[str]:
    notes = []
    pipeline = getattr(args, 'run_data_pipeline', None) is True
    if not pipeline:
        for name in _PIPELINE_VALUES:
            if getattr(args, name, None) not in (None, '', []):
                pipeline = True
                break
    if pipeline:
        notes.append(_PIPELINE_WARN)
    if getattr(args, 'colab_cache_dir', None):
        notes.append(
            'fullFold: ignoring --cache_dir (ColabFold cache). '
            'Benchmark cache is --cache-dir; JAX cache is --jax-compilation-cache-dir.'
        )
    if getattr(args, 'lowercache_dir', None):
        notes.append('fullFold: ignoring --lowercache_dir.')
    return notes
