"""AlphaFold 3 ModelRunner helpers against the installed alphafold3 package.

Mirrors the CLI helpers in DeepMind's run_alphafold.py so fullFold does not
need the AlphaFold 3 git checkout on sys.path.
"""

from __future__ import annotations

import datetime
import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fullFold.af3args as af3args

_CANON = {
    'af3': 'alphafold3',
    'of3': 'openfold3',
    'openbind': 'openbind0',
}


def canonical_model(name: str) -> str:
    return _CANON.get(name, name)


def make_model_config(
    *,
    flash_attention_implementation: str = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
    model_name: str = 'alphafold3',
):
    from alphafold3.model import model
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = (
        flash_attention_implementation
    )
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    apply_model_name(config, model_name)
    return config


def apply_model_name(config, model_name: str) -> None:
    """Set ColabFold ``global_config.model`` when that tree is installed."""
    name = canonical_model(model_name)
    if not af3args.colabfold_installed():
        if name != 'alphafold3':
            raise SystemExit(
                f'fullFold: --model {model_name} requires a ColabFold alphafold3 install'
            )
        return
    from alphafold3.model import model_registry
    model_registry.get(name).configure(config)


def install_fast_mode(cfg) -> None:
    """Patch this process with af3-faster kernels. No-op when mode is off."""
    if getattr(cfg, 'mode', 'off') != 'fast':
        return
    try:
        from af3_faster.install import apply_fast_env, install_fast
    except ImportError as e:
        raise RuntimeError(
            'fullFold: --mode fast requires the af3-faster package'
        ) from e
    apply_fast_env()
    install_fast()


def _family_marks(model_name: str) -> tuple[str, ...]:
    name = canonical_model(model_name)
    if name == 'openfold3':
        return ('openfold', 'of3')
    if name == 'openbind0':
        return ('openbind',)
    return (name,)


def _weight_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith('.bin.zst') or path.suffix in {'.npz', '.pkl', '.bin'}:
            out.append(path)
    return out


def _family_weights(root: Path, model_name: str) -> bool:
    marks = _family_marks(model_name)
    for path in _weight_files(root):
        name = path.name.lower()
        if name.startswith('af3'):
            continue
        if any(mark in name for mark in marks):
            return True
    return False


def resolve_model_dir(cfg) -> Path:
    """Directory of weights for ``cfg.model``.

    AlphaFold 3 stays at ``--model-dir`` and is never downloaded. OpenFold3 and
    OpenBind0 need their own blob; an ``af3.bin.zst`` in the same folder does
    not count. Downloading uses the installed ColabFold loader.
    """
    name = canonical_model(cfg.model)
    root = Path(cfg.model_dir).expanduser()
    if name == 'alphafold3':
        return root
    if not af3args.colabfold_installed():
        raise SystemExit(
            f'fullFold: --model {cfg.model} requires a ColabFold alphafold3 install'
        )
    if _family_weights(root, name):
        return root
    if not cfg.download_weights:
        raise SystemExit(
            f'fullFold: no {name} weights in {root}; expected {name}.bin.zst'
        )
    dest = None if _weight_files(root) else str(root)
    from alphafold3.model import weights
    found = _call_accepting(
        weights.ensure_weights,
        name,
        dest,
        download=True,
        precision=cfg.weights_precision,
    )
    return Path(found)


def load_ccd(fold_input):
    """ColabFold decoded CCD when that module exists, otherwise the orig CCD."""
    try:
        from alphafold3.constants import decoded_ccd
    except ImportError:
        from alphafold3.constants import chemical_components
        return chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    return decoded_ccd.get_ccd(user_ccd=fold_input.user_ccd)


def _has_msa(fold_input) -> bool:
    for chain in getattr(fold_input, 'chains', ()) or ():
        if getattr(chain, 'unpaired_msa', None) or getattr(chain, 'paired_msa', None):
            return True
    return False


def apply_model_features(batch, fold_input, model_name: str, refeaturise, model_dir):
    """Apply ColabFold featurisation conventions when the spec declares any."""
    if not af3args.colabfold_installed():
        return batch
    from alphafold3.model import model_registry
    spec = model_registry.get(canonical_model(model_name))
    if not getattr(spec, 'featurise', None):
        return batch
    from alphafold3.model import model_features
    return _call_accepting(
        model_features.apply,
        batch,
        spec,
        refeaturise=refeaturise,
        model_dir=model_dir,
        esm=None,
        lm_pair=None,
        has_msa=_has_msa(fold_input),
        fold_input=fold_input,
        cyclic=False,
    )


def featurise_fold(fold_input, seed: int, buckets, *, ccd=None, cfg=None):
    """One seed, padded to ``buckets``, with ColabFold conventions when present."""
    import dataclasses
    from alphafold3.data import featurisation
    one = dataclasses.replace(fold_input, rng_seeds=[seed])
    ccd = ccd or load_ccd(fold_input)
    extra = featurise_kwargs(cfg)
    model = cfg.model if cfg is not None else 'alphafold3'
    model_dir = cfg.model_dir if cfg is not None else None

    def once():
        return _call_accepting(
            featurisation.featurise_input,
            fold_input=one, ccd=ccd, buckets=buckets, verbose=False, **extra,
        )[0]

    return apply_model_features(once(), fold_input, model, once, model_dir)


def featurise_kwargs(cfg) -> dict:
    """Inference-time featurisation flags that ``featurise_input`` may accept."""
    if cfg is None:
        return {}
    kw = {
        'resolve_msa_overlaps': cfg.resolve_msa_overlaps,
        'fix_standalone_glycans': cfg.fix_standalone_glycans,
    }
    if cfg.conformer_max_iterations is not None:
        kw['conformer_max_iterations'] = cfg.conformer_max_iterations
    if cfg.max_template_date:
        date = datetime.date.fromisoformat(cfg.max_template_date)
        # DeepMind names the argument ref_max_modified_date; ColabFold may
        # use max_template_date. Whichever the installed function accepts is kept.
        kw['ref_max_modified_date'] = date
        kw['max_template_date'] = date
    return kw


def _call_accepting(fn, *args, **kwargs):
    """Call ``fn``, dropping keywords its signature does not declare."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    names = set(sig.parameters)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in names})


def output_terms(model_name: str) -> str:
    """Terms file for this model. Non-AF3 models use the ColabFold registry text."""
    name = canonical_model(model_name)
    if name != 'alphafold3':
        from alphafold3.model import model_registry
        return model_registry.get(name).output_terms()
    from etils import epath
    import alphafold3.cpp
    path = epath.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    return path.read_text()


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(self, config, device, model_dir: Path):
        from etils import epath
        self._model_config = config
        self._device = device
        self._model_dir = epath.Path(model_dir)

    @functools.cached_property
    def model_params(self):
        from alphafold3.model import params
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(self) -> Callable[..., Any]:
        import haiku as hk
        import jax
        from alphafold3.model import model

        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(self, featurised_example, rng_key):
        import jax
        from jax import numpy as jnp
        import numpy as np
        from alphafold3.model.components import utils

        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )
        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        return result

    def extract_inference_results(self, batch, result, target_name: str):
        from alphafold3.model import model
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(self, result, num_tokens: int):
        embeddings = {}
        if 'single_embeddings' in result:
            embeddings['single_embeddings'] = result['single_embeddings'][
                :num_tokens
            ].astype('float16')
        if 'pair_embeddings' in result:
            embeddings['pair_embeddings'] = result['pair_embeddings'][
                :num_tokens, :num_tokens
            ].astype('float16')
        return embeddings or None

    def extract_distogram(self, result, num_tokens: int):
        if 'distogram' not in result['distogram']:
            return None
        return result['distogram']['distogram'][:num_tokens, :num_tokens, :]


def write_fold_input_json(fold_input, output_dir) -> None:
    from etils import epath
    output_dir = epath.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{fold_input.sanitised_name()}_data.json'
    print(f'Writing model input JSON to {path}')
    path.write_text(fold_input.to_json())
