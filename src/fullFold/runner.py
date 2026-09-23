"""AlphaFold 3 ModelRunner helpers against the installed alphafold3 package.

Mirrors the CLI helpers in DeepMind's run_alphafold.py so fullFold does not
need the AlphaFold 3 git checkout on sys.path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_model_config(
    *,
    flash_attention_implementation: str = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
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
    return config


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
