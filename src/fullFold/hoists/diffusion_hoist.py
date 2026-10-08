# Copyright 2026 Anthropic, PBC
# Copyright 2024 DeepMind Technologies Limited
#
# Modified by fullFold. Ported from af3-faster (Anthropic kit copy of
# sokrypton's ColabFold pair-conditioning hoist) against
# google-deepmind/alphafold3 v3.0.4. OpenFold3 branches and the AF3
# no-op token-transformer hoist are dropped; the step uses stock
# Transformer (v3.0.4 already shares one projection per super-block).
#
# Licensed under the Apache License, Version 2.0.

"""Compute token-pair diffusion conditioning once per sample.

Stock orig rebuilds the pair half of ``DiffusionHead._conditioning``
(LayerNorm, projection, two transitions on N x N x 128) at every
denoising step. Those ops do not depend on the noisy coordinates or the
noise level. This computes them once and reuses them.
"""

from __future__ import annotations

import functools

_STATE = {'installed': False, 'traced': 0}
_STOCK = {}


def _pair_conditioning(self, batch, embeddings, use_conditioning):
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import diffusion_transformer as DT
    from alphafold3.model.network import featurization
    import jax.numpy as jnp

    pair_embedding = use_conditioning * embeddings['pair']
    rel_features = featurization.create_relative_encoding(
        seq_features=batch.token_features,
        max_relative_idx=32,
        max_relative_chain=2,
    ).astype(pair_embedding.dtype)
    features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
    pair_cond = hm.Linear(
        self.config.conditioning.pair_channel,
        precision='highest',
        name='pair_cond_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='pair_cond_initial_norm',
        )(features_2d)
    )
    for idx in range(2):
        pair_cond += DT.transition_block(
            pair_cond, 2, self.global_config, name=f'pair_transition_{idx}',
        )
    return pair_cond


def _single_conditioning(self, batch, embeddings, noise_level, use_conditioning):
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.network import diffusion_head as DH
    from alphafold3.model.network import diffusion_transformer as DT
    from alphafold3.model.network import noise_level_embeddings
    import jax.numpy as jnp

    single_embedding = use_conditioning * embeddings['single']
    target_feat = embeddings['target_feat']
    features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
    single_cond = hm.LayerNorm(
        use_fast_variance=False,
        create_offset=False,
        name='single_cond_initial_norm',
    )(features_1d)
    single_cond = hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='single_cond_initial_projection',
    )(single_cond)
    noise_embedding = noise_level_embeddings.noise_embeddings(
        sigma_scaled_noise_level=noise_level / DH.SIGMA_DATA,
    )
    single_cond += hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='noise_embedding_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='noise_embedding_initial_norm',
        )(noise_embedding)
    )
    for idx in range(2):
        single_cond += DT.transition_block(
            single_cond, 2, self.global_config, name=f'single_transition_{idx}',
        )
    return single_cond


def _make_head(DH):
    from alphafold3.model.components import haiku_modules as hm
    from alphafold3.model.components import utils
    from alphafold3.model.network import atom_cross_attention
    from alphafold3.model.network import diffusion_transformer as DT
    import jax.numpy as jnp

    Stock = DH.DiffusionHead

    class HoistDiffusionHead(Stock):
        def __call__(
            self,
            positions_noisy,
            noise_level,
            batch,
            embeddings,
            use_conditioning,
            mode='stock',
            pre=None,
        ):
            if mode == 'stock':
                return super().__call__(
                    positions_noisy, noise_level, batch, embeddings,
                    use_conditioning,
                )
            with utils.bfloat16_context():
                if mode == 'precompute':
                    pair_cond = _pair_conditioning(
                        self, batch, embeddings, use_conditioning)
                    return dict(pair_cond=pair_cond)
                assert mode == 'step' and pre is not None
                trunk_single_cond = _single_conditioning(
                    self, batch, embeddings, noise_level, use_conditioning)
                trunk_pair_cond = pre['pair_cond']
                sequence_mask = batch.token_features.mask
                atom_mask = batch.predicted_structure_info.atom_mask
                act = positions_noisy * atom_mask[..., None]
                act = act / jnp.sqrt(noise_level**2 + DH.SIGMA_DATA**2)
                enc = atom_cross_attention.atom_cross_att_encoder(
                    token_atoms_act=act,
                    trunk_single_cond=embeddings['single'],
                    trunk_pair_cond=trunk_pair_cond,
                    config=self.config,
                    global_config=self.global_config,
                    batch=batch,
                    name='diffusion',
                )
                act = enc.token_act
                act = jnp.asarray(act, dtype=jnp.float32)
                act += hm.Linear(
                    act.shape[-1],
                    precision='highest',
                    initializer=self.global_config.final_init,
                    name='single_cond_embedding_projection',
                )(
                    hm.LayerNorm(
                        use_fast_variance=False,
                        create_offset=False,
                        name='single_cond_embedding_norm',
                    )(trunk_single_cond)
                )
                act = jnp.asarray(act, dtype=jnp.float32)
                trunk_single_cond = jnp.asarray(
                    trunk_single_cond, dtype=jnp.float32)
                trunk_pair_cond = jnp.asarray(
                    trunk_pair_cond, dtype=jnp.float32)
                sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)
                transformer = DT.Transformer(
                    self.config.transformer, self.global_config)
                act = transformer(
                    act=act,
                    single_cond=trunk_single_cond,
                    mask=sequence_mask,
                    pair_cond=trunk_pair_cond,
                )
                act = hm.LayerNorm(
                    use_fast_variance=False,
                    create_offset=False,
                    name='output_norm',
                )(act)
                position_update = atom_cross_attention.atom_cross_att_decoder(
                    token_act=act,
                    enc=enc,
                    config=self.config,
                    global_config=self.global_config,
                    batch=batch,
                    name='diffusion',
                )
                skip_scaling = DH.SIGMA_DATA**2 / (
                    noise_level**2 + DH.SIGMA_DATA**2)
                out_scaling = (
                    noise_level * DH.SIGMA_DATA
                    / jnp.sqrt(noise_level**2 + DH.SIGMA_DATA**2)
                )
                return (
                    skip_scaling * positions_noisy + out_scaling * position_update
                ) * atom_mask[..., None]

    return HoistDiffusionHead


def sample_diffusion_hoisted(self, batch, embeddings, *, sample_config):
    import haiku as hk
    from alphafold3.model.network import diffusion_head as DH

    _STATE['traced'] += 1
    key = hk.next_rng_key()
    pre = self.diffusion_module(
        None, None, batch=batch, embeddings=embeddings,
        use_conditioning=True, mode='precompute',
    )
    denoising_step = functools.partial(
        self.diffusion_module,
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
        mode='step',
        pre=pre,
    )
    return DH.sample(
        denoising_step=denoising_step, batch=batch, key=key, config=sample_config,
    )


def install() -> bool:
    if _STATE['installed']:
        return True
    import haiku as hk
    from alphafold3.model import model as af3_model
    from alphafold3.model.network import diffusion_head as DH

    if not _STOCK:
        _STOCK.update(sample=af3_model.Model._sample_diffusion, dh_cls=DH.DiffusionHead)
    DH.DiffusionHead = _make_head(DH)
    af3_model.Model._sample_diffusion = hk.transparent(sample_diffusion_hoisted)
    _STATE['installed'] = True
    return True


def uninstall() -> None:
    if not _STATE['installed'] or not _STOCK:
        return
    from alphafold3.model import model as af3_model
    from alphafold3.model.network import diffusion_head as DH
    af3_model.Model._sample_diffusion = _STOCK['sample']
    DH.DiffusionHead = _STOCK['dh_cls']
    _STATE['installed'] = False


def report() -> dict:
    return {'installed': _STATE['installed'], 'traced': _STATE['traced']}
