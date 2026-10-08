# Copyright 2026 Anthropic, PBC
# Copyright 2024 DeepMind Technologies Limited
#
# Modified by fullFold. Ported from af3-faster (Anthropic kit copy of
# sokrypton's ColabFold noise-sharing hoist) against
# google-deepmind/alphafold3 v3.0.4.
#
# Licensed under the Apache License, Version 2.0.

"""Share one noise embedding across the samples of a denoising step.

Stock ``diffusion_head.sample`` (AlphaFold 3 v3.0.4) carries
``noise_level_prev`` in the per-sample scan carry, so the noise embedding
and AdaLN projections are traced once per sample. This version feeds the
schedule as an unbatched scan input so those ops run once per step.
"""

from __future__ import annotations

_STATE = {'installed': False, 'traced': 0, 'stock': None}


def _make(DH):
    import haiku as hk
    import jax
    import jax.numpy as jnp

    def sample(denoising_step, batch, key, config):
        _STATE['traced'] += 1
        mask = batch.predicted_structure_info.atom_mask

        def apply_denoising_step(carry, levels):
            key, positions = carry
            noise_level, noise_level_prev = levels
            key, key_noise, key_aug = jax.random.split(key, 3)
            positions = DH.random_augmentation(
                rng_key=key_aug, positions=positions, mask=mask)
            gamma = config.gamma_0 * (noise_level > config.gamma_min)
            t_hat = noise_level_prev * (1 + gamma)
            noise_scale = config.noise_scale * jnp.sqrt(
                jnp.maximum(t_hat**2 - noise_level_prev**2, 0.0)
            )
            noise = noise_scale * jax.random.normal(key_noise, positions.shape)
            positions_noisy = positions + noise
            positions_denoised = denoising_step(positions_noisy, t_hat)
            grad = (positions_noisy - positions_denoised) / t_hat
            d_t = noise_level - t_hat
            positions_out = positions_noisy + config.step_scale * d_t * grad
            return (key, positions_out), positions_out

        num_samples = config.num_samples
        noise_levels = DH.noise_schedule(jnp.linspace(0, 1, config.steps + 1))
        key, noise_key = jax.random.split(key)
        positions = jax.random.normal(
            noise_key, (num_samples,) + mask.shape + (3,))
        positions *= noise_levels[0]
        init = (jax.random.split(key, num_samples), positions)
        step = hk.vmap(
            apply_denoising_step, in_axes=(0, None),
            split_rng=(not hk.running_init()),
        )
        result, _ = hk.scan(
            step, init, (noise_levels[1:], noise_levels[:-1]), unroll=4)
        _, positions_out = result
        final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))
        return {'atom_positions': positions_out, 'mask': final_dense_atom_mask}

    sample._cond_share = True
    return sample


def install() -> bool:
    if _STATE['installed']:
        return True
    from alphafold3.model.network import diffusion_head as DH
    _STATE['stock'] = DH.sample
    DH.sample = _make(DH)
    _STATE['installed'] = True
    return True


def uninstall() -> None:
    if not _STATE['installed']:
        return
    from alphafold3.model.network import diffusion_head as DH
    DH.sample = _STATE['stock']
    _STATE['installed'] = False


def report() -> dict:
    return {'installed': _STATE['installed'], 'traced': _STATE['traced']}
