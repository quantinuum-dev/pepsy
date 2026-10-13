"""Pepsy's existing JAX/flat-Z2 pipeline, following Symmray's GPU example.

This validates amplitudes and a compiled real-loss gradient, not a TDVP run.
Use JAX_PLATFORMS=cpu for CPU or the configured JAX CUDA backend for GPU.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from pepsy.vmc import (SpinOrbitalColumns, fermionic_peps_rand,
                       make_fermionic_peps_batched_amplitude_function,
                       pack_fermionic_peps_ansatz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lx', type=int, default=2)
    parser.add_argument('--ly', type=int, default=3)
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--chi', type=int, default=2)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--contraction', choices=('boundary', 'ctmrg', 'hotrg', 'exact'), default='boundary')
    parser.add_argument('--seed', type=int, default=22)
    args = parser.parse_args()
    if args.d % 2:
        parser.error('Use equal Z2 sectors with an even total D')
    jax.config.update('jax_enable_x64', True)
    n = args.lx * args.ly
    peps = fermionic_peps_rand('Z2', args.lx, args.ly, args.d, flat=True,
        n_fermions_per_spin=(n // 2, n - n // 2), dtype='complex128', seed=args.seed)
    ansatz = pack_fermionic_peps_ansatz(peps)
    params = jax.tree_util.tree_map(jnp.asarray, ansatz.params)
    # The upstream example interleaves up/down occupations. Keep this map
    # explicit; Pepsy also applies the corresponding native Fock phase.
    columns = SpinOrbitalColumns(tuple(range(0, 2 * n, 2)), tuple(range(1, 2 * n, 2)))
    rng = np.random.default_rng(args.seed)
    x = rng.permuted(np.tile([0, 1], (args.batch, n)), axis=1).astype(np.int32)
    options = dict(contraction=args.contraction,
        chi=None if args.contraction == 'exact' else args.chi, cutoff=0.)
    scaled = make_fermionic_peps_batched_amplitude_function(
        ansatz, columns, output='mantissa_exponent', jit=True, **options)
    amplitude = make_fermionic_peps_batched_amplitude_function(
        ansatz, columns, output='amplitude', jit=True, **options)
    eager = make_fermionic_peps_batched_amplitude_function(
        ansatz, columns, output='amplitude', jit=False, **options)
    started = time.perf_counter()
    mantissa, exponent = jax.block_until_ready(scaled(x, params=params))
    values = jax.block_until_ready(amplitude(x, params=params))
    expected = np.asarray(eager(x, params=params))
    np.testing.assert_allclose(np.asarray(values), expected, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(mantissa * 10**exponent), expected, atol=1e-9, rtol=1e-10)
    loss = jax.jit(lambda ps: jnp.square(jnp.abs(amplitude(x, params=ps))).sum())
    gradient = jax.block_until_ready(jax.jit(jax.grad(loss))(params))
    leaves, treedef = jax.tree_util.tree_flatten(params)
    deltas = [rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape) for z in leaves]
    norm = np.sqrt(sum(np.vdot(d, d).real for d in deltas))
    deltas = [d / norm for d in deltas]
    prediction = sum(np.real(np.sum(np.asarray(g) * d))
        for g, d in zip(jax.tree_util.tree_leaves(gradient), deltas, strict=True))
    shifted = lambda sign: jax.tree_util.tree_unflatten(treedef,
        [z + sign * 1e-5 * d for z, d in zip(leaves, deltas, strict=True)])
    finite_difference = float((loss(shifted(1)) - loss(shifted(-1))) / 2e-5)
    if not all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(gradient)):
        raise RuntimeError('Compiled gradient contains non-finite values')
    np.testing.assert_allclose(prediction, finite_difference, rtol=2e-6, atol=1e-7)
    print(json.dumps({'device': str(jax.devices()[0]), 'D': peps.max_bond(),
        'chi': options['chi'], 'cutoff': 0., 'dtype': str(values.dtype),
        'jit_amplitude_max_error': float(np.max(np.abs(np.asarray(values) - expected))),
        'gradient_relative_error': abs(prediction - finite_difference) / max(abs(finite_difference), 1e-12),
        'compile_and_validation_seconds': time.perf_counter() - started}), flush=True)


if __name__ == '__main__':
    main()
