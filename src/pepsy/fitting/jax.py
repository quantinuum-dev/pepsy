"""Pure, fixed-topology JAX wrappers around the existing FIT sweep kernel."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import quimb.tensor as qtn

from .local import FIT


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_convergence(residual, tolerance, label):
    # One check per solve, including under jit. Never allow a failed forward
    # or adjoint solve to masquerade as a converged implicit gradient.
    value = float(residual)
    if not np.isfinite(value) or value > tolerance:
        raise RuntimeError(f"{label} did not converge: relative residual {value:g} > {tolerance:g}")


def _parameter_layout(params):
    import jax

    leaves, structure = jax.tree.flatten(params)
    return structure, tuple((x.shape, np.dtype(x.dtype)) for x in leaves)


def _check_parameters(params, layout, name, *, check_dtype=True):
    import jax

    leaves, structure = jax.tree.flatten(params)
    if structure != layout[0]:
        raise ValueError(f"{name} must preserve the packed parameter tree")
    for leaf, (shape, dtype) in zip(leaves, layout[1]):
        if getattr(leaf, "shape", None) != shape:
            raise ValueError(f"{name} must preserve packed parameter shapes")
        if check_dtype and np.dtype(leaf.dtype) != dtype:
            raise ValueError(f"{name} must preserve packed parameter dtypes")


@dataclass(frozen=True, eq=False)
class FunctionalFIT:
    """Immutable callable operating on Quimb-packed numeric parameter trees.

    Construct with :func:`make_jax_fit`. Each evaluation owns a fresh FIT;
    no network, cached environment, or tracer survives a call. ``to_network``
    reconstructs the fitted MPS/MPO with its original native metadata.
    """

    target_params: object
    initial_params: object
    _target_skeleton: object
    _guess_skeleton: object
    _evaluate: object
    _residual: object = None
    _guess_layout: object = None

    def __call__(self, target_params, guess_params=None):
        return self._evaluate(target_params, self.initial_params if guess_params is None else guess_params)

    def to_network(self, fitted_params):
        if self._guess_layout is not None:
            _check_parameters(fitted_params, self._guess_layout, "fitted_params", check_dtype=False)
        return qtn.unpack(fitted_params, self._guess_skeleton)

    def residual(self, target_params, fitted_params):
        """Gauge-aligned relative fixed-point residual for an implicit fit."""
        if self._residual is None:
            raise ValueError("residual is available only for differentiation='implicit'")
        return self._residual(target_params, fitted_params)


def make_jax_fit(
    target,
    p,
    *,
    n_iter=2,
    block_size=1,
    max_bond=None,
    sweep_sequence="RL",
    checkpoint=True,
    differentiation="unrolled",
    max_iterations=100,
    tolerance=None,
    adjoint_tolerance=None,
    adjoint_maxiter=100,
    contraction_opt="greedy",
):
    """Build a pure JAX fit with finite-sweep or checked implicit derivatives.

    ``unrolled`` differentiates exactly ``n_iter`` directional sweeps, also
    through the supplied guess. Checkpointing uses ``jax.checkpoint``.
    ``implicit`` converges complete RL cycles and differentiates the accepted
    fixed point using one gauge-aligned cycle and JAX GMRES (whose public
    implementation uses ``lax.custom_linear_solve``). Its initial guess is a
    numerical warm start and has zero derivative. Only dense open MPS/MPO
    charts with positive, fixed ranks are supported for implicit derivatives;
    native Symmray storage remains supported by finite-sweep evaluation
    subject to the existing FIT kernel's capabilities.

    Both paths preserve the guess topology and require static ranks. There
    is no adaptive cutoff, rank growth, normalization, or mutable FIT cache.
    Failed forward/adjoint convergence raises even under ``jax.jit``.
    Default residual tolerances follow parameter precision (1e-9 for float64,
    64 machine epsilons for float32). Implicit derivatives are first order.
    Dense block updates use the composed projector split; native block
    updates retain their existing SVD rule and spectral-chart limitations.
    """
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from ..backends import register_jax_linalg

    n_iter = _positive_integer("n_iter", n_iter)
    block_size = _positive_integer("block_size", block_size)
    if block_size not in (1, 2):
        raise ValueError("block_size must be 1 or 2")
    if differentiation not in ("unrolled", "implicit"):
        raise ValueError("differentiation must be 'unrolled' or 'implicit'")
    if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)) or p.cyclic:
        raise ValueError("p must be an open-boundary MPS or MPO")
    sweep_sequence = FIT._validate_sweep_sequence(sweep_sequence)
    max_bond = p.max_bond() if max_bond is None else _positive_integer("max_bond", max_bond)
    if max_bond != p.max_bond():
        raise ValueError("Prepare p at the requested max_bond before constructing a fixed-rank wrapper")
    # Freeze caller-owned metadata; closures must not consult a live MPS/MPO
    # after it is retagged, resized, or otherwise reused outside this wrapper.
    length, site_tag_id = int(p.L), p.site_tag_id
    # Register before tracing; this makes the functional API usable without
    # relying on process-global settings from a previous notebook/test.
    register_jax_linalg(stabilized=True, qr_rank_policy="adaptive")
    target_copy, guess_copy = target.copy(), p.copy()
    # Packed replacement arrays need not satisfy the original tensors'
    # isometry annotations. Keep topology/charges but never reuse those
    # annotations for changed inputs or reconstructed fitted tensors.
    for network in (target_copy, guess_copy):
        for tensor in network:
            tensor.modify(left_inds=None)
    target_params, target_skeleton = qtn.pack(target_copy)
    guess_params, guess_skeleton = qtn.pack(guess_copy)
    target_params = jax.tree.map(lambda x: jnp.array(x, copy=True), target_params)
    guess_params = jax.tree.map(lambda x: jnp.array(x, copy=True), guess_params)
    structure = jax.tree.structure(guess_params)
    shapes = [x.shape for x in jax.tree.leaves(guess_params)]
    target_layout, guess_layout = _parameter_layout(target_params), _parameter_layout(guess_params)
    native = any(type(t.data).__module__.split('.', 1)[0] == "symmray" for t in p)

    def checked(evaluate):
        def call(target_arrays, guess_arrays):
            _check_parameters(target_arrays, target_layout, "target_params")
            _check_parameters(guess_arrays, guess_layout, "guess_params")
            return evaluate(target_arrays, guess_arrays)
        return call

    def sweep(target_arrays, guess_arrays, count=n_iter):
        target_network = qtn.unpack(target_arrays, target_skeleton)
        # FIT updates tensor arrays rather than a network's external log10
        # exponent. Put this full-chain target in the guess's scale frame so
        # reconstructing with the original guess metadata retains its value.
        target_network.distribute_exponent(new_exponent=guess_skeleton.exponent)
        fit = FIT(target_network,
                  p=qtn.unpack(guess_arrays, guess_skeleton), cutoffs=0.,
                  site_tag_id=site_tag_id, contraction_opt=contraction_opt,
                  split_method="svd" if native else "projector")
        fit.run_eff(n_iter=count, block_size=block_size, max_bond=max_bond,
                    cutoff=0., sweep_sequence=sweep_sequence, rtol=None,
                    collect_split_diagnostics=False)
        arrays, _ = qtn.pack(fit.p)
        if jax.tree.structure(arrays) != structure or [x.shape for x in jax.tree.leaves(arrays)] != shapes:
            raise ValueError("FIT changed the fixed parameter topology or ranks; prepare a compatible p")
        return arrays

    if differentiation == "unrolled":
        evaluate = jax.checkpoint(sweep) if checkpoint else sweep
        return FunctionalFIT(target_params, guess_params, target_skeleton, guess_skeleton,
                             checked(evaluate), _guess_layout=guess_layout)

    if sweep_sequence != "RL":
        raise ValueError("Implicit fitting requires complete sweep_sequence='RL' cycles")
    if native:
        raise NotImplementedError("Implicit native Symmray gauge alignment is not implemented; use unrolled")
    max_iterations = _positive_integer("max_iterations", max_iterations)
    adjoint_maxiter = _positive_integer("adjoint_maxiter", adjoint_maxiter)
    precision = np.finfo(np.result_type(*[x.real.dtype for x in jax.tree.leaves(guess_params)])).eps
    tolerance = max(1e-9, 64*precision) if tolerance is None else tolerance
    adjoint_tolerance = max(1e-9, 64*precision) if adjoint_tolerance is None else adjoint_tolerance
    for name, value in (("tolerance", tolerance), ("adjoint_tolerance", adjoint_tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    # Work entirely in real coordinates: complex FIT is not holomorphic.
    complex_leaves = [jnp.iscomplexobj(x) for x in jax.tree.leaves(guess_params)]
    if any(complex_leaves) and not all(complex_leaves):
        raise ValueError("Implicit fitting requires a uniformly real or complex guess parameter chart")
    if any(jnp.iscomplexobj(x) for x in jax.tree.leaves(target_params)) and not all(complex_leaves):
        raise ValueError("A complex implicit target requires a complex guess parameter chart")
    def real_tree(arrays):
        leaves = jax.tree.leaves(arrays)
        return tuple(jnp.stack((x.real, x.imag)) if complex_ else x
                     for x, complex_ in zip(leaves, complex_leaves))
    _, unravel_real = ravel_pytree(real_tree(guess_params))
    def encode(arrays):
        return ravel_pytree(real_tree(arrays))[0]
    def decode(vector):
        leaves = [x[0] + 1j*x[1] if complex_ else x
                  for x, complex_ in zip(unravel_real(vector), complex_leaves)]
        return jax.tree.unflatten(structure, leaves)

    def align(arrays, reference):
        # RL ends at site zero, with right-isometric tensors on sites 1..L-1.
        # Fix their residual unitary bond gauges by smooth polar alignment.
        # The composed polar derivative remains defined at repeated overlap
        # spectra (the identity at a fixed point). Its forward is unitary;
        # an unfinished Newton iteration would change the physical network.
        from ..backends.linalg_jax_projector import polar_unitary_jax

        tn = qtn.unpack(arrays, guess_skeleton)
        ref = qtn.unpack(reference, guess_skeleton)
        for site in range(length - 1, 0, -1):
            tensor, previous = tn[site], tn[site - 1]
            bond = tn.bond(site - 1, site)
            order = (bond, *(ix for ix in tensor.inds if ix != bond))
            matrix = tensor.transpose(*order).data
            shape = matrix.shape
            matrix = matrix.reshape((shape[0], -1))
            rmatrix = ref[site].transpose(*order).data.reshape(matrix.shape)
            overlap = rmatrix @ matrix.conj().T
            unitary = polar_unitary_jax(overlap)
            tensor.modify(data=(unitary @ matrix).reshape(shape), inds=order)
            tensor.transpose_like_(ref[site])
            order_left = (*(ix for ix in previous.inds if ix != bond), bond)
            data = previous.transpose(*order_left).data
            previous.modify(data=(data.reshape((-1, shape[0])) @ unitary.conj().T).reshape(data.shape), inds=order_left)
            previous.transpose_like_(ref[site - 1])
        return qtn.pack(tn)[0]

    def cycle(target_arrays, vector, reference):
        return encode(align(sweep(target_arrays, decode(vector), count=2), reference))
    def relative(a, b):
        return jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)

    def solve(target_arrays, initial):
        start = encode(initial)
        def condition(state):
            iteration, _, residual = state
            return (iteration < max_iterations) & (residual > tolerance)
        def body(state):
            iteration, vector, _ = state
            updated = cycle(target_arrays, vector, decode(vector))
            return iteration + 1, updated, relative(updated, vector)
        _, vector, _ = jax.lax.while_loop(condition, body,
                                          (jnp.asarray(0), start, jnp.asarray(jnp.inf, dtype=start.dtype)))
        residual = relative(cycle(target_arrays, vector, decode(vector)), vector)
        jax.debug.callback(lambda r: _require_convergence(r, tolerance, "FIT fixed point"), residual)
        return vector

    @jax.custom_vjp
    def implicit(target_arrays, initial):
        return decode(solve(target_arrays, initial))
    def forward(target_arrays, initial):
        vector = solve(target_arrays, initial)
        return decode(vector), (target_arrays, initial, vector)
    def backward(saved, cotangent):
        from jax.scipy.sparse.linalg import gmres

        target_arrays, initial, vector = saved
        reference = decode(vector)
        # Transpose the real coordinate decoder, respecting JAX's complex
        # cotangent convention. No forward-mode rule for raw FIT is needed.
        rhs = jax.vjp(decode, vector)[1](cotangent)[0]
        pullback = jax.vjp(lambda y: cycle(target_arrays, y, reference), vector)[1]
        matvec = lambda v: v - pullback(v)[0]
        adjoint, _ = gmres(matvec, rhs, tol=adjoint_tolerance*.1, atol=0.,
                           maxiter=adjoint_maxiter, restart=min(30, vector.size))
        denominator = jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
        residual = jnp.linalg.norm(matvec(adjoint) - rhs) / denominator
        jax.debug.callback(lambda r: _require_convergence(r, adjoint_tolerance, "FIT adjoint"), residual)
        gradient = jax.vjp(lambda theta: cycle(theta, vector, reference), target_arrays)[1](adjoint)[0]
        return gradient, jax.tree.map(jnp.zeros_like, initial)
    implicit.defvjp(forward, backward)
    def residual(theta, fitted):
        _check_parameters(theta, target_layout, "target_params")
        _check_parameters(fitted, guess_layout, "fitted_params")
        return relative(cycle(theta, encode(fitted), fitted), encode(fitted))
    return FunctionalFIT(target_params, guess_params, target_skeleton, guess_skeleton,
                         checked(implicit), residual, guess_layout)


def compress_jax_fit(
    tn, max_bond=None, cutoff=0., *, site_tags=None, bsz=1,
    differentiation="auto", max_iterations=None, tolerance=None,
    adjoint_tolerance=None, adjoint_maxiter=100, tn_fit="zipup",
    sweep_sequence="RL", sweep_reverse=False, optimize="greedy",
    canonize=True, permute_arrays=True, equalize_norms=False,
    normalize=False, inplace=False, tol=0., progbar=False,
    checkpoint=True, compress_opts=None,
):
    """Pepsy FIT compressor compatible with Quimb's public callable dispatch.

    Fit a single open 1D-like network, including layered PEPS boundaries.
    ``auto`` uses implicit differentiation for dense arrays and preserves
    Quimb's finite-sweep native Symmray fitting path. Explicit ``implicit``
    rejects native arrays until their gauge charts are validated.
    ``differentiation='unrolled'`` selects a fixed directional sweep budget.
    ``max_iterations`` defaults to 100 RL cycles or two finite sweeps,
    respectively. No random initialization, rank growth or dynamic cutoff is
    performed. Native Symmray arrays retain their graded metadata.
    """
    if cutoff != 0.:
        raise ValueError("Pepsy JAX FIT compression requires cutoff=0.0 (fixed ranks)")
    if tol != 0. or progbar:
        raise ValueError("Use tolerance for implicit FIT; unrolled FIT requires fixed sweeps (tol=0, progbar=False)")
    if compress_opts:
        raise ValueError("Pepsy JAX FIT does not support split overrides in compress_opts")
    if differentiation not in ("auto", "implicit", "unrolled"):
        raise ValueError("differentiation must be 'auto', 'unrolled' or 'implicit'")
    if not isinstance(tn, qtn.TensorNetwork):
        raise TypeError("Pepsy JAX FIT compression requires a single TensorNetwork")
    max_bond = _positive_integer("max_bond", max_bond)
    bsz = _positive_integer("bsz", bsz)
    if bsz not in (1, 2):
        raise ValueError("bsz must be 1 or 2")
    native = any(type(t.data).__module__.split('.', 1)[0] == "symmray" for t in tn)
    if differentiation == "auto":
        differentiation = "unrolled" if native else "implicit"
    if native and differentiation == "implicit":
        raise NotImplementedError("Implicit native Symmray gauge alignment is not implemented; use differentiation='unrolled'")
    if differentiation == "unrolled" and (tolerance is not None or adjoint_tolerance is not None
                                          or adjoint_maxiter != 100):
        raise ValueError("Implicit solver controls require differentiation='implicit' on a dense boundary")
    iterations = _positive_integer("max_iterations", (100 if differentiation == "implicit" else 2)
                                   if max_iterations is None else max_iterations)
    if native:
        return qtn.tensor_network_1d_compress(
            tn, method="fit", max_bond=max_bond, cutoff=0., bsz=bsz,
            max_iterations=iterations, tol=0., progbar=False, tn_fit=tn_fit,
            site_tags=site_tags, sweep_sequence=sweep_sequence,
            sweep_reverse=sweep_reverse, optimize=optimize, canonize=canonize,
            permute_arrays=permute_arrays, equalize_norms=equalize_norms,
            normalize=normalize, inplace=inplace,
        )
    tags = tuple(tn.site_tags if site_tags is None else site_tags)
    if len(tags) < 2 or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("Pepsy JAX FIT requires at least two ordered single site tags")
    if sweep_reverse:
        tags = tags[::-1]
    target = tn.copy()
    # Quimb's boundary callable supplies already-parsed site groups. Keep all
    # existing tags/outer legs, adding a private chain format only for FIT.
    tag_id = qtn.rand_uuid() + "{}"
    for site, tag in enumerate(tags):
        target.add_tag(tag_id.format(site), where=tag)
    # Exact local contraction/fusion turns the boundary's layered target into
    # the native chain representation expected by FIT. No target bond is
    # truncated; charge blocks and graded contraction order stay upstream.
    for site in range(len(tags)):
        target.contract_tags_(tag_id.format(site), which="any")
    target.fuse_multibonds_()
    if isinstance(tn_fit, qtn.TensorNetwork):
        guess = tn_fit.copy()
        for site, tag in enumerate(tags):
            guess.add_tag(tag_id.format(site), where=tag)
    else:
        if isinstance(tn_fit, str):
            guess_options = {"method": tn_fit}
        elif isinstance(tn_fit, dict):
            guess_options = dict(tn_fit)
        else:
            raise ValueError("Pepsy JAX FIT requires a tensor-derived tn_fit or supplied network")
        guess_options.setdefault("method", "zipup")
        guess_options.setdefault("max_bond", max_bond)
        guess_options.setdefault("cutoff", 0.)
        guess_options.update(site_tags=tags, optimize=optimize, permute_arrays=False, inplace=False)
        guess = qtn.tensor_network_1d_compress(target, **guess_options)
    for site in range(len(tags)):
        if len(guess.select_tensors(tag_id.format(site))) != 1:
            raise ValueError("Pepsy JAX FIT requires one guess tensor per boundary site")
    p = guess.view_as(qtn.MatrixProductState, L=len(tags), cyclic=False,
                      site_tag_id=tag_id, site_ind_id="__PEPSY_FIT_PHYS{}__")
    if p.max_bond() > max_bond:
        raise ValueError("tn_fit exceeds the requested max_bond; prepare a fixed-rank guess within the cap")
    wrapper = make_jax_fit(target, p, n_iter=iterations, block_size=bsz,
                           max_bond=p.max_bond(), sweep_sequence=sweep_sequence,
                           checkpoint=checkpoint, differentiation=differentiation,
                           max_iterations=iterations, tolerance=tolerance,
                           adjoint_tolerance=adjoint_tolerance,
                           adjoint_maxiter=adjoint_maxiter, contraction_opt=optimize)
    fitted = wrapper.to_network(wrapper(wrapper.target_params))
    fitted.drop_tags([tag_id.format(site) for site in range(len(tags))])
    result = tn if inplace else tn.copy()
    result.remove_all_tensors()
    result.add_tensor_network(fitted, virtual=True, check_collisions=False)
    result.exponent = fitted.exponent
    if normalize:
        result[tags[0]].normalize_()
        result.exponent = 0.
    elif equalize_norms is True:
        result.equalize_norms_()
    elif equalize_norms:
        result.equalize_norms_(value=equalize_norms)
    if permute_arrays and isinstance(result, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
        if isinstance(permute_arrays, str):
            result.permute_arrays(permute_arrays)
        else:
            result.permute_arrays()
    return result
