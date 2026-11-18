"""Single-layer regional environments for experimental PEPO splits."""

import math
from collections import OrderedDict

import numpy as np
import quimb.tensor as qtn
from scipy.optimize import least_squares, minimize
from scipy.sparse.linalg import LinearOperator, lsmr


class RegionalMapCache:
    """Bounded per-run cache whose maps exclude their active pair tensors."""

    def __init__(self, max_elements):
        self.max_elements = max_elements
        self.entries = OrderedDict()
        self.elements = 0
        self.hits = self.misses = self.evictions = self.invalidations = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key, value):
        size = value.size
        if size > self.max_elements:
            return
        while self.elements + size > self.max_elements:
            _, old = self.entries.popitem(last=False)
            self.elements -= old.size
            self.evictions += 1
        self.entries[key] = value
        self.elements += size

    def invalidate(self, changed_sites):
        changed = set(changed_sites)
        for key in tuple(self.entries):
            if not changed.issubset(key[0]):
                self.elements -= self.entries.pop(key).size
                self.invalidations += 1

    def report(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    invalidations=self.invalidations, entries=len(self.entries), elements=self.elements)


def l_regions(shape):
    """All three-site L shapes, including every orientation."""
    lx, ly = shape
    return [tuple(sorted(set(((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))) - {missing}))
            for x in range(lx - 1) for y in range(ly - 1)
            for missing in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))]


def _act(data, gate, axes):
    dims = tuple(data.shape[i] for i in axes)
    g = np.asarray(gate).reshape(*dims, *dims)
    rest = [i for i in range(data.ndim) if i not in axes]
    result = np.tensordot(g, data, axes=(tuple(range(len(axes), 2 * len(axes))), axes))
    return result.transpose(np.argsort(list(axes) + rest))


def _factor_jacobian(operator, a, b):
    """Real Jacobian of a complex linear map of the matrix product ``a @ b``."""
    tensor = operator.reshape(-1, a.shape[0], b.shape[1])
    ja = np.einsum("mij,kj->mik", tensor, b, optimize=True).reshape(len(operator), -1)
    jb = np.einsum("mij,ik->mkj", tensor, a, optimize=True).reshape(len(operator), -1)
    jac = np.concatenate((ja, jb), axis=1)
    return np.block([[jac.real, -jac.imag], [jac.imag, jac.real]])


def _als_refine(matrix, operator, rhs, a, b, region_sizes, tol, max_sweeps):
    """Improve a feasible pair with equality-constrained least-squares updates."""
    k = a.shape[1]
    sizes = np.cumsum(region_sizes)[:-1]

    def acceptable(candidate, old_error):
        if not np.isfinite(candidate).all():
            return False
        residual = operator @ candidate.ravel() - rhs
        regional = np.split(residual, sizes)
        if any(not np.isfinite(v).all() or np.linalg.norm(v) > tol for v in regional):
            return False
        return np.linalg.norm(candidate - matrix) < old_error

    sweeps = 0
    updates = 0
    shaped = operator.reshape(-1, *matrix.shape)
    for _ in range(max_sweeps):
        before = updates
        try:
            # Work in orthonormal row space: the objective in the free left
            # factor is ordinary Euclidean distance to M @ Q†.
            _, _, q = np.linalg.svd(b, full_matrices=False)
            left = matrix @ q.conj().T
            left_map = np.einsum("mij,kj->mik", shaped, q, optimize=True).reshape(len(operator), -1)
            correction = np.linalg.lstsq(left_map, left_map @ left.ravel() - rhs, rcond=None)[0]
            new_left = left - correction.reshape(left.shape)
            candidate = new_left @ q
            if acceptable(candidate, np.linalg.norm(a @ b - matrix)):
                a, b = new_left, q
                updates += 1

            # Mirror the same projection in an orthonormal column space.
            q, _, _ = np.linalg.svd(a, full_matrices=False)
            q = q[:, :k]
            right = q.conj().T @ matrix
            right_map = np.einsum("mij,ik->mkj", shaped, q, optimize=True).reshape(len(operator), -1)
            correction = np.linalg.lstsq(right_map, right_map @ right.ravel() - rhs, rcond=None)[0]
            new_right = right - correction.reshape(right.shape)
            candidate = q @ new_right
            if acceptable(candidate, np.linalg.norm(a @ b - matrix)):
                a, b = q, new_right
                updates += 1
        except np.linalg.LinAlgError:
            break
        sweeps += 1
        if before == updates:
            break
    return a, b, sweeps, updates


def _weighted_als_refine(matrix, hard, hard_rhs, soft, a, b, region_sizes, tol, max_sweeps,
                         max_pair_error):
    """Minimize weighted regional error in the nullspace of hard constraints."""
    shape = matrix.shape
    hard_tensor, soft_tensor = hard.reshape(-1, *shape), soft.reshape(-1, *shape)
    borders = np.cumsum(region_sizes)[:-1]

    def cost(pair):
        residual = soft @ (pair - matrix).ravel()
        return float(np.vdot(residual, residual).real)

    def step(h, w, current, rebuild):
        # A thin SVD builds only the row-space projector, not an n-by-n
        # nullspace matrix. Corrections therefore preserve the feasible seed.
        _, singular, vh = np.linalg.svd(h, full_matrices=False)
        threshold = (singular[0] * max(h.shape) * np.finfo(float).eps) if singular.size else 0.
        basis = vh[singular > threshold]

        def project(v):
            return v - basis.conj().T @ (basis @ v)

        def mv(v):
            return w @ project(v)

        def rmv(v):
            return project(w.conj().T @ v)

        linear = LinearOperator(w.shape, matvec=mv, rmatvec=rmv, dtype=np.complex128)
        residual = soft @ (matrix - rebuild(current)).ravel()
        delta = project(lsmr(linear, residual, atol=1e-11, btol=1e-11,
                             conlim=1e5, maxiter=200)[0])
        return rebuild(current + delta.reshape(current.shape))

    def acceptable(candidate, old):
        if not np.isfinite(candidate).all():
            return False
        if max_pair_error is not None and np.linalg.norm(candidate - matrix) > max_pair_error:
            return False
        residual = hard @ candidate.ravel() - hard_rhs
        if any(not np.isfinite(x).all() or np.linalg.norm(x) > tol
               for x in np.split(residual, borders)):
            return False
        new_cost = cost(candidate)
        return np.isfinite(new_cost) and new_cost < old

    def guarded_update(base, proposal, old_cost):
        for exponent in range(31):
            candidate = base + (proposal - base) * (0.5 ** exponent)
            if acceptable(candidate, old_cost):
                return candidate
        return None

    sweeps = updates = 0
    for _ in range(max_sweeps):
        before = updates
        try:
            _, _, q = np.linalg.svd(b, full_matrices=False)
            current = (a @ b) @ q.conj().T
            h = np.einsum("mij,kj->mik", hard_tensor, q, optimize=True).reshape(len(hard), -1)
            w = np.einsum("mij,kj->mik", soft_tensor, q, optimize=True).reshape(len(soft), -1)
            proposal = step(h, w, current, lambda z: z @ q)
            candidate = guarded_update(current @ q, proposal, cost(a @ b))
            if candidate is not None:
                a, b = candidate @ q.conj().T, q
                updates += 1

            q, _, _ = np.linalg.svd(a, full_matrices=False)
            current = q.conj().T @ (a @ b)
            h = np.einsum("mij,ik->mkj", hard_tensor, q, optimize=True).reshape(len(hard), -1)
            w = np.einsum("mij,ik->mkj", soft_tensor, q, optimize=True).reshape(len(soft), -1)
            proposal = step(h, w, current, lambda z: q @ z)
            candidate = guarded_update(q @ current, proposal, cost(a @ b))
            if candidate is not None:
                a, b = q, q.conj().T @ candidate
                updates += 1
        except np.linalg.LinAlgError:
            break
        sweeps += 1
        if updates == before:
            break
    return a, b, sweeps, updates


def _map(state, sites, pair_inds, pair_shape, region, optimize):
    """Materialize the exact single-layer map, with both pair tensors removed."""
    tensors = [state[s].copy() for s in state.gen_site_coos() if s not in sites]
    output = []
    region = set(region)
    active = set(sites)
    for site in state.gen_site_coos():
        upper, lower = state.upper_ind(*site), state.lower_ind(*site)
        dim_u = state[site].ind_size(upper)
        dim_l = state[site].ind_size(lower)
        if site not in region:
            if dim_u != dim_l:
                raise ValueError("partial trace requires square physical legs")
            tensors.append(qtn.Tensor(np.eye(dim_u), inds=(upper, lower)))
        else:
            for ind, dim in ((upper, dim_u), (lower, dim_l)):
                if site in active:
                    new = f"__pepsy_dmt_{site[0]}_{site[1]}_{len(output)}"
                    tensors.append(qtn.Tensor(np.eye(dim), inds=(ind, new)))
                    output.append(new)
                else:
                    output.append(ind)
    # Every pair index and every regional physical index remains open.
    result = qtn.tensor_contract(*tensors, output_inds=(*pair_inds, *output), optimize=optimize)
    return np.asarray(result.data).reshape(math.prod(pair_shape), -1).T


def _boundary_map(state, sites, pair_inds, pair_shape, region, optimize, *, chi, method, cutoff,
                  max_iterations=10):
    """Build one frozen linear map with open legs and compressed boundaries."""
    tn = state.copy()
    tn.exponent = 0.0  # The exterior scale cancels in the local equality constraints.
    region = set(region)
    aliases = {}
    for site in sites:
        tensor = tn[site]
        inds = tuple(i for i in tensor.inds if i in pair_inds)
        identities = []
        for index in inds:
            alias = qtn.rand_uuid()
            aliases[index] = alias
            identities.append(qtn.Tensor(np.eye(tensor.ind_size(index)), inds=(index, alias)))
        dummy = qtn.tensor_contract(*identities, output_inds=inds + tuple(aliases[i] for i in inds),
                                    optimize=optimize)
        tensor.modify(data=dummy.data, inds=dummy.inds)
    outputs = []
    for site in tn.gen_site_coos():
        upper, lower = tn.upper_ind(*site), tn.lower_ind(*site)
        tensor = tn[site]
        if site in region:
            outputs.extend((upper, lower))
        else:
            if tensor.ind_size(upper) != tensor.ind_size(lower):
                raise ValueError("partial trace requires square physical legs")
            traced = tensor.trace(upper, lower, preserve_tensor=True)
            tensor.modify(data=traced.data, inds=traced.inds)
    keep = tuple(aliases[i] for i in pair_inds) + tuple(outputs)
    options = {"max_iterations": max_iterations} if method == "dmrg" else {}
    contracted = tn.contract_boundary(
        max_bond=chi, cutoff=cutoff, mode="direct" if method == "direct" else "fit",
        around=tuple(set(sites) | region), final_contract=False, equalize_norms=False,
        **options,
    )
    result = contracted.contract(output_inds=keep, optimize=optimize, preserve_tensor=True)
    return np.asarray(result.data).reshape(math.prod(pair_shape), -1).T


def compress_pair(state, gate, sites, chi, regions, *, tol, max_nfev, optimize, max_map_elements,
                  boundary_chi=None, boundary_method="direct", boundary_cutoff=0.,
                  boundary_max_iterations=10, map_cache=None, refine=True,
                  refine_method="slsqp", refine_sweeps=4, objective_regions=None,
                  objective_max_pair_growth=2.):
    """Build frozen maps and seek a feasible rank-capped pair; do not mutate input."""
    left, right = (state[s] for s in sites)
    bonds = tuple(left.bonds(right))
    if len(bonds) != 1:
        raise ValueError("DMT requires one bond between the active sites")
    bond = bonds[0]
    left_inds = tuple(i for i in left.inds if i != bond)
    right_inds = tuple(i for i in right.inds if i != bond)
    pair_inds = left_inds + right_inds
    pair = qtn.tensor_contract(left, right, output_inds=pair_inds, optimize=optimize)
    shape = pair.shape
    upper = tuple(pair_inds.index(state.upper_ind(*s)) for s in sites)
    lower = tuple(pair_inds.index(state.lower_ind(*s)) for s in sites)
    data = _act(np.asarray(pair.data), gate, upper)
    data = _act(data, np.conj(gate), lower)
    nl, nr = math.prod(shape[:len(left_inds)]), math.prod(shape[len(left_inds):])
    matrix = data.reshape(nl, nr)
    if not np.isfinite(matrix).all():
        raise ValueError("PEPO DMT target contains nonfinite values")
    if boundary_chi is not None and max(nl * nl, nr * nr) > max_map_elements:
        raise ValueError("DMT boundary open-leg tensor exceeds max_map_elements")
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    k = min(chi, len(singular))
    left0 = u[:, :k] * np.sqrt(singular[:k])
    right0 = np.sqrt(singular[:k])[:, None] * vh[:k, :]
    all_regions = list(dict.fromkeys(regions + [r for r, _ in (objective_regions or [])]))
    region_maps = {}
    total_elements = sum(math.prod(state[s].ind_size(state.upper_ind(*s))
                                   * state[s].ind_size(state.lower_ind(*s)) for s in region)
                         * matrix.size for region in all_regions)
    if total_elements > max_map_elements:
        raise ValueError("DMT regional maps exceed max_map_elements; use fewer regions or a smaller system")
    for region in all_regions:
        key = (frozenset(sites), tuple(sites), pair_inds, tuple(shape), tuple(region),
               boundary_chi, boundary_method, boundary_cutoff, boundary_max_iterations)
        cached = map_cache.get(key) if map_cache is not None else None
        if cached is None:
            if boundary_chi is None:
                cached = _map(state, sites, pair_inds, shape, region, optimize)
            else:
                cached = _boundary_map(state, sites, pair_inds, shape, region, optimize,
                            chi=boundary_chi, method=boundary_method, cutoff=boundary_cutoff,
                            max_iterations=boundary_max_iterations)
            if map_cache is not None:
                map_cache.put(key, cached)
        region_maps[region] = cached
        if not np.isfinite(cached).all():
            raise ValueError("PEPO DMT environment contains nonfinite values")
    maps = [region_maps[region] for region in regions]
    if objective_regions is None:
        soft = None
    else:
        weighted = [np.sqrt(weight) * region_maps[region] for region, weight in objective_regions if weight > 0]
        soft = np.concatenate(weighted) if weighted else np.zeros((0, matrix.size), dtype=complex)
        if not np.isfinite(soft).all():
            raise ValueError("weighted PEPO DMT objective contains nonfinite values")
    # Stacking captures all requested equations; scaling keeps their residuals
    # comparable without changing their common zero set.
    targets = [op @ matrix.ravel() for op in maps]
    scales = [max(np.linalg.norm(t), np.linalg.norm(op) * np.linalg.norm(matrix) * 1e-14, 1e-100)
              for op, t in zip(maps, targets)]
    operator = np.concatenate([op / scale for op, scale in zip(maps, scales)])
    rhs = np.concatenate([t / scale for t, scale in zip(targets, scales)])
    nleft = nl * k

    def pack(a, b):
        values = np.concatenate((a.ravel(), b.ravel()))
        return np.concatenate((values.real, values.imag))

    def unpack(x):
        n = len(x) // 2
        z = x[:n] + 1j * x[n:]
        return z[:nleft].reshape(nl, k), z[nleft:].reshape(k, nr)

    def residual(x):
        a, b = unpack(x)
        delta = operator @ (a @ b).ravel() - rhs
        return np.concatenate((delta.real, delta.imag))

    def residual_jac(x):
        a, b = unpack(x)
        return _factor_jacobian(operator, a, b)

    x = pack(left0, right0)
    evaluations = 0
    retried = False
    if np.linalg.norm(residual(x)) > tol:
        result = least_squares(residual, x, jac=residual_jac, max_nfev=max_nfev, ftol=tol * 0.1,
                               xtol=tol * 0.1, gtol=tol * 0.1)
        x = result.x
        evaluations = result.nfev
    a, b = unpack(x)
    candidate = a @ b
    errors = [float(np.linalg.norm(op @ candidate.ravel() - target) / scale)
              for op, target, scale in zip(maps, targets, scales)]
    # A small gradient does not imply that every hard constraint meets tol.
    # Retry with tighter solver stopping thresholds using only the remaining
    # evaluation budget; the original per-region acceptance check still applies.
    if (np.isfinite(errors).all() and max(errors, default=0.0) > tol
            and evaluations and evaluations < max_nfev):
        stop = max(10 * np.finfo(float).eps, min(1e-13, tol * 1e-5))
        result = least_squares(residual, x, jac=residual_jac,
                               max_nfev=max_nfev - evaluations,
                               ftol=stop, xtol=stop, gtol=stop)
        x = result.x
        evaluations += result.nfev
        retried = True
        a, b = unpack(x)
        candidate = a @ b
        errors = [float(np.linalg.norm(op @ candidate.ravel() - target) / scale)
                  for op, target, scale in zip(maps, targets, scales)]
    if not np.isfinite(candidate).all() or not np.isfinite(errors).all() or max(errors, default=0.0) > tol:
        raise ValueError(f"PEPO DMT could not satisfy constraints at chi={chi}; "
                         f"largest relative residual {max(errors):.3g}")
    # Compress redundant complex equations before attempting to minimize the
    # local objective. Always keep the feasible seed if refinement fails.
    seed_pair_error = float(np.linalg.norm(candidate - matrix))
    max_pair_error = (None if soft is None or objective_max_pair_growth is None else
                      objective_max_pair_growth * max(seed_pair_error,
                                                      np.finfo(float).eps * np.linalg.norm(matrix)))
    def objective_value(pair):
        delta = pair - matrix
        value = delta.ravel() if soft is None else soft @ delta.ravel()
        return float(np.vdot(value, value).real)

    seed_objective = objective_value(candidate)
    refined = False
    als_sweeps = als_updates = 0
    lower_bound = np.linalg.norm(singular[k:])
    at_lower_bound = (np.linalg.norm(candidate - matrix) <= lower_bound + tol * max(np.linalg.norm(matrix), 1e-300)
                      if soft is None else seed_objective <= (
                          32 * np.finfo(float).eps * np.linalg.norm(soft) * np.linalg.norm(matrix)
                      ) ** 2)
    if refine and not at_lower_bound and refine_method == "als":
        if soft is None:
            new_a, new_b, als_sweeps, als_updates = _als_refine(
                matrix, operator, rhs, a, b, [len(op) for op in maps], tol, refine_sweeps,
            )
        else:
            new_a, new_b, als_sweeps, als_updates = _weighted_als_refine(
                matrix, operator, rhs, soft, a, b, [len(op) for op in maps], tol, refine_sweeps,
                max_pair_error,
            )
        new_candidate = new_a @ new_b
        new_errors = [float(np.linalg.norm(op @ new_candidate.ravel() - target) / scale)
                      for op, target, scale in zip(maps, targets, scales)]
        if (als_updates and np.isfinite(new_candidate).all() and np.isfinite(new_errors).all()
                and max(new_errors, default=0.0) <= tol
                and (max_pair_error is None or np.linalg.norm(new_candidate - matrix) <= max_pair_error)
                and objective_value(new_candidate) < seed_objective):
            a, b, candidate, errors = new_a, new_b, new_candidate, new_errors
            refined = True
        else:
            als_updates = 0
    if not refine or at_lower_bound or refine_method == "als":
        rank = 0  # The feasible seed already achieves the unconstrained SVD bound.
    else:
        _, spectrum, vh_map = np.linalg.svd(operator, full_matrices=False)
        rank = int(np.sum(spectrum > spectrum[0] * np.finfo(float).eps * max(operator.shape)))
    if 0 < 2 * rank < len(x):
        basis = vh_map[:rank]

        def constraints(v):
            la, rb = unpack(v)
            delta = basis @ (la @ rb - matrix).ravel()
            return np.concatenate((delta.real, delta.imag))

        def constraints_jac(v):
            la, rb = unpack(v)
            return _factor_jacobian(basis, la, rb)

        def objective(v):
            la, rb = unpack(v)
            return objective_value(la @ rb)

        def objective_jac(v):
            la, rb = unpack(v)
            delta = la @ rb - matrix
            grad = delta if soft is None else (soft.conj().T @ (soft @ delta.ravel())).reshape(matrix.shape)
            grad_a = 2 * grad @ rb.conj().T
            grad_b = 2 * la.conj().T @ grad
            return pack(grad_a, grad_b)

        result = minimize(objective, x, jac=objective_jac, method="SLSQP",
                          constraints={"type": "eq", "fun": constraints, "jac": constraints_jac},
                          options={"maxiter": min(max_nfev, 50), "ftol": tol * 0.1})
        new_a, new_b = unpack(result.x)
        new_candidate = new_a @ new_b
        new_errors = [float(np.linalg.norm(op @ new_candidate.ravel() - target) / scale)
                      for op, target, scale in zip(maps, targets, scales)]
        if (np.isfinite(new_candidate).all() and np.isfinite(new_errors).all()
                and max(new_errors, default=0.0) <= tol
                and (max_pair_error is None or np.linalg.norm(new_candidate - matrix) <= max_pair_error)
                and objective_value(new_candidate) < seed_objective):
            a, b, candidate, errors = new_a, new_b, new_candidate, new_errors
            refined = True
    work = state.copy()
    work[sites[0]].modify(data=a.reshape(*shape[:len(left_inds)], k), inds=(*left_inds, bond))
    work[sites[1]].modify(data=b.reshape(k, *shape[len(left_inds):]), inds=(bond, *right_inds))
    return work, {"solver": "regional-map-feasibility", "regions": len(regions),
                  "environment": "exact" if boundary_chi is None else "boundary-mps",
                  "boundary_chi": boundary_chi,
                  "boundary_method": boundary_method if boundary_chi is not None else None,
                  "max_density_residual": max(errors, default=0.0),
                  "pair_error": float(np.linalg.norm(candidate - matrix)),
                  "seed_pair_error": seed_pair_error, "bond_dimension": k,
                  "objective": "pair" if soft is None else "weighted_regions",
                  "seed_objective": seed_objective, "objective_value": objective_value(candidate),
                  "objective_max_pair_error": max_pair_error,
                  "objective_exponent": float(state.exponent),
                  "objective_refined": refined, "feasibility_evaluations": evaluations,
                  "feasibility_retried": retried, "refinement_enabled": refine,
                  "refinement_method": refine_method if refine else None,
                  "als_sweeps": als_sweeps, "als_updates": als_updates}
