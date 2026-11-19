"""Compare protected MPO evolution and MPS TEBD with exact L=16 references.

Run from the repository root with the selected environment active::

    python examples/dmt_ising_dynamics.py --output /tmp/dmt-ising

Hamiltonians use Pauli matrices, H = sum ZZ + hx sum X + hz sum Z.
The tilted case uses White et al.'s coupling ratios (arXiv:1707.01506),
and the near-y state follows their Eq. (32). This is a finite-chain test,
not a reproduction of their L=64/128 figures. Output includes exact
Hamiltonian and identical-gate state-vector references, raw measurements,
per-truncation audits, timing, and plots. No dense density matrix is built.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from pepsy.operators import PauliMPO
from pepsy.optimizers import MpoOptimizer, MpsOptimizer, PauliDMT


I, X, Y, Z = np.array([
    [[1, 0], [0, 1]], [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]], [[1, 0], [0, -1]],
], dtype=complex)
PAULIS = np.array([I, X, Y, Z])
MODELS = {"tfim": (1.05, 0.0), "tilted": (0.9045, 0.8090),
          "paper-pauli": (0.9045 / 2, 0.8090 / 2)}


def initial_vectors(n, kind="near-y"):
    if kind in {"near-y", "near-y-balanced", "near-y-figure"}:
        # Paper sites are one-based, including the period-eight sign pattern.
        g = [.1 * (-1 if (j + 1) % 8 in (1, 2, 7, 0) else 1) for j in range(n)]
        # The balanced amplitudes reproduce the published Fig. 4 points and
        # Fig. 7's initial energy; keep the literal printed Eq. (32) separate.
        phase = -1j if kind == "near-y-figure" else 1j
        vectors = np.array([[phase * (1 + value), 1 - value if kind != "near-y" else 1]
                            for value in g])
    elif kind == "z-plus":
        vectors = np.tile([1, 0], (n, 1)).astype(complex)
    elif kind == "neel":
        vectors = np.array([[1, 0] if j % 2 == 0 else [0, 1] for j in range(n)], dtype=complex)
    else:
        raise ValueError(f"unknown initial state: {kind}")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def product_vector(vectors):
    result = np.ones(1, dtype=complex)
    for vector in vectors:
        result = np.kron(result, vector)
    return result


def bond_hamiltonians(n, hx, hz):
    field = hx * X + hz * Z
    # Endpoint fields occur in one bond; interior fields occur in two.
    return [np.kron(Z, Z) + np.kron(field * (1 if j == 0 else .5), I)
            + np.kron(I, field * (1 if j == n - 2 else .5)) for j in range(n - 1)]


def sweep_gates(terms, dt, *, merge_turnaround=True):
    """Symmetric forward/backward sweep, matching paper-style TEBD ordering."""
    half = [expm(-.5j * dt * h) for h in terms]
    if not merge_turnaround:
        order = list(range(len(terms))) + list(reversed(range(len(terms))))
        return [(half[j], (j, j + 1)) for j in order]
    return ([(half[j], (j, j + 1)) for j in range(len(terms) - 1)]
            + [(expm(-1j * dt * terms[-1]), (len(terms) - 1, len(terms)))]
            + [(half[j], (j, j + 1)) for j in reversed(range(len(terms) - 1))])


def apply_vector(psi, gate, sites, n):
    """Independent bit-order-explicit state-vector gate application."""
    sites = tuple(sites)
    other = tuple(j for j in range(n) if j not in sites)
    order = sites + other
    panel = psi.reshape((2,) * n).transpose(order).reshape(2 ** len(sites), -1)
    return (gate @ panel).reshape((2,) * n).transpose(np.argsort(order)).reshape(-1)


def vector_marginal(psi, sites, n):
    other = tuple(j for j in range(n) if j not in sites)
    panel = psi.reshape((2,) * n).transpose(tuple(sites) + other).reshape(2 ** len(sites), -1)
    return panel @ panel.conj().T


def sparse_hamiltonian(n, hx, hz):
    """Build H independently from bit flips and diagonal Z eigenvalues."""
    indices = np.arange(2**n, dtype=np.int64)
    spins = [1 - 2 * ((indices >> (n - 1 - j)) & 1) for j in range(n)]
    diagonal = sum(spins[j] * spins[j + 1] for j in range(n - 1)) + hz * sum(spins)
    rows = [indices] + [indices ^ (1 << (n - 1 - j)) for j in range(n)]
    cols = [indices] * (n + 1)
    data = [diagonal] + [np.full(2**n, hx)] * n
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2**n, 2**n)).tocsr()


def mpo_cores(mpo):
    cores = []
    for j in range(mpo.L):
        bonds = ((mpo.bond(j - 1, j),) if j else ())
        bonds += ((mpo.bond(j, j + 1),) if j < mpo.L - 1 else ())
        a = mpo[j].transpose(*(bonds + (mpo.upper_ind(j), mpo.lower_ind(j)))).data
        dl = cores[-1].shape[1] if cores else 1
        cores.append(np.asarray(a).reshape(dl, -1, 2, 2))
    # Keep absolute trace drift visible; normalize only reported expectations.
    cores[0] = cores[0] * 10. ** mpo.exponent
    return cores


def pauli_cores(operator):
    return [np.einsum("abp,pij->abij", core, PAULIS)
            for core in operator.to_pauli_cores(copy=False)]


class Marginals:
    """Direct physical-index contraction, independent of DMT protected maps."""

    def __init__(self, cores):
        self.cores = cores
        self.left = [np.ones(1, dtype=complex)]
        for core in cores:
            self.left.append(self.left[-1] @ np.trace(core, axis1=2, axis2=3))
        self.right = [None] * (len(cores) + 1)
        self.right[-1] = np.ones(1, dtype=complex)
        for j in reversed(range(len(cores))):
            self.right[j] = np.trace(cores[j], axis1=2, axis2=3) @ self.right[j + 1]

    @property
    def trace(self):
        return complex(self.left[-1][0])

    def region(self, sites):
        sites = tuple(sites)
        if not sites:
            return np.array([[self.trace]])
        a, b = min(sites), max(sites)
        value = self.left[a]
        for j in range(a, b + 1):
            core = self.cores[j]
            if j in sites:
                value = np.moveaxis(np.tensordot(value, core, axes=(-1, 0)), -3, -1)
            else:
                value = value @ np.trace(core, axis1=2, axis2=3)
        value = value @ self.right[b + 1]
        k = len(sites)
        order = tuple(range(0, 2 * k, 2)) + tuple(range(1, 2 * k, 2))
        return value.transpose(order).reshape(2**k, 2**k)


def partial_trace(matrix, keep, n):
    value = matrix.reshape((2,) * (2 * n))
    size = n
    for j in reversed(range(n)):
        if j not in keep:
            value = np.trace(value, axis1=j, axis2=j + size)
            size -= 1
    return value.reshape(2**size, 2**size)


def local_data(triples, hx, hz, trace):
    """Derive every site/bond observable from normalized three-site densities."""
    n = len(triples) + 2
    one, two = [], []
    for j in range(n):
        start = min(j, n - 3)
        one.append(partial_trace(triples[start], (j - start,), 3))
    for j in range(n - 1):
        start = min(j, n - 3)
        two.append(partial_trace(triples[start], (j - start, j + 1 - start), 3))
    xyz = np.einsum("aij,pji->ap", one, PAULIS[1:])
    zz = np.einsum("aij,ji->a", two, np.kron(Z, Z))
    density = zz + .5 * hx * (xyz[:-1, 0] + xyz[1:, 0])
    density += .5 * hz * (xyz[:-1, 2] + xyz[1:, 2])
    density[0] += .5 * (hx * xyz[0, 0] + hz * xyz[0, 2])
    density[-1] += .5 * (hx * xyz[-1, 0] + hz * xyz[-1, 2])
    energy = density.sum()
    hermitian = (triples + triples.conj().transpose(0, 2, 1)) / 2
    return {
        "trace": trace, "energy": energy, "xyz": xyz, "zz": zz,
        "energy_density": density,
        "energy_mode": np.dot(np.cos(np.pi * (np.arange(n - 1) + .5) / 4), density) / (n - 1),
        "energy_fourier": -np.dot(np.exp(1j * np.pi * (np.arange(n - 1) + 1) / 4), density) / n,
        "triple_rdm": triples,
        "max_local_hermiticity_defect": np.max(np.linalg.norm(
            triples - triples.conj().transpose(0, 2, 1), axis=(1, 2))),
        "min_local_eigenvalue": min(np.linalg.eigvalsh(hermitian).min(), 0.),
    }


def measure_vector(psi, n, hx, hz):
    norm = np.vdot(psi, psi)
    triples = np.array([vector_marginal(psi, tuple(range(j, j + 3)), n)
                        for j in range(n - 2)]) / norm
    return local_data(triples, hx, hz, norm)


def measure_mpo(cores, hx, hz):
    panels = Marginals(cores)
    triples = np.array([panels.region(range(j, j + 3)) for j in range(len(cores) - 2)])
    return local_data(triples / panels.trace, hx, hz, panels.trace)


def measure_mps(mps, hx, hz):
    """Three-site pure-state marginals without a whole-chain dense vector."""
    cores = []
    for j in range(mps.L):
        bonds = ((mps.bond(j - 1, j),) if j else ())
        bonds += ((mps.bond(j, j + 1),) if j < mps.L - 1 else ())
        array = np.asarray(mps[j].transpose(*(bonds + (mps.site_ind(j),))).data)
        dl = cores[-1].shape[1] if cores else 1
        cores.append(array.reshape(dl, -1, 2))
    left = [np.ones((1, 1), dtype=complex)]
    for core in cores:
        left.append(np.einsum("ab,acs,bds->cd", left[-1], core, core.conj(), optimize=True))
    right = [None] * (mps.L + 1)
    right[-1] = np.ones((1, 1), dtype=complex)
    for j in reversed(range(mps.L)):
        right[j] = np.einsum("acs,bds,cd->ab", cores[j], cores[j].conj(), right[j + 1], optimize=True)
    triples = []
    for j in range(mps.L - 2):
        panel = np.einsum("abs,bct,cdu->astud", *cores[j:j + 3], optimize=True)
        panel = panel.reshape(cores[j].shape[0], 8, cores[j + 2].shape[1])
        triples.append(np.einsum("ab,apr,bqs,rs->pq", left[j], panel, panel.conj(), right[j + 3], optimize=True))
    norm = left[-1][0, 0]
    represented_norm = norm * 10. ** (2 * mps.exponent)
    return local_data(np.array(triples) / norm, hx, hz, represented_norm)


def gated_triples(old, gate, where):
    """Exact local target of one gate, using at most four-site contractions."""
    expected = []
    for j in range(len(old.cores) - 2):
        region = tuple(range(j, j + 3))
        if not set(region).intersection(where):
            target = old.region(region)
        else:
            union = tuple(sorted(set(region).union(where)))
            # At most four sites: explicitly embed U in the union density.
            basis = np.eye(2**len(union), dtype=complex)
            locations = tuple(union.index(q) for q in where)
            embedded = np.column_stack([apply_vector(v, gate, locations, len(union))
                                        for v in basis.T])
            density = embedded @ old.region(union) @ embedded.conj().T
            target = partial_trace(density, tuple(union.index(q) for q in region), len(union))
        expected.append(target)
    return np.array(expected)


@dataclass
class Audit:
    checked_gates: int = 0
    max_triple_defect: float = 0.
    max_energy_jump: float = 0.
    max_trace_defect: float = 0.
    seconds: float = 0.

    def check(self, before, after, gate, where, hx, hz):
        start = perf_counter()
        old, new = Marginals(before), Marginals(after)
        expected = gated_triples(old, gate, where)
        for j, target in enumerate(expected):
            region = tuple(range(j, j + 3))
            self.max_triple_defect = max(self.max_triple_defect,
                                        float(np.linalg.norm(new.region(region) - target)))
        actual = np.array([new.region(range(j, j + 3)) for j in range(len(after) - 2)])
        e_target = local_data(np.array(expected), hx, hz, old.trace)["energy"]
        e_actual = local_data(actual, hx, hz, new.trace)["energy"]
        self.max_energy_jump = max(self.max_energy_jump, float(abs(e_actual - e_target)))
        self.max_trace_defect = max(self.max_trace_defect, float(abs(new.trace - old.trace)))
        self.checked_gates += 1
        self.seconds += perf_counter() - start


def references(vectors, hx, hz, gates, dt, steps, sample_steps):
    n = len(vectors)
    initial = product_vector(vectors)
    start = perf_counter()
    h = sparse_hamiltonian(n, hx, hz)
    # Equally spaced sample times let SciPy reuse its exponential action work.
    continuous = expm_multiply(-1j * h, initial, start=0., stop=dt * steps,
                               num=len(sample_steps), traceA=0.)
    exact = [measure_vector(v, n, hx, hz) for v in continuous]
    psi = initial.copy()
    trotter = [measure_vector(psi, n, hx, hz)]
    for step in range(1, steps + 1):
        for gate, where in gates:
            psi = apply_vector(psi, gate, where, n)
        if step in sample_steps:
            trotter.append(measure_vector(psi, n, hx, hz))
    print(json.dumps({"reference_seconds": perf_counter() - start}), flush=True)
    return {"exact": exact, "trotter": trotter}


def evolve(method, vectors, gates, chi, hx, hz, steps, sample_steps, audit_steps, *,
           truncation="paper", refinement_sweeps=0, audit_all=False, backend="numpy",
           preservation_check="raise", preservation_log=None, check_regions=True):
    n = len(vectors)
    audit = Audit()
    tracked = {"discarded_physical_norm_max": 0., "constraint_relative_max": 0.,
               "lossy_splits": 0, "preservation_error_count": 0,
               "preservation_error_examples": [], "max_reported_trace_error": None}
    locations = {0, len(gates) // 2, len(gates) - 1}
    step_now = 0
    gate_now = 0
    if backend == "torch":
        # MPS diagnostics need this optional Autoray operation in a fresh
        # process too; an earlier Pauli-DMT import must not supply it by luck.
        from pepsy.backends import reg_stop_gradient_torch
        reg_stop_gradient_torch()

    def selected(step, index):
        return audit_all or (step in audit_steps and index in locations)

    class ObservedMpo(MpoOptimizer):
        def _check_finite(self, finite_check, p=None):
            nonlocal gate_now
            super()._check_finite(finite_check, p)
            if selected(step_now, gate_now):
                gate, where = gates[gate_now]
                audit.check(mpo_cores(self.p), mpo_cores(p), gate, where, hx, hz)
            gate_now += 1

    def convert(array, *, real=False):
        if backend == "numpy":
            return np.asarray(array, dtype=np.float64 if real else np.complex128)
        if backend != "torch":
            raise ValueError(f"unsupported benchmark backend: {backend}")
        import torch
        return torch.tensor(np.array(array, copy=True), dtype=torch.float64 if real else torch.complex128,
                            device="cpu")

    native_gates = [(convert(gate), where) for gate, where in gates]
    initial_mpo = qtn.MPO_product_operator([np.outer(v, v.conj()) for v in vectors])
    initial_mpo.apply_to_arrays(convert)
    if method.startswith("mps"):
        mode = "svd" if method == "mps-svd" else "direct"
        initial_mps = qtn.MPS_product_state(vectors)
        initial_mps.apply_to_arrays(convert)
        opt = MpsOptimizer(initial_mps, native_gates, chi=chi, mode=mode,
                           contraction_opt="greedy")
    elif method == "mpo-dmt":
        # MpoOptimizer uses the transposed public payload convention.
        opt = ObservedMpo(initial_mpo, [(u.T, w) for u, w in native_gates], chi=chi,
                          mode="dmt", contraction_opt="greedy")
    elif method.startswith("pauli-dmt"):
        cores = [np.einsum("pij,ji->p", PAULIS, np.outer(v, v.conj()))[None, None] / 2
                 for v in vectors]
        if method == "pauli-dmt-real":
            # A Hermitian density has real Pauli coefficients. Real storage
            # keeps that invariant through otherwise degenerate SVD choices.
            cores = [core.real for core in cores]
        cores = [convert(core, real=method == "pauli-dmt-real") for core in cores]
        opt = PauliDMT(PauliMPO.from_pauli_cores(cores), max_bond=chi, cutoff=0.,
                       diagnostics=True, truncation=truncation,
                       max_refinement_sweeps=refinement_sweeps,
                       preservation_check=preservation_check, check_regions=check_regions)
    else:
        raise ValueError(method)

    def read():
        if method.startswith("mps"):
            return measure_mps(opt.p, hx, hz)
        cores = mpo_cores(opt.p) if method == "mpo-dmt" else pauli_cores(opt.state)
        return measure_mpo(cores, hx, hz)

    def record(reports, physical_scale):
        for report_index, report in enumerate(reports):
            for bond in report.bond_reports:
                tracked["preservation_error_count"] += len(bond.preservation_errors)
                if bond.preservation_errors and preservation_log is not None:
                    preservation_log({"method": method, "step": step_now,
                                      "gate_index": report_index if method == "mpo-dmt" else gate_now,
                                      "bond": bond.bond, "errors": bond.preservation_errors,
                                      "relative_trace_error": bond.trace_relative_error})
                for message in bond.preservation_errors:
                    if len(tracked["preservation_error_examples"]) < 8:
                        tracked["preservation_error_examples"].append(message)
                if bond.trace_relative_error is not None:
                    tracked["max_reported_trace_error"] = max(tracked["max_reported_trace_error"] or 0.,
                                                               bond.trace_relative_error)
                error = bond.discarded_weight * physical_scale
                tracked["discarded_physical_norm_max"] = max(tracked["discarded_physical_norm_max"], error)
                tracked["constraint_relative_max"] = max(tracked["constraint_relative_max"],
                                                          bond.relative_constraint_error or 0.)
                tracked["lossy_splits"] += int(error > 1e-10)

    measurements = [read()]
    seconds, readout_seconds = 0., 0.
    for step_now in range(1, steps + 1):
        start = perf_counter()
        audit_before = audit.seconds
        if method.startswith("mps"):
            opt.run(cutoff=0., cutoff_mode="rel", stabilize_unitary=True)
        elif method == "mpo-dmt":
            gate_now = 0
            opt.run(cutoff=0., cutoff_mode="rel", fidelity_samples=0, dmt_diagnostics=True,
                    dmt_truncation=truncation, dmt_max_refinement_sweeps=refinement_sweeps,
                    dmt_preservation_check=preservation_check, dmt_check_regions=check_regions)
            record(opt.get_dmt_reports(), 1.)
        else:
            for index, (gate, where) in enumerate(gates):
                gate_now = index
                preparation = perf_counter()
                before = pauli_cores(opt.state) if selected(step_now, index) else None
                if before is not None:
                    audit.seconds += perf_counter() - preparation
                try:
                    opt.apply_gate(native_gates[index][0], where)
                except FloatingPointError as exc:
                    exc.add_note(f"{method}: step={step_now}, gate={index}, sites={where}, backend={backend}")
                    raise
                record((opt.report,), 2. ** (n / 2))
                if before is not None:
                    preparation = perf_counter()
                    after = pauli_cores(opt.state)
                    audit.seconds += perf_counter() - preparation
                    audit.check(before, after, gate, where, hx, hz)
        seconds += perf_counter() - start - (audit.seconds - audit_before)
        if step_now in sample_steps:
            start = perf_counter()
            measurements.append(read())
            readout_seconds += perf_counter() - start
        if step_now == steps or step_now % max(1, steps // 5) == 0:
            print(json.dumps({"method": method, "chi": chi, "step": step_now,
                              "steps": steps, "evolution_seconds": seconds}), flush=True)
    bonds = (opt.state.pauli_bond_dimensions if method.startswith("pauli-dmt") else opt.p.bond_sizes())
    if (max(bonds) > chi
            or not all(np.all(np.isfinite(row["triple_rdm"])) for row in measurements)):
        raise AssertionError(f"{method}: invalid bond cap or nonfinite trajectory")
    benchmark_issues = []
    if (audit.max_triple_defect > 1e-9 or audit.max_energy_jump > 1e-9
            or max(abs(row["trace"] - 1) for row in measurements) > 1e-8):
        message = (f"{method}: benchmark preservation thresholds exceeded: "
                   f"audit={asdict(audit)}, max_trace_drift="
                   f"{max(abs(row['trace'] - 1) for row in measurements)}")
        if preservation_check == "raise":
            raise AssertionError(message)
        benchmark_issues.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    arrays = (opt.state.to_pauli_cores(copy=False) if method.startswith("pauli-dmt")
              else [tensor.data for tensor in opt.p])
    dtype = "float64" if method == "pauli-dmt-real" else "complex128"
    if any(str(array.dtype).removeprefix("torch.") != dtype for array in arrays):
        raise AssertionError(f"{method}: benchmark storage precision changed")
    return measurements, {"method": method, "chi": chi, "backend": backend, "dtype": dtype,
                          "preservation_check": preservation_check, "check_regions": check_regions,
                          "benchmark_warnings": benchmark_issues,
                          "gate_dtype": str(native_gates[0][0].dtype), "device": "cpu",
                          "evolution_seconds": seconds,
                          "readout_seconds": readout_seconds, "final_max_bond": max(bonds),
                          "audit": asdict(audit), **tracked}


def pack(measurements):
    return {key: np.array([row[key] for row in measurements]) for key in measurements[0]}


def summary(data, refs, timing):
    exact, trotter = refs["exact"], refs["trotter"]
    result = dict(timing)
    result.update({
        "max_energy_drift_per_site": float(np.max(abs(data["energy"] - data["energy"][0])) / data["xyz"].shape[1]),
        "max_energy_error_per_site": float(np.max(abs(data["energy"] - exact["energy"])) / data["xyz"].shape[1]),
        "max_trace_defect": float(np.max(abs(data["trace"] - 1))),
        "max_imaginary_observable": float(max(np.max(abs(data[key].imag)) for key in ("energy", "xyz", "zz"))),
        "max_local_hermiticity_defect": float(np.max(data["max_local_hermiticity_defect"])),
        "min_local_eigenvalue": float(np.min(data["min_local_eigenvalue"])),
    })
    for label, ref in (("exact", exact), ("trotter", trotter)):
        for key in ("xyz", "zz", "energy_density", "energy_mode", "energy_fourier"):
            error = abs(data[key] - ref[key])
            result[f"{key}_rmse_vs_{label}"] = float(np.sqrt(np.mean(error**2)))
            result[f"{key}_max_error_vs_{label}"] = float(np.max(error))
        norms = np.linalg.norm(data["triple_rdm"] - ref["triple_rdm"], axis=(-1, -2))
        result[f"triple_rdm_rmse_vs_{label}"] = float(np.sqrt(np.mean(norms**2)))
        result[f"triple_rdm_final_rmse_vs_{label}"] = float(np.sqrt(np.mean(norms[-1]**2)))
    return result


def plot_results(path, model, times, curves, *, filename_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps = sorted({int(label.split("chi=")[-1]) for label in curves if "chi=" in label})
    if len(caps) > 1:
        for cap in caps:
            subset = {label: data for label, data in curves.items()
                      if "chi=" not in label or label.endswith(f"chi={cap}")}
            plot_results(path, f"{model}-chi{cap}", times, subset, filename_prefix=filename_prefix)
        return
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 11, "lines.linewidth": 1.8})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    exact = curves["exact"]
    colors = {method: color for method, color in zip(
        ("mps-svd", "mps-direct", "mpo-dmt", "pauli-dmt", "pauli-dmt-real"),
        plt.cm.viridis(np.linspace(.05, .9, 5)))}
    for label, data in curves.items():
        method = label.split(" chi=")[0]
        style = {"color": colors[method]} if method in colors else {}
        if method == "mps-svd":
            style["linestyle"] = ":"
        if label == "exact":
            style = {"linewidth": 2.5, "color": "black"}
        if label == "trotter":
            style = {"color": "grey", "linestyle": "--"}
        n = data["xyz"].shape[1]
        axes[0, 0].plot(times, (data["energy"].real - exact["energy"][0].real) / n, label=label, **style)
        axes[0, 1].plot(times, data["xyz"][:, n // 2, 2].real, **style)
        axes[0, 2].plot(times, data["energy_mode"].real, **style)
        if label != "exact":
            axes[1, 0].semilogy(times[1:], np.maximum(1e-16, np.sqrt(np.mean(abs(data["xyz"][1:] - exact["xyz"][1:])**2, axis=(1, 2)))), **style)
            error = np.linalg.norm(data["triple_rdm"] - exact["triple_rdm"], axis=(-1, -2))
            axes[1, 1].semilogy(times[1:], np.maximum(1e-16, np.sqrt(np.mean(error[1:]**2, axis=1))), **style)
        axes[1, 2].plot(times, data["min_local_eigenvalue"], **style)
    titles = ("Energy drift per site", "Central Z", "Energy-density cosine mode k=pi/4",
              "All-site X/Y/Z RMSE vs exact H", "Three-site RDM RMSE vs exact H", "Minimum eigenvalue: Hermitian RDM part")
    labels = (r"$(E(t)-E(0))/L$", r"$\langle Z_{L/2}\rangle$", r"$e_{\pi/4}$",
              "Observable RMSE", "RDM Frobenius RMSE", "Three-site minimum eigenvalue")
    for ax, title, ylabel in zip(axes.flat, titles, labels):
        ax.set_title(title)
        ax.set_xlabel("Time (Pauli ZZ coupling = 1)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4)
    fig.suptitle(f"L={len(exact['xyz'][0])} {model}: equal bond cap, unequal representation cost", fontsize=17)
    fig.savefig(path / f"{filename_prefix}{model}.png", dpi=300)
    fig.savefig(path / f"{filename_prefix}{model}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["tfim", "tilted"])
    parser.add_argument("--chis", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--methods", nargs="+", choices=("mps-svd", "mps-direct", "mpo-dmt", "pauli-dmt", "pauli-dmt-real"),
                        default=["mps-svd", "mps-direct", "mpo-dmt", "pauli-dmt-real"])
    parser.add_argument("--truncation", choices=("paper", "optimized"), default="paper")
    parser.add_argument("--refinement-sweeps", type=int, default=0)
    parser.add_argument("--audit-all", action="store_true", help="independently check every DMT gate")
    parser.add_argument("--dt", type=float, default=.05)
    parser.add_argument("--time", type=float, default=10.)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--unmerged-turnaround", action="store_true",
                        help="keep both half-gate truncations at the sweep turnaround, as in Appendix A")
    parser.add_argument("--initial", choices=("near-y", "near-y-balanced", "near-y-figure", "z-plus", "neel"), default="near-y")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                        help="CPU evolution backend; complex128 except explicit real-Pauli storage")
    parser.add_argument("--preservation-check", choices=("raise", "warn", "off"), default="warn")
    parser.add_argument("--skip-region-checks", action="store_true",
                        help="skip full marginal verification, keeping bond checks unless policy is off")
    parser.add_argument("--cpu-svd", choices=("torch", "scipy_gesdd", "scipy_gesvd"), default="torch",
                        help="Torch CPU SVD forward driver")
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-ising"))
    args = parser.parse_args()
    steps = round(args.time / args.dt) if args.dt > 0 else 0
    if (args.length < 3 or args.dt <= 0 or steps < 1 or args.sample_every < 1
            or steps % args.sample_every or not np.isclose(steps * args.dt, args.time)
            or min(args.chis) < 1 or args.threads < 1 or args.refinement_sweeps < 0):
        parser.error("require L>=3, positive dt/time/chi/threads, and time divisible by dt*sample-every")
    args.output.mkdir(parents=True, exist_ok=True)
    error_path = args.output / "preservation-errors.jsonl"
    error_path.write_text("")

    def log_preservation_error(record):
        with error_path.open("a") as stream:
            stream.write(json.dumps({"model": model, "chi": chi, **record}) + "\n")

    config = vars(args) | {"output": str(args.output), "python": platform.python_version(),
                           "hamiltonians": {name: {"J": 1., "hx": MODELS[name][0], "hz": MODELS[name][1]}
                                            for name in args.models},
                           "schedule": "symmetric forward/backward bond sweep",
                           "cutoff": 0., "constraint_mode": "strict",
                           "versions": {name: version(name) for name in ("numpy", "scipy", "quimb", "autoray", "cotengra")}}
    if args.backend == "torch":
        config["versions"]["torch"] = version("torch")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    from threadpoolctl import threadpool_limits
    summaries = []
    with threadpool_limits(limits=args.threads):
        if args.backend == "torch":
            import torch
            torch.set_num_threads(args.threads)
            from pepsy.backends import TorchLinalgConfig
            TorchLinalgConfig(cpu_svd=args.cpu_svd).register()
        vectors = initial_vectors(args.length, args.initial)
        sample_steps = set(range(0, steps + 1, args.sample_every))
        times = np.arange(0, steps + 1, args.sample_every) * args.dt
        audit_steps = {1, max(1, steps // 3), max(1, 2 * steps // 3), steps}
        for model in args.models:
            hx, hz = MODELS[model]
            gates = sweep_gates(bond_hamiltonians(args.length, hx, hz), args.dt,
                                merge_turnaround=not args.unmerged_turnaround)
            print(json.dumps({"model": model, "hx": hx, "hz": hz}), flush=True)
            curves = {key: pack(value) for key, value in references(vectors, hx, hz, gates, args.dt, steps, sample_steps).items()}
            summaries.append({"model": model, **summary(curves["trotter"], curves, {"method": "trotter", "chi": None})})
            for chi in args.chis:
                for method in args.methods:
                    measured, timing = evolve(method, vectors, gates, chi, hx, hz, steps, sample_steps, audit_steps,
                                              truncation=args.truncation, refinement_sweeps=args.refinement_sweeps,
                                              audit_all=args.audit_all, backend=args.backend,
                                              preservation_check=args.preservation_check,
                                              preservation_log=log_preservation_error,
                                              check_regions=not args.skip_region_checks)
                    label = f"{method} chi={chi}"
                    curves[label] = pack(measured)
                    row = {"model": model, **summary(curves[label], curves, timing)}
                    summaries.append(row)
                    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                    np.savez_compressed(args.output / f"{model}-{method}-chi{chi}.npz", times=times, **curves[label])
                    print(json.dumps(row), flush=True)
            for label in ("exact", "trotter"):
                np.savez_compressed(args.output / f"{model}-{label}.npz", times=times, **curves[label])
            plot_results(args.output, model, times, curves)


if __name__ == "__main__":
    main()
