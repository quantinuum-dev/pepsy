"""Exact-reference 2D Ising dynamics in MPO storage (not native PEPOs).

Example: python examples/dmt_2d_dynamics.py --shape 3 2 --chis 32 48 64
    --steps 100 --dt .1 --methods dmt fit --output /tmp/dmt-2d.json

Every physical nearest-neighbor edge is included. Continuous-time and identical
Trotter statevectors are independent references for the depolarized pure input.
Raw trace, Hermiticity and positivity remain visible; no state repair is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn
from scipy.linalg import eigh, expm
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits

from pepsy.operators import square_lattice_dmt_regions
from pepsy.optimizers import MpoOptimizer
from pepsy.tensors import OneDMap

try:
    from examples import dmt_ising_dynamics as ref
    from examples.dmt_window_scaling import gated_window
except ModuleNotFoundError:
    import dmt_ising_dynamics as ref
    from dmt_window_scaling import gated_window


def problem(lx, ly, dt, *, hx=1.05, hz=.5, pure_weight=.6, map_mode="snake-row-major",
            initial_kind="depolarized-pure", polarization=.4, dense_reference=True):
    """Distribute each site's field over its incident physical edges once."""
    limit = 10 if dense_reference else 20
    if min(lx, ly) < 2 or lx * ly > limit:
        raise ValueError(f"Benchmark requires both sides >=2 and at most {limit} sites in this reference mode")
    if not dense_reference and initial_kind != "depolarized-pure":
        raise ValueError("Sparse statevector references require the depolarized-pure input")
    if not 0 <= pure_weight <= 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require a positive finite dt and pure_weight in [0, 1]")
    if initial_kind not in {"depolarized-pure", "product-mixed"} or not 0 <= polarization <= 1:
        raise ValueError("Unknown initial_kind or polarization outside [0, 1]")
    n = lx * ly
    mapper = OneDMap(lx, ly, mode=map_mode)
    coords, mapping = mapper.build()
    physical_edges = ([((x, y), (x + 1, y)) for x in range(lx - 1) for y in range(ly)]
                      + [((x, y), (x, y + 1)) for x in range(lx) for y in range(ly - 1)])
    edges = [tuple(mapping[c] for c in edge) for edge in physical_edges]
    degree = Counter(s for edge in edges for s in edge)
    field = hx * ref.X + hz * ref.Z
    terms = [np.kron(ref.Z, ref.Z) + np.kron(field / degree[a], ref.I)
             + np.kron(ref.I, field / degree[b]) for a, b in edges]
    half = [expm(-.5j * dt * h) for h in terms]
    # Symmetric second-order product; merge only the identical turnaround pair.
    gates = list(zip(half[:-1], edges[:-1]))
    gates += [(expm(-1j * dt * terms[-1]), edges[-1])]
    gates += list(reversed(list(zip(half[:-1], edges[:-1]))))
    vectors = []
    for j in range(n):
        x, y = coords[j]
        theta = np.pi / 2 + .35 * (-1)**(x + y)
        phi = .7 + .3 * np.cos(np.pi * x / (lx - 1))
        vectors.append(np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)]))
    psi = ref.product_vector(vectors)
    pure = qtn.MPO_product_operator([np.outer(v, v.conj()) for v in vectors])
    initial = pure_weight * pure + (1 - pure_weight) / 2**n * qtn.MPO_identity(n)
    # A 16-spin density has 2^32 entries. The sparse-reference caller uses
    # only a statevector plus the analytically invariant identity component.
    initial_density = density(psi, pure_weight) if dense_reference else None
    if initial_kind == "product-mixed":
        factors = [(1 - polarization) * ref.I / 2 + polarization * np.outer(v, v.conj()) for v in vectors]
        initial = qtn.MPO_product_operator(factors)
        initial_density = np.ones((1, 1))
        for factor in factors:
            initial_density = np.kron(initial_density, factor)
    # Independent bit-flip Hamiltonian, not a sum of the Trotter matrices.
    basis = np.arange(2**n)
    z = [1 - 2 * ((basis >> (n - 1 - j)) & 1) for j in range(n)]
    zz_diagonal = sum(z[a] * z[b] for a, b in edges)
    diagonal = zz_diagonal + hz * sum(z)
    if dense_reference:
        hamiltonian = np.diag(diagonal).astype(complex)
        for j in range(n):
            hamiltonian[basis ^ (1 << (n - 1 - j)), basis] += hx
    else:
        rows = [basis] + [basis ^ (1 << (n - 1 - j)) for j in range(n)]
        data = [diagonal] + [np.full(2**n, hx)] * n
        hamiltonian = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.tile(basis, n + 1))),
                                 shape=(2**n, 2**n)).tocsr().astype(complex)
    regions = square_lattice_dmt_regions(lx, ly, mapper=mapper, kind="plaquettes")
    return dict(n=n, edges=edges, terms=terms, gates=gates, psi=psi, initial=initial,
                hamiltonian=hamiltonian, regions=regions, pure_weight=pure_weight,
                initial_kind=initial_kind, initial_density=initial_density,
                hx=hx, hz=hz, zz_diagonal=zz_diagonal)


def density(psi, weight):
    return weight * np.outer(psi, psi.conj()) + (1 - weight) / len(psi) * np.eye(len(psi))


def apply_density(matrix, gate, where, n):
    """Independent tensor-axis ket and conjugate-bra gate action."""
    value = matrix.reshape((2,) * (2 * n))
    for support, operator in [(tuple(where), gate), (tuple(n + j for j in where), gate.conj())]:
        order = support + tuple(j for j in range(2 * n) if j not in support)
        panel = value.transpose(order).reshape(2**len(support), -1)
        value = (operator @ panel).reshape((2,) * (2 * n)).transpose(np.argsort(order))
    return value.reshape(2**n, 2**n)


def exact_references(model, dt, steps, sample_every):
    values, vectors = eigh(model["hamiltonian"])
    amplitudes = vectors.conj().T @ model["psi"]
    trotter = model["psi"].copy()
    mixed = model["initial_kind"] == "product-mixed"
    if mixed:
        initial_eigenbasis = vectors.conj().T @ model["initial_density"] @ vectors
        trotter = model["initial_density"].copy()
    result = {}
    for step in range(steps + 1):
        if step:
            for gate, where in model["gates"]:
                trotter = (apply_density(trotter, gate, where, model["n"]) if mixed
                           else ref.apply_vector(trotter, gate, where, model["n"]))
        if step % sample_every == 0 or step == steps:
            phase = np.exp(-1j * step * dt * values)
            if mixed:
                exact = vectors @ (phase[:, None] * initial_eigenbasis * phase.conj()[None]) @ vectors.conj().T
                result[step] = exact, trotter.copy()
            else:
                exact = vectors @ (phase * amplitudes)
                result[step] = density(exact, model["pure_weight"]), density(trotter, model["pure_weight"])
    return result


def diagnostics(actual, exact, trotter, model):
    """Physical diagnostics; Hermitian-part spectra do not repair the state."""
    n = model["n"]
    trace = np.trace(actual)
    norm = np.linalg.norm(exact)
    regions = model["regions"]
    panels = [ref.partial_trace(actual, r, n) for r in regions]
    truth = [ref.partial_trace(exact, r, n) for r in regions]
    circuit = [ref.partial_trace(trotter, r, n) for r in regions]
    spectrum = np.linalg.eigvalsh((actual + actual.conj().T) / 2)
    one = np.array([ref.partial_trace(actual, (j,), n) for j in range(n)])
    one_exact = np.array([ref.partial_trace(exact, (j,), n) for j in range(n)])
    xyz = np.einsum("aij,pji->ap", one, ref.PAULIS[1:])
    xyz_exact = np.einsum("aij,pji->ap", one_exact, ref.PAULIS[1:])
    # Only the trace of these products is required, not dense O(2^(3N))
    # matrix multiplication during each independent readout.
    energy = np.einsum("ij,ji->", actual, model["hamiltonian"])
    exact_energy = np.einsum("ij,ji->", exact, model["hamiltonian"])
    return dict(
        trace_real=float(trace.real), trace_imag=float(trace.imag), trace_error=float(abs(trace - 1)),
        energy_per_site_real=float(energy.real / n),
        energy_per_site_imag=float(energy.imag / n),
        energy_error_per_site=float(abs(energy - exact_energy) / n),
        energy_error_per_site_trace_normalized=(float(abs(energy / trace - exact_energy) / n)
                                                if abs(trace) > 1e-14 else None),
        hs_relative_exact=float(np.linalg.norm(actual - exact) / norm),
        hs_relative_trotter=float(np.linalg.norm(actual - trotter) / norm),
        trotter_hs_relative=float(np.linalg.norm(trotter - exact) / norm),
        max_plaquette_error=float(max(np.linalg.norm(a - b) for a, b in zip(panels, truth))),
        max_plaquette_trotter_error=float(max(np.linalg.norm(a - b) for a, b in zip(panels, circuit))),
        trotter_max_plaquette_error=float(max(np.linalg.norm(a - b) for a, b in zip(circuit, truth))),
        max_xyz_error=float(np.max(abs(xyz - xyz_exact))),
        min_eigenvalue_hermitian_part=float(spectrum[0]),
        negativity_hermitian_part=float(-spectrum[spectrum < 0].sum()),
        min_plaquette_eigenvalue_hermitian_part=float(min(
            np.linalg.eigvalsh((a + a.conj().T) / 2)[0] for a in panels)),
        relative_hermiticity_defect=float(np.linalg.norm(actual - actual.conj().T) / norm),
    )


class AuditedOptimizer(MpoOptimizer):
    """Independent per-gate audit on selected timesteps, outside replay timing."""

    def __init__(self, *args, model, **kwargs):
        self.model = model
        self.audit = False
        self.audit_seconds = 0.
        self.gate_number = 0
        self.max_preservation_error = 0.
        self.audited_gates = 0
        self.audit_source = None
        super().__init__(*args, **kwargs)

    def _check_finite(self, finite_check, p=None):
        super()._check_finite(finite_check, p)
        if p is None:
            return
        if self.audit:
            started = perf_counter()
            gate, where = self.model["gates"][self.gate_number]
            # FIT keeps a local live `p` until the replay ends; self.p is not
            # its per-gate source. Own the preceding physical cores instead.
            before = self.audit_source
            after = ref.Marginals([a.copy() for a in ref.mpo_cores(p)])
            for region in self.model["regions"]:
                target = gated_window(before, gate, where, region)
                self.max_preservation_error = max(self.max_preservation_error,
                                                  float(np.linalg.norm(after.region(region) - target)))
            self.audited_gates += 1
            self.audit_source = after
            self.audit_seconds += perf_counter() - started
        self.gate_number += 1


def native_state(state, backend):
    if backend == "numpy":
        return state.copy()
    import torch
    state = state.copy()
    state.apply_to_arrays(lambda a: torch.tensor(np.array(a, copy=True), dtype=torch.complex128))
    return state


def trajectory(model, references, *, method, chi, steps, dt, backend="numpy", fit_sweeps=4,
               progress=False):
    queue = [(u.T, sites) for u, sites in model["gates"]]
    if backend == "torch":
        import torch
        queue = [(torch.tensor(u, dtype=torch.complex128), sites) for u, sites in queue]
    is_dmt = method in {"dmt", "optimized"}
    opt = AuditedOptimizer(
        native_state(model["initial"], backend), queue, chi=chi, mode="dmt" if is_dmt else "dmrg", model=model,
        contraction_opt="greedy",
    )
    row = dict(method=method, chi=chi, backend=backend, status="completed", completed_steps=0,
               update_seconds=0., readout_seconds=0., audit_seconds=0., samples=[],
               warning_count=0, warning_examples=[], max_bond_observed=0)
    audit_steps = {1, max(1, steps // 2), steps}
    for step in range(steps + 1):
        if step:
            started = perf_counter()
            audit_before = opt.audit_seconds
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    opt.audit = step in audit_steps
                    opt.gate_number = 0
                    if opt.audit:
                        audit_started = perf_counter()
                        observed = opt.p.copy()
                        observed.apply_to_arrays(lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else a)
                        opt.audit_source = ref.Marginals([a.copy() for a in ref.mpo_cores(observed)])
                        opt.audit_seconds += perf_counter() - audit_started
                    opts = dict(cutoff=0., target_cutoff=0., fidelity_samples=0)
                    if is_dmt:
                        opts.update(dmt_regions=model["regions"], dmt_preservation_check="warn",
                                    dmt_truncation="optimized" if method == "optimized" else "paper",
                                    dmt_max_refinement_sweeps=2)
                    else:
                        opts.update(n_iter=fit_sweeps, fit_init_strategy="direct", fit_rtol=None,
                                    fit_block_size=2, fit_adaptive_sweeps=2,
                                    fit_target_strategy="layered")
                    opt.run(**opts)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                row.update(status="failed", failure_step=step, error=f"{type(exc).__name__}: {exc}")
                break
            finally:
                audit = opt.audit_seconds - audit_before
                row["update_seconds"] += perf_counter() - started - audit
                row["warning_count"] += len(caught)
                for warning in caught:
                    message = str(warning.message)
                    if message not in row["warning_examples"] and len(row["warning_examples"]) < 4:
                        row["warning_examples"].append(message)
            row["completed_steps"] = step
            row["max_bond_observed"] = max(row["max_bond_observed"], opt.p.max_bond())
        if step in references:
            started = perf_counter()
            observed = opt.p.copy()
            observed.apply_to_arrays(lambda a: a.detach().cpu().numpy() if hasattr(a, "detach") else a)
            actual = ref.Marginals(ref.mpo_cores(observed)).region(tuple(range(model["n"])))
            bond = opt.p.max_bond()
            row["max_bond_observed"] = max(row["max_bond_observed"], bond)
            row["samples"].append(dict(step=step, time=step * dt, bond=bond,
                                       **diagnostics(actual, *references[step], model)))
            row["readout_seconds"] += perf_counter() - started
            if progress:
                sample = row["samples"][-1]
                print(json.dumps(dict(method=method, chi=chi, time=step * dt,
                                      plaquette_error=sample["max_plaquette_error"],
                                      trace_error=sample["trace_error"])), flush=True)
    row.update(audit_seconds=opt.audit_seconds, audited_gates=opt.audited_gates,
               max_per_gate_preservation_error=opt.max_preservation_error,
               storage_backends=sorted({ar.infer_backend(t.data) for t in opt.p}),
               storage_dtypes=sorted({ar.get_dtype_name(t.data) for t in opt.p}))
    return row


def plot_results(result, destination):
    """Render saved measurements without rerunning the simulation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    chis = sorted({r["chi"] for r in result["rows"]})
    colors = {chi: plt.get_cmap("tab10")(i) for i, chi in enumerate(chis)}
    keys = ["max_plaquette_error", "hs_relative_trotter", "energy_error_per_site",
            "trace_error", "min_eigenvalue_hermitian_part", "relative_hermiticity_defect"]
    labels = ["Plaquette error vs continuous exact", "Relative HS error vs exact circuit",
              "Energy error per site", "Raw trace error", "Minimum eigenvalue of Hermitian part",
              "Relative Hermiticity defect"]
    for row in result["rows"]:
        if row["status"] != "completed":
            continue
        style = {"dmt": "-", "optimized": ":", "fit": "--"}[row["method"]]
        times = [s["time"] for s in row["samples"]]
        for axis, key in zip(axes.flat, keys):
            data = [s[key] for s in row["samples"]]
            if key != "min_eigenvalue_hermitian_part":
                data = np.maximum(data, 1e-16)
            axis.plot(times, data, style, color=colors[row["chi"]],
                      label=f"{row['method'].upper()} χ={row['chi']}")
    reference = next(r for r in result["rows"] if r["status"] == "completed")
    times = [s["time"] for s in reference["samples"]]
    axes[0, 0].plot(times, np.maximum([s["trotter_max_plaquette_error"] for s in reference["samples"]], 1e-16),
                    "k-.", label="Exact circuit timestep error")
    for axis, key, label in zip(axes.flat, keys, labels):
        axis.set(xlabel="t (J=1)", title=label)
        axis.grid(alpha=.2)
        if key == "min_eigenvalue_hermitian_part":
            axis.axhline(0, color="black", linewidth=.7)
        else:
            axis.set_yscale("log")
            if key in {"max_plaquette_error", "energy_error_per_site"}:
                # Roundoff at t=0 should not flatten all later physical errors
                # against the top of a fourteen-decade plotting range.
                values = [s[key] for row in result["rows"] if row["status"] == "completed"
                          for s in row["samples"] if s["time"] > 0 and s[key] > 1e-10]
                if key == "max_plaquette_error":
                    values += [s["trotter_max_plaquette_error"] for s in reference["samples"]
                               if s["time"] > 0 and s["trotter_max_plaquette_error"] > 1e-10]
                if values:
                    axis.set_ylim(10**np.floor(np.log10(min(values))), max(values) * 2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    config = result["config"]
    shape = "×".join(map(str, config["shape"]))
    initial = config.get("initial_kind", "depolarized-pure")
    figure.suptitle(f"2D tilted Ising • {shape} • {initial} • Δt={config['dt']} • complex128")
    destination = Path(destination)
    figure.savefig(destination, dpi=180)
    figure.savefig(destination.with_suffix(".svg"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=2, type=int, default=[3, 2])
    parser.add_argument("--chis", nargs="+", type=int, default=[32, 48, 64])
    parser.add_argument("--methods", nargs="+", choices=["dmt", "optimized", "fit"], default=["dmt", "fit"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--fit-sweeps", type=int, default=4)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--map-mode", default="snake-row-major")
    parser.add_argument("--initial-kind", choices=["depolarized-pure", "product-mixed"], default="depolarized-pure")
    parser.add_argument("--polarization", type=float, default=.4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dmt-2d.json"))
    parser.add_argument("--plot-only", type=Path, help="Plot an existing result JSON instead of evolving")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(json.loads(args.plot_only.read_text()), args.output.with_suffix(".png"))
        return
    if min(args.steps, args.sample_every, args.fit_sweeps, *args.chis) < 1:
        parser.error("steps, sample-every, fit-sweeps and chis must be positive")
    if args.backend == "torch":
        import torch
        torch.set_num_threads(1)
    config = vars(args).copy()
    config["output"] = str(config["output"])
    output = dict(config=config, dtype="complex128", cpu_threads=1,
                  versions={p: version(p) for p in ["numpy", "scipy", "quimb", "autoray", "torch"]}, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = problem(*args.shape, args.dt, map_mode=args.map_mode,
                        initial_kind=args.initial_kind, polarization=args.polarization)
        references = exact_references(model, args.dt, args.steps, args.sample_every)
        output.update(edges=model["edges"], regions=model["regions"],
                      hamiltonian="sum_<ij> Z_i Z_j + 1.05 sum_i X_i + 0.5 sum_i Z_i",
                      pure_weight=model["pure_weight"])
        for method in args.methods:
            for chi in args.chis:
                row = trajectory(model, references, method=method, chi=chi, steps=args.steps, dt=args.dt,
                                 backend=args.backend, fit_sweeps=args.fit_sweeps, progress=True)
                output["rows"].append(row)
                args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
