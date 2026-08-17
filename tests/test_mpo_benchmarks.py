"""Small and large-chain accuracy benchmarks for higher-order MPO baselines.

These are intentionally regression-sized rather than performance harnesses:
they compare the finite MPO construction with first-order Trotter and a
two-site cluster-expansion baseline on a four-site chain. The slow benchmark
also compares parameterized transverse-field Ising MPOs with Trotter MPOs at
``L=32`` without forming dense operators.
"""

from itertools import product

import numpy as np
import pytest

from pepsy.operators import (
    FirstDegreeMPO,
    MPOBasis,
    MPOParameter,
    MPOProductTerm,
)


def _paulis():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.diag([1.0, -1.0])
    return x, y, z


def _term_cases():
    """Return varied one- and two-site finite-chain term layouts."""
    x, y, z = _paulis()
    return {
        "nearest_neighbor": (
            ((0,), (z,)),
            ((0, 1), (x, x)),
            ((1, 2), (z, z)),
            ((2, 3), (x, z)),
        ),
        "long_range": (
            ((0,), (z,)),
            ((0, 3), (x, z)),
            ((1, 2), (y, x)),
        ),
        "shared_support": (
            ((0, 1), (x, x)),
            ((0, 1), (y, x)),
            ((1, 2), (z, z)),
            ((2, 3), (z, x)),
        ),
    }


def _local_matrix(operators):
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def _embedded_product(L, site, left, right):
    identity = np.eye(left.shape[0])
    factors = [identity for _ in range(L)]
    factors[site] = left
    factors[site + 1] = right
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _embedded_gate(gate, where, L):
    """Embed a one- or two-site gate at arbitrary sites for a dense oracle."""
    where = tuple(where)
    dimension = 2**L
    embedded = np.zeros((dimension, dimension), dtype=gate.dtype)
    for input_index in range(dimension):
        input_bits = [
            (input_index >> (L - 1 - site)) & 1
            for site in range(L)
        ]
        local_input = 0
        for site in where:
            local_input = (local_input << 1) | input_bits[site]
        for local_output in range(2 ** len(where)):
            output_bits = list(input_bits)
            for position, site in enumerate(where):
                output_bits[site] = (
                    local_output >> (len(where) - 1 - position)
                ) & 1
            output_index = 0
            for bit in output_bits:
                output_index = (output_index << 1) | bit
            embedded[output_index, input_index] = gate[local_output, local_input]
    return embedded


def _dense_hamiltonian(L, term_specs):
    identity = np.eye(2)
    hamiltonian = np.zeros((2**L, 2**L), dtype=complex)
    for sites, operators in term_specs:
        factors = [identity] * L
        for site, operator in zip(sites, operators):
            factors[site] = operator
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        hamiltonian += term
    return hamiltonian


def _trotter_reference(L, term_specs, dt):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    result = np.eye(2**L, dtype=complex)
    gates = []
    for sites, operators in term_specs:
        gate = scipy_linalg.expm(-1j * dt * _local_matrix(operators))
        result = _embedded_gate(gate, sites, L) @ result
        gates.append((gate, tuple(sites)))
    return result, gates


def _baseline_operators(dt):
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x, _, z = _paulis()
    L = 4
    local_terms = (
        (0, x, x),
        (1, z, z),
        (2, x, z),
    )
    local_operators = tuple(
        _embedded_product(L, site, left, right)
        for site, left, right in local_terms
    )
    hamiltonian = sum(local_operators)
    exact = scipy_linalg.expm(dt * hamiltonian)

    # First-order Lie-Trotter: product of exact exponentials of the local
    # two-site terms in a fixed left-to-right ordering.
    trotter = np.eye(2**L)
    for local_operator in local_operators:
        trotter = scipy_linalg.expm(dt * local_operator) @ trotter

    # The p=2 cluster baseline keeps each connected two-site exponential
    # correction and all products whose supports are disjoint. Overlapping
    # connected clusters are omitted, as in a finite truncation of the
    # cluster expansion.
    corrections = tuple(
        scipy_linalg.expm(dt * local_operator) - np.eye(2**L)
        for local_operator in local_operators
    )
    cluster = np.zeros_like(exact)
    for selected_mask in product((False, True), repeat=len(corrections)):
        selected = tuple(
            index for index, chosen in enumerate(selected_mask) if chosen
        )
        if any(right - left == 1 for left, right in zip(selected, selected[1:])):
            continue
        term = np.eye(2**L)
        for index in selected:
            term = corrections[index] @ term
        cluster += term

    return L, local_terms, exact, trotter, cluster


def test_higher_order_mpo_accuracy_benchmark_against_trotter_and_cluster():
    """Report deterministic errors against both finite-chain baselines."""
    dt = 0.08
    L, local_terms, exact, trotter, cluster = _baseline_operators(dt)
    x, _, z = _paulis()
    H = FirstDegreeMPO.from_local_terms(
        L,
        [
            MPOProductTerm((site, site + 1), (left, right))
            for site, left, right in local_terms
        ],
    )

    errors = {}
    for order in (1, 2, 3):
        U = H.extensive_exponential(
            dt,
            order=order,
            cache_history=False,
        )
        errors[f"mpo_order_{order}"] = np.linalg.norm(
            U.to_mpo().to_dense() - exact,
        )
    errors["trotter_first_order"] = np.linalg.norm(trotter - exact)
    errors["cluster_two_site"] = np.linalg.norm(cluster - exact)

    # The MPO's Taylor order controls the expected convergence, while the
    # baselines are retained as independent reference methods rather than
    # being asserted to have the same error ordering.
    assert errors["mpo_order_3"] < errors["mpo_order_2"] < errors["mpo_order_1"]
    assert all(np.isfinite(value) for value in errors.values())
    assert errors["trotter_first_order"] > 0.0
    assert errors["cluster_two_site"] > 0.0
    assert x.shape == z.shape == (2, 2)


@pytest.mark.parametrize("case_name", tuple(_term_cases()))
def test_higher_order_mpo_converges_for_varied_term_layouts(case_name):
    """Orders one through three converge for local, long-range, and shared terms."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    L = 4
    term_specs = _term_cases()[case_name]
    hamiltonian = _dense_hamiltonian(L, term_specs)
    exact = scipy_linalg.expm(-1j * 0.04 * hamiltonian)
    H = FirstDegreeMPO.from_local_terms(
        L,
        [MPOProductTerm(sites, operators) for sites, operators in term_specs],
    )

    errors = []
    for order in (1, 2, 3):
        U = H.extensive_exponential(
            -1j * 0.04,
            order=order,
            mode="optimal",
            cache_history=False,
        )
        errors.append(np.linalg.norm(U.to_mpo().to_dense() - exact))

    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1.0e-4
    assert all(np.isfinite(error) for error in errors)


@pytest.mark.parametrize("case_name", tuple(_term_cases()))
@pytest.mark.parametrize("mode", ("svd", "mpo"))
def test_mpo_optimizer_replays_trotter_for_varied_term_layouts(case_name, mode):
    """MpoOptimizer reproduces the dense first-order Trotter product."""
    qtn = pytest.importorskip("quimb.tensor")
    from pepsy import MpoOptimizer  # pylint: disable=import-outside-toplevel

    L = 4
    term_specs = _term_cases()[case_name]
    trotter, gates = _trotter_reference(L, term_specs, 0.017)
    # MpoOptimizer's public gate convention stores Quimb gates as
    # (output, input), and the optimizer transposes them before application.
    optimizer_gates = [((gate.T, None), where) for gate, where in gates]
    optimizer = MpoOptimizer(
        qtn.MPO_identity(L, dtype="complex128"),
        gates=optimizer_gates,
        chi=32,
        mode=mode,
    )
    output = optimizer.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
    )

    np.testing.assert_allclose(
        output.to_dense(),
        trotter,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_mpo_optimizer_dmrg_replays_trotter_without_variational_drift():
    """The FIT/DMRG replay also reaches the exact small-system Trotter MPO."""
    qtn = pytest.importorskip("quimb.tensor")
    from pepsy import MpoOptimizer  # pylint: disable=import-outside-toplevel

    L = 4
    trotter, gates = _trotter_reference(
        L,
        _term_cases()["nearest_neighbor"],
        0.017,
    )
    optimizer = MpoOptimizer(
        qtn.MPO_identity(L, dtype="complex128"),
        gates=[((gate.T, None), where) for gate, where in gates],
        chi=16,
        mode="dmrg",
    )
    output = optimizer.run(
        n_iter=2,
        fit_block_size=2,
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
    )

    np.testing.assert_allclose(
        output.to_dense(),
        trotter,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


@pytest.mark.parametrize("case_name", tuple(_term_cases()))
def test_first_order_trotter_has_quadratic_step_error(case_name):
    """The independent Trotter baseline exhibits its expected dt scaling."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    L = 4
    term_specs = _term_cases()[case_name]
    exact_hamiltonian = _dense_hamiltonian(L, term_specs)
    errors = []
    for dt in (0.04, 0.02):
        exact = scipy_linalg.expm(-1j * dt * exact_hamiltonian)
        trotter, _ = _trotter_reference(L, term_specs, dt)
        errors.append(np.linalg.norm(trotter - exact))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.08)


def _parameterized_basis(term_specs):
    return MPOBasis.from_local_terms(
        4,
        [
            MPOProductTerm(
                sites,
                operators,
                coefficient=MPOParameter(f"c{index}"),
            )
            for index, (sites, operators) in enumerate(term_specs)
        ],
    )


def _numpy_basis_objective(basis, values, time):
    U = basis.evolution_mpo(
        {f"c{index}": value for index, value in enumerate(values)},
        dt=-1j * time,
        order=2,
        mode="optimal",
        cache_history=False,
    )
    return sum(np.real(np.asarray(array)).sum() for array in U.arrays)


def test_parameterized_mpo_torch_gradients_match_finite_differences():
    """Torch reverse-mode gradients agree with a full construction FD oracle."""
    torch = pytest.importorskip("torch")
    term_specs = _term_cases()["long_range"]
    basis = _parameterized_basis(term_specs)

    def objective(*inputs):
        values, time = inputs[:-1], inputs[-1]
        U = basis.evolution_mpo(
            {f"c{index}": value for index, value in enumerate(values)},
            dt=-1j * time,
            order=2,
            mode="optimal",
            cache_history=False,
        )
        return sum(array.real.sum() for array in U.arrays)

    values = [
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in (0.7, -0.2, 0.35)
    ]
    time = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    gradients = torch.autograd.grad(objective(*values, time), (*values, time))

    finite_differences = []
    point = [0.7, -0.2, 0.35, 0.02]
    for index in range(len(point)):
        step = 1.0e-5
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        finite_differences.append(
            (
                _numpy_basis_objective(basis, plus[:-1], plus[-1])
                - _numpy_basis_objective(basis, minus[:-1], minus[-1])
            )
            / (2.0 * step)
        )

    np.testing.assert_allclose(
        [gradient.detach().numpy() for gradient in gradients],
        finite_differences,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


@pytest.mark.slow
def test_parameterized_mpo_jax_gradients_match_finite_differences():
    """JAX reverse-mode gradients agree with the same FD oracle."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    term_specs = _term_cases()["long_range"][:2]
    basis = _parameterized_basis(term_specs)

    def objective(*inputs):
        values, time = inputs[:-1], inputs[-1]
        U = basis.evolution_mpo(
            {f"c{index}": value for index, value in enumerate(values)},
            dt=-1j * time,
            order=2,
            mode="optimal",
            cache_history=False,
        )
        return sum(jnp.real(array).sum() for array in U.arrays)

    point = (0.7, -0.2, 0.02)
    value, gradients = jax.value_and_grad(
        objective,
        argnums=(0, 1, 2),
    )(*point)
    assert jnp.isfinite(value)

    finite_differences = []
    for index in range(len(point)):
        step = 1.0e-4
        plus = list(point)
        minus = list(point)
        plus[index] += step
        minus[index] -= step
        finite_differences.append(
            (
                _numpy_basis_objective(basis, plus[:-1], plus[-1])
                - _numpy_basis_objective(basis, minus[:-1], minus[-1])
            )
            / (2.0 * step)
        )

    np.testing.assert_allclose(
        np.asarray(gradients),
        finite_differences,
        rtol=3.0e-3,
        atol=3.0e-4,
    )


def _tfi_basis(L):
    """Build one cached basis for ``J ZZ + hx X + hz Z`` on an open chain."""
    x, _, z = _paulis()
    terms = [
        MPOProductTerm((site, site + 1), (z, z), MPOParameter("J"))
        for site in range(L - 1)
    ]
    terms.extend(
        MPOProductTerm((site,), (x,), MPOParameter("hx"))
        for site in range(L)
    )
    terms.extend(
        MPOProductTerm((site,), (z,), MPOParameter("hz"))
        for site in range(L)
    )
    return MPOBasis.from_local_terms(L, terms)


def _tfi_gate_stream(L, parameters, dt):
    """Return a fixed-order local gate stream for one TFI Trotter step."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x, _, z = _paulis()
    gates = [
        (
            scipy_linalg.expm(
                -1j * dt * parameters["J"] * np.kron(z, z),
            ),
            (site, site + 1),
        )
        for site in range(L - 1)
    ]
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["hx"] * x),
            (site,),
        )
        for site in range(L)
    )
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["hz"] * z),
            (site,),
        )
        for site in range(L)
    )
    return gates


def _run_tfi_trotter(L, parameters, dt, order, *, n_substeps=1):
    """Replay Lie or Strang Trotter steps as an MPO without densifying."""
    qtn = pytest.importorskip("quimb.tensor")
    from pepsy import MpoOptimizer  # pylint: disable=import-outside-toplevel

    stream = []
    for _ in range(n_substeps):
        if order == 1:
            stream.extend(
                _tfi_gate_stream(L, parameters, dt / n_substeps),
            )
            continue
        if order != 2:
            raise ValueError("the TFI regression supports Trotter orders 1 and 2")
        half_step = _tfi_gate_stream(
            L,
            parameters,
            dt / (2 * n_substeps),
        )
        stream.extend(half_step)
        stream.extend(reversed(half_step))

    # MpoOptimizer accepts Quimb's gate convention (output, input) and
    # transposes it internally before applying a ket-only gate.
    optimizer = MpoOptimizer(
        qtn.MPO_identity(L, dtype="complex128"),
        gates=[((gate.T, None), where) for gate, where in stream],
        chi=32,
        mode="mpo",
    )
    return optimizer.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
    )


def _relative_mpo_error(left, right):
    """Return a normalized MPO norm difference, retaining tensor-network form."""
    return float(abs((left - right).norm() / right.norm()))


@pytest.mark.slow
def test_parameterized_tfi_large_chain_mpo_against_higher_order_trotter():
    """Compare cached TFI MPO orders with Lie and Strang Trotter at ``L=32``.

    The two parameter sets exercise rebinding of the same compiled ``J``,
    ``hx``, and ``hz`` topology. The four-substep Strang result is a more
    accurate MPO reference, so no dense ``2**L`` operator is required.
    """
    L = 32
    dt = 0.02
    parameter_sets = (
        {"J": 1.0, "hx": 0.7, "hz": 0.2},
        {"J": 0.35, "hx": 1.1, "hz": -0.25},
    )
    basis = _tfi_basis(L)

    for parameters in parameter_sets:
        hamiltonian = basis.build(parameters)
        mpo_order_2 = hamiltonian.extensive_exponential(
            -1j * dt,
            order=2,
            mode="optimal",
            cache_history=False,
            history_storage="streaming",
        ).to_mpo()
        mpo_order_3 = hamiltonian.extensive_exponential(
            -1j * dt,
            order=3,
            mode="optimal",
            cache_history=False,
            history_storage="streaming",
        ).to_mpo()

        trotter_order_1 = _run_tfi_trotter(
            L,
            parameters,
            dt,
            order=1,
        )
        trotter_order_2 = _run_tfi_trotter(
            L,
            parameters,
            dt,
            order=2,
        )
        reference = _run_tfi_trotter(
            L,
            parameters,
            dt,
            order=2,
            n_substeps=4,
        )

        errors = {
            "mpo_order_2": _relative_mpo_error(mpo_order_2, reference),
            "mpo_order_3": _relative_mpo_error(mpo_order_3, reference),
            "trotter_order_1": _relative_mpo_error(trotter_order_1, reference),
            "trotter_order_2": _relative_mpo_error(trotter_order_2, reference),
        }
        assert errors["mpo_order_3"] < errors["mpo_order_2"]
        assert errors["trotter_order_2"] < errors["trotter_order_1"]
        assert errors["mpo_order_3"] < errors["trotter_order_2"]
        assert all(np.isfinite(error) for error in errors.values())

    assert basis.cache_info["compiled"] is True
    assert basis.cache_info["compiled_terms"] == 3 * L - 1
    assert basis.cache_info["builds"] == len(parameter_sets)


def _tfi_xyz_basis(L):
    """Build a TFI basis with a Y field and long-range ``X Y Z`` terms."""
    x, y, z = _paulis()
    terms = [
        MPOProductTerm((site, site + 1), (z, z), MPOParameter("J"))
        for site in range(L - 1)
    ]
    terms.extend(
        MPOProductTerm((site,), (x,), MPOParameter("hx"))
        for site in range(L)
    )
    terms.extend(
        MPOProductTerm((site,), (y,), MPOParameter("hy"))
        for site in range(L)
    )
    terms.extend(
        MPOProductTerm((site,), (z,), MPOParameter("hz"))
        for site in range(L)
    )
    terms.extend(
        MPOProductTerm(
            (site, site + 1, site + 3),
            (x, y, z),
            MPOParameter("gxyz"),
        )
        for site in range(L - 3)
    )
    return MPOBasis.from_local_terms(L, terms)


def _tfi_xyz_gate_stream(L, parameters, dt):
    """Return Trotter gates for the extended TFI/long-range XYZ model."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    x, y, z = _paulis()
    gates = [
        (
            scipy_linalg.expm(
                -1j * dt * parameters["J"] * np.kron(z, z),
            ),
            (site, site + 1),
        )
        for site in range(L - 1)
    ]
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["hx"] * x),
            (site,),
        )
        for site in range(L)
    )
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["hy"] * y),
            (site,),
        )
        for site in range(L)
    )
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["hz"] * z),
            (site,),
        )
        for site in range(L)
    )
    xyz = np.kron(np.kron(x, y), z)
    gates.extend(
        (
            scipy_linalg.expm(-1j * dt * parameters["gxyz"] * xyz),
            (site, site + 1, site + 3),
        )
        for site in range(L - 3)
    )
    return gates


def _run_tfi_xyz_trotter(L, parameters, dt, order, *, n_substeps=1):
    """Replay the extended TFI Trotter stream through ``MpoOptimizer``."""
    qtn = pytest.importorskip("quimb.tensor")
    from pepsy import MpoOptimizer  # pylint: disable=import-outside-toplevel

    stream = []
    for _ in range(n_substeps):
        if order == 1:
            stream.extend(
                _tfi_xyz_gate_stream(L, parameters, dt / n_substeps),
            )
            continue
        if order != 2:
            raise ValueError("the XYZ regression supports Trotter orders 1 and 2")
        half_step = _tfi_xyz_gate_stream(
            L,
            parameters,
            dt / (2 * n_substeps),
        )
        stream.extend(half_step)
        stream.extend(reversed(half_step))

    # The direct MPO backend supports the non-contiguous three-site gates.
    # Ket-only input uses the public (output, input) gate convention.
    optimizer = MpoOptimizer(
        qtn.MPO_identity(L, dtype="complex128"),
        gates=[((gate.T, None), where) for gate, where in stream],
        chi=8,
        mode="mpo",
    )
    return optimizer.run(
        progbar=False,
        cutoff=0.0,
        fidelity_samples=0,
    )


@pytest.mark.slow
def test_parameterized_tfi_xyz_large_chain_algorithms_one_two_three():
    """Compare MPO Algorithms 1/2/3 with Trotter for ``X_i Y_{i+1} Z_{i+3}``."""
    L = 32
    dt = 0.01
    parameters = {
        "J": 1.0,
        "hx": 0.7,
        "hy": -0.23,
        "hz": 0.2,
        "gxyz": 0.15,
    }
    rebinding = {
        "J": 0.35,
        "hx": 1.1,
        "hy": 0.4,
        "hz": -0.25,
        "gxyz": -0.3,
    }
    basis = _tfi_xyz_basis(L)
    hamiltonian = basis.build(parameters)
    # Compile and bind a second coupling point as well, verifying that the
    # long-range XYZ coefficient shares the same cached topology.
    rebound_hamiltonian = basis.build(rebinding)

    mpo = {
        order: hamiltonian.extensive_exponential(
            -1j * dt,
            order=order,
            mode="optimal",
            cache_history=False,
            history_storage="streaming",
        ).to_mpo()
        for order in (1, 2, 3)
    }
    trotter_order_1 = _run_tfi_xyz_trotter(
        L,
        parameters,
        dt,
        order=1,
    )
    trotter_order_2 = _run_tfi_xyz_trotter(
        L,
        parameters,
        dt,
        order=2,
    )
    reference = _run_tfi_xyz_trotter(
        L,
        parameters,
        dt,
        order=2,
        n_substeps=2,
    )

    errors = {
        "mpo_algorithm_1": _relative_mpo_error(mpo[1], reference),
        "mpo_algorithm_2": _relative_mpo_error(mpo[2], reference),
        "mpo_algorithm_3": _relative_mpo_error(mpo[3], reference),
        "trotter_order_1": _relative_mpo_error(trotter_order_1, reference),
        "trotter_order_2": _relative_mpo_error(trotter_order_2, reference),
    }
    assert errors["mpo_algorithm_3"] < errors["mpo_algorithm_2"]
    assert errors["mpo_algorithm_2"] < errors["mpo_algorithm_1"]
    assert errors["trotter_order_2"] < errors["trotter_order_1"]
    assert errors["mpo_algorithm_3"] < errors["trotter_order_2"]
    assert all(np.isfinite(error) for error in errors.values())
    assert basis.cache_info["compiled_terms"] == 5 * L - 4
    assert basis.cache_info["builds"] == 2
    assert rebound_hamiltonian.bond_dimensions == basis.bond_dimensions
