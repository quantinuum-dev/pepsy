"""Independent orientation and gate checks for the 4x5 TFIM experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy.optimizers import MpoOptimizer, PepoOptimizer
from pepsy.tensors import OneDMap


def test_tfim_example_single_gate_readouts_match_vector_reference():
    path = Path(__file__).parents[1] / "examples/pepo_dmt_ising_4x5.py"
    spec = importlib.util.spec_from_file_location("pepo_tfim_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    z = np.diag([1., -1.]).astype(complex)
    local = (np.eye(2) + x) / 2
    gate = expm(.05j * np.kron(z, z))
    pair = ((0, 0), (0, 1))
    region = ((0, 0), (0, 1), (1, 0))
    _, index = OneDMap(4, 5, mode="snake-row-major").build()
    psi = example.vector_gate(np.full(2 ** 20, 2 ** -10, dtype=complex), gate, pair)
    expected = example.vector_density(psi, region)
    pepo = PepoOptimizer(qtn.PEPO_product_operator([[local] * 5 for _ in range(4)]),
                         [(gate, pair)], chi=4, mode="dmt", regions=[region], boundary_chi=8).run()
    mpo = MpoOptimizer(qtn.MPO_product_operator([local] * 20),
                       [(gate.T, tuple(index[s] for s in pair))], chi=32, mode="dmt",
                       contraction_opt="greedy").run(
                           cutoff=0., dmt_regions=[tuple(sorted(index[s] for s in region))],
                           fidelity_samples=0)
    np.testing.assert_allclose(example.pepo_density(pepo, region), expected, atol=1e-11)
    np.testing.assert_allclose(example.mpo_density(mpo, region, index), expected, atol=1e-11)


def test_ising_observables_for_uniform_x_product_state():
    path = Path(__file__).parents[1] / "examples/pepo_dmt_ising_4x5.py"
    spec = importlib.util.spec_from_file_location("pepo_tfim_example_observables", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    x = np.array([[0., 1.], [1., 0.]], dtype=complex)
    local = (np.eye(2) + x) / 2
    state = qtn.PEPO_product_operator([[local] * 5 for _ in range(4)])
    edges = [((i, j), (i, j + 1)) for i in range(4) for j in range(4)]
    edges += [((i, j), (i + 1, j)) for i in range(3) for j in range(5)]
    obs = example.ising_observables(lambda region: example.pepo_density(state, region), edges, .7)
    assert abs(obs["energy_real"] + 14.) < 1e-12
    assert obs["max_abs_z"] < 1e-12
    assert obs["max_x_reflection_defect"] < 1e-12
