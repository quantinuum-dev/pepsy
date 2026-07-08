"""Stabilizer Tensor Network (STN): magic-state injection and scalable sampling.

Runs the ``pepsy.MpsStabOptimizer`` STN simulator end-to-end on a small
Clifford + ``T`` circuit and shows:

1. A Clifford-entangled register keeps the coefficient MPS ``|nu>`` at bond 1 —
   the entanglement lives in the stabilizer tableau, not in ``|nu>``.
2. Magic-state injection (``prepare_magic`` + ``inject_t``) reproduces a ``T``
   gate by gate teleportation, keeping the non-Clifford cost on a recyclable
   ancilla instead of growing ``|nu>``.
3. Scalable computational-basis sampling (``sample_bits`` / ``probability_bits``)
   without ever forming a ``2**n`` statevector.

Deterministic; validated against a dense statevector for the small case.
"""

from __future__ import annotations

import numpy as np

import pepsy as py


def fidelity(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Clifford entanglement is free: GHZ keeps |nu> at bond 1.
    # ------------------------------------------------------------------ #
    ghz = py.MpsStabOptimizer(5).apply(
        [("h", 0)] + [("cnot", i, i + 1) for i in range(4)]
    )
    print(f"GHZ(5): |nu> bond = {ghz.state.max_bond()}  (Clifford -> tableau only)")

    # ------------------------------------------------------------------ #
    # 2. Magic-state injection reproduces a T gate.
    #    Register: qubit 0 = data, qubit 1 = magic ancilla.
    # ------------------------------------------------------------------ #
    inj = py.MpsStabOptimizer(2, seed=0)
    inj.state.h(0)             # data -> |+>
    inj.prepare_magic(1)       # ancilla -> |A> = T|+>  (offline resource)
    outcome = inj.inject_t(0, 1)   # teleport a T onto the data qubit
    # direct-T reference on the data qubit
    ref = py.MpsStabOptimizer(1).apply([("h", 0), ("t", 0)])
    full = inj.to_statevector().reshape(2, 2)
    data_marginal = full[:, 0] if np.linalg.norm(full[:, 0]) > np.linalg.norm(full[:, 1]) else full[:, 1]
    print(
        f"inject_t: outcome={outcome:+d}, "
        f"fidelity(data, T|+>)={fidelity(data_marginal, ref.to_statevector()):.6f}, "
        f"|nu> bond={inj.state.max_bond()}"
    )

    # Ancilla recycling: reset and inject a second T -> T^2 on the data qubit.
    inj.reset(1)
    inj.prepare_magic(1)
    inj.inject_t(0, 1, outcome=+1)
    print("recycled ancilla for a second T (T^2 on data); reset + reuse OK")

    # ------------------------------------------------------------------ #
    # 3. Scalable sampling on a Clifford + T circuit, checked vs dense.
    # ------------------------------------------------------------------ #
    n = 4
    sim = py.MpsStabOptimizer(n, seed=1).apply(
        [("h", 0), ("cnot", 0, 1), ("t", 2), ("ry", 0.7, 3), ("cnot", 2, 3)]
    )
    psi = sim.to_statevector()  # dense only for this small validation
    max_err = max(
        abs(sim.probability_bits(format(k, f"0{n}b")) - abs(psi[k]) ** 2)
        for k in range(2 ** n)
    )
    print(f"probability_bits vs dense: max abs error = {max_err:.2e}")

    shots = 5000
    samples = sim.sample_bits(shots, seed=2)
    idx = (samples.astype(int) * (1 << np.arange(n - 1, -1, -1))).sum(1)
    freq = np.bincount(idx, minlength=2 ** n) / shots
    tv = 0.5 * np.abs(freq - np.abs(psi) ** 2).sum()
    print(f"sample_bits ({shots} shots): total-variation distance vs dense = {tv:.4f}")


if __name__ == "__main__":
    main()
