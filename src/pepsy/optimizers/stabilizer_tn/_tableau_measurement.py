"""Exact, branch-local eligibility for native tableau measurement."""

import autoray as ar
import numpy as np

from ._backend import array_namespace


def _product_states(optimizer, sites):
    """Certify separated Pauli eigenstates using live, exact array predicates."""
    p = optimizer.p
    if getattr(p, "cyclic", False):
        return None, "cyclic_coefficients"
    if any(getattr(t.data, "requires_grad", False) for t in p.tensors):
        return None, "trainable_coefficients"
    if any(hasattr(t.data, "blocks") for t in p.tensors):
        return None, "block_coefficients"
    physical_sites = tuple(optimizer._mps_site(logical) for logical in sites)
    if any(p.bond_size(site, neighbor) != 1
           for site in physical_sites for neighbor in (site - 1, site + 1)
           if 0 <= neighbor < p.L):
        return None, "uncertified_entanglement"
    states = [None] * optimizer.n
    if not sites:
        return tuple(states), None
    arrays = [p[site].data for site in physical_sites]
    try:
        xp = array_namespace(arrays[0])
        vectors = xp.stack([xp.reshape(data, (2,)) for data in arrays])
        a, b = vectors[:, 0], vectors[:, 1]
        valid = xp.logical_and(xp.isfinite(a), xp.isfinite(b))
        cases = xp.stack((
            xp.logical_and(a != 0, b == 0),
            xp.logical_and(a == 0, b != 0),
            xp.logical_and(a != 0, b == a),
            xp.logical_and(a != 0, b == -a),
            xp.logical_and(a != 0, b == 1j * a),
            xp.logical_and(a != 0, b == -1j * a),
        ), axis=1)
        flags = np.asarray(ar.to_numpy(xp.logical_and(cases, valid[:, None])))
        labels = ("Z+", "Z-", "X+", "X-", "Y+", "Y-")
        for row, logical in enumerate(sites):
            selected = np.flatnonzero(flags[row])
            if len(selected) != 1:
                return None, "uncertified_local_state"
            states[logical] = labels[int(selected[0])]
        return tuple(states), None
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None, "unsupported_array"


def computational_coefficient_bits(optimizer):
    """Compatibility certificate for exactly computational product |p>."""
    states, _reason = _product_states(optimizer, tuple(range(optimizer.n)))
    if states is None or any(state[0] != "Z" for state in states):
        return None
    return tuple(state == "Z-" for state in states)


def _preparation(states):
    """Local Clifford U with U|0> equal to the certified coefficient product."""
    import stim

    frame = stim.Tableau(len(states))
    for site, state in enumerate(states):
        if state is None:
            continue
        if state[1] == "-":
            frame.append(stim.Tableau.from_named_gate("X"), [site])
        if state[0] in "XY":
            frame.append(stim.Tableau.from_named_gate("H"), [site])
        if state[0] == "Y":
            frame.append(stim.Tableau.from_named_gate("S"), [site])
    return frame


def _simulator(tableau):
    import stim

    sim = stim.TableauSimulator()
    sim.set_num_qubits(len(tableau))
    sim.do_tableau(tableau, range(len(tableau)))
    return sim


def certified_tableau(optimizer):
    """Materialize C|Pauli eigenstates> without changing the live basis."""
    states, _reason = _product_states(optimizer, tuple(range(optimizer.n)))
    if states is None:
        return None
    if all(state == "Z+" for state in states):
        return optimizer.state._sim, states
    physical = _preparation(states).then(optimizer.state._sim.current_inverse_tableau().inverse())
    return _simulator(physical), states


def _certificate_with_reason(optimizer, observable):
    """Certify a full state or separated region; explain conservative fallback."""
    whole = certified_tableau(optimizer)
    if whole is not None:
        sim, states = whole
        return (sim, states, observable, None), None
    frame = optimizer.state.frame_pauli(observable)
    support = tuple(i for i in range(optimizer.n) if frame[i])
    states, reason = _product_states(optimizer, support)
    if states is None:
        return None, reason
    return (_simulator(_preparation(states)), states, frame, support), None


def certificate(optimizer, observable):
    return _certificate_with_reason(optimizer, observable)[0]


def fallback_reason(optimizer, observable):
    return _certificate_with_reason(optimizer, observable)[1]


def probabilities(optimizer, observable):
    """Return tableau Born weights, or None without a live certificate."""
    certified = certificate(optimizer, observable)
    if certified is None:
        return None
    sim, _states, measured, _region = certified
    expectation = sim.peek_observable_expectation(measured)
    return (1.0 + expectation) / 2.0, (1.0 - expectation) / 2.0


def measure(optimizer, observable, outcome, *, kind):
    """Collapse through Stim, preserving Pepsy RNG, scale and norm records."""
    certified = certificate(optimizer, observable)
    if certified is None:
        return None
    sim, states, measured, region = certified
    expectation = sim.peek_observable_expectation(measured)
    weights = (1.0 + expectation) / 2.0, (1.0 - expectation) / 2.0
    forced = optimizer._validate_outcome(outcome)
    if forced is None:
        result = 1 if optimizer._rng.random() < weights[0] else -1
    else:
        result = forced
    probability = weights[0 if result > 0 else 1]
    if probability == 0.0:
        raise ValueError(f"forced outcome {result:+d} has 0 probability (native tableau).")
    if observable.weight == 0:
        return result
    updated = sim.copy()
    updated.postselect_observable(measured, desired_value=result < 0)
    if region is not None:
        import stim

        localizer = updated.current_inverse_tableau().inverse()
        # Prove the right-basis update leaves the entire magic complement alone.
        identity = stim.Tableau(optimizer.n)
        for site in range(optimizer.n):
            if site not in region and (
                localizer.x_output(site) != identity.x_output(site)
                or localizer.z_output(site) != identity.z_output(site)
            ):
                raise RuntimeError("Native region collapse touched uncertified coefficient sites.")
        physical = localizer.then(optimizer.state._sim.current_inverse_tableau().inverse())
        updated = stim.TableauSimulator()
        updated.set_num_qubits(optimizer.n)
        updated.do_tableau(physical, range(optimizer.n))
    center = optimizer._canonize_p_single()
    event = optimizer._make_norm_event(kind, branch_probability=probability)
    represented_norm = optimizer._renorm_p_at(center)
    # Absorb local eigenstate preparations into the basis, keeping scalar phase.
    # Exact reconstruction avoids cancellation residues from applying U†.
    for logical, state in enumerate(states):
        if state is not None and state != "Z+":
            site = optimizer._mps_site(logical)
            tensor = optimizer.p[site]
            xp = array_namespace(tensor.data)
            vector = xp.reshape(tensor.data, (2,))
            scalar = vector[1] if state == "Z-" else vector[0] * 2**.5
            tensor.modify(data=xp.reshape(xp.stack((scalar, xp.zeros_like(scalar))), tensor.shape))
    optimizer.state._sim = updated
    optimizer.state._inv_tableau = None
    optimizer.state._clifford_unitary_cache = None
    optimizer.state._identity_cache = None
    optimizer._reset_infidelity()
    if event is not None:
        event.measurement_backend = "stim" if region is None else "stim_region"
    optimizer._commit_norm_event(event, projected_norm=represented_norm * probability**.5)
    optimizer._record()
    return result
