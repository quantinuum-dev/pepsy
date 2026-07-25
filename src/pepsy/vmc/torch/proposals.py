"""Symmetry-preserving proposal kernels and proposal diagnostics."""

from __future__ import annotations

import math

from ..torch_types import FermionSiteEncoding, _check_positive_int, _require_torch
from ._common import _as_long_matrix, _iter_edges, _run_cheap_torch_kernel
from .amplitude import (
    _call_amplitude_fn,
    _call_log_amplitude_fn,
    _resolve_log_amplitude_fn,
)
from .results import TorchMetropolisResult, _make_progress

__all__ = [
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "propose_spinful_u1_exchange_or_hopping",
    "propose_spinful_z2_exchange_or_hopping",
    "propose_spinful_z2z2_exchange_or_hopping",
    "_accumulate_accepted_proposal_stats",
    "_accumulate_proposal_stats",
    "_adapt_proposal_mix_rate",
    "_empty_proposal_stats",
    "_merge_proposal_stats",
    "_proposal_move_score",
    "_proposal_mix_rates",
    "_safe_metropolis_log_ratio",
    "_safe_metropolis_ratio",
    "_warmup_proposal_mix",
]


def _proposal_sites(i, j, configs):
    """Build fixed-shape edge indices for proposal kernels."""
    torch = _require_torch()
    sites = torch.as_tensor((i, j), dtype=torch.long, device=configs.device)
    return sites.reshape(1, 2).expand(configs.shape[0], -1)


def _spin_exchange_kernel(configs, sites):
    torch = _require_torch()
    endpoints = torch.gather(configs, 1, sites)
    changed = endpoints[:, 0] != endpoints[:, 1]
    values = torch.stack(
        (
            torch.where(changed, endpoints[:, 1], endpoints[:, 0]),
            torch.where(changed, endpoints[:, 0], endpoints[:, 1]),
        ),
        dim=1,
    )
    return configs.scatter(1, sites, values), changed


_PROPOSAL_MOVE_NAMES = (
    "exchange",
    "hopping",
    "spin_flip",
    "pair_toggle",
)
_MOVE_EXCHANGE, _MOVE_HOPPING, _MOVE_SPIN_FLIP, _MOVE_PAIR_TOGGLE = range(4)


def _spinful_exchange_hopping_kernel(
    configs,
    sites,
    hopping_rate,
    encoding_codes,
    branch_random,
    d0_random,
    d2_random,
):
    """Branch-free U1U1 proposal core suitable for ``torch.compile``."""
    torch = _require_torch()
    endpoints = torch.gather(configs, 1, sites)
    ci, cj = endpoints[:, 0], endpoints[:, 1]
    empty, double, up, down = encoding_codes.unbind()
    changed = ci != cj

    n_up_i = ((ci == up) | (ci == double)).to(torch.long)
    n_up_j = ((cj == up) | (cj == double)).to(torch.long)
    n_down_i = ((ci == down) | (ci == double)).to(torch.long)
    n_down_j = ((cj == down) | (cj == double)).to(torch.long)
    delta_n = ((n_up_i + n_down_i) - (n_up_j + n_down_j)).abs()

    exchange = (branch_random < (1.0 - hopping_rate)) & changed
    hopping = (~exchange) & changed
    swap = exchange | (hopping & (delta_n == 1))
    next_i = torch.where(swap, cj, ci)
    next_j = torch.where(swap, ci, cj)

    d0 = hopping & (delta_n == 0)
    d0_i = torch.where(d0_random, double, empty)
    d0_j = torch.where(d0_random, empty, double)
    next_i = torch.where(d0, d0_i, next_i)
    next_j = torch.where(d0, d0_j, next_j)

    d2 = hopping & (delta_n == 2)
    d2_i = torch.where(d2_random, down, up)
    d2_j = torch.where(d2_random, up, down)
    next_i = torch.where(d2, d2_i, next_i)
    next_j = torch.where(d2, d2_j, next_j)
    move_codes = torch.where(
        branch_random < (1.0 - hopping_rate),
        torch.full_like(branch_random, _MOVE_EXCHANGE, dtype=torch.long),
        torch.full_like(branch_random, _MOVE_HOPPING, dtype=torch.long),
    )
    return (
        configs.scatter(1, sites, torch.stack((next_i, next_j), dim=1)),
        changed,
        move_codes,
    )


def propose_spin_exchange(
    i,
    j,
    configs,
    *,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose spin exchange on one edge for binary spin configs."""
    configs = _as_long_matrix(configs)
    proposed, changed = _run_cheap_torch_kernel(
        "spin-exchange-proposal",
        _spin_exchange_kernel,
        configs,
        _proposal_sites(i, j, configs),
        compile_kernels=compile_kernels,
    )
    if _return_move_codes:
        torch = _require_torch()
        return (
            proposed,
            changed,
            torch.full(
                (configs.shape[0],),
                _MOVE_EXCHANGE,
                dtype=torch.long,
                device=configs.device,
            ),
        )
    return proposed, changed


def propose_spinful_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose spinful Hubbard exchange/hopping moves on one edge.

    The proposal preserves ``N_up`` and ``N_down``. With probability
    ``1 - hopping_rate`` it swaps the two local site states. Otherwise it uses
    local hopping-style moves over ``empty/up/down/double`` states, following
    the sampling options in ``sjdu10/vmc_torch``.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    device = configs.device
    batch = configs.shape[0]

    if compile_kernels:
        encoding_codes = torch.as_tensor(
            (encoding.empty, encoding.double, encoding.up, encoding.down),
            dtype=configs.dtype,
            device=device,
        )
        randoms = torch.rand(
            (3, batch),
            device=device,
            generator=generator,
        )
        result = _run_cheap_torch_kernel(
            "spinful-exchange-hopping-proposal",
            _spinful_exchange_hopping_kernel,
            configs,
            _proposal_sites(i, j, configs),
            float(hopping_rate),
            encoding_codes,
            randoms[0],
            randoms[1] < 0.5,
            randoms[2] < 0.5,
            compile_kernels=True,
        )
        if _return_move_codes:
            return result
        return result[:2]

    proposed = configs.clone()

    ci = configs[:, i]
    cj = configs[:, j]
    changed = ci != cj
    if not torch.any(changed):
        if _return_move_codes:
            rand = torch.rand(batch, device=device, generator=generator)
            move_codes = torch.where(
                rand < (1.0 - hopping_rate),
                torch.full_like(rand, _MOVE_EXCHANGE, dtype=torch.long),
                torch.full_like(rand, _MOVE_HOPPING, dtype=torch.long),
            )
            return proposed, changed, move_codes
        return proposed, changed

    n_up, n_down = encoding.decode(configs)
    ni = n_up[:, i] + n_down[:, i]
    nj = n_up[:, j] + n_down[:, j]
    delta_n = (ni - nj).abs()

    rand = torch.rand(batch, device=device, generator=generator)
    is_exchange = (rand < (1.0 - hopping_rate)) & changed
    is_hopping = (~is_exchange) & changed
    move_codes = torch.where(
        rand < (1.0 - hopping_rate),
        torch.full_like(rand, _MOVE_EXCHANGE, dtype=torch.long),
        torch.full_like(rand, _MOVE_HOPPING, dtype=torch.long),
    )

    swap_mask = is_exchange | (is_hopping & (delta_n == 1))
    proposed[swap_mask, i] = cj[swap_mask]
    proposed[swap_mask, j] = ci[swap_mask]

    mask_d0 = is_hopping & (delta_n == 0)
    if torch.any(mask_d0):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d0, i] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.double, device=device),
            torch.as_tensor(encoding.empty, device=device),
        )
        proposed[mask_d0, j] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.empty, device=device),
            torch.as_tensor(encoding.double, device=device),
        )

    mask_d2 = is_hopping & (delta_n == 2)
    if torch.any(mask_d2):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d2, i] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.down, device=device),
            torch.as_tensor(encoding.up, device=device),
        )
        proposed[mask_d2, j] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.up, device=device),
            torch.as_tensor(encoding.down, device=device),
        )

    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_u1_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves that preserve total spinful particle number only.

    In addition to the ``U1U1``-safe exchange and hopping moves, this rule
    includes single-site ``up <-> down`` flips. Those flips allow a ``U1``
    walker to move between different spin-resolved sectors while preserving
    ``N_up + N_down``. The selected edge and endpoint are fixed before the
    local state is inspected, so no proposal-probability correction is needed
    for the spin-flip branch.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result

    spin_flip_rate = float(spin_flip_rate)
    if not 0.0 <= spin_flip_rate <= 1.0:
        raise ValueError("spin_flip_rate must be between 0 and 1.")
    if spin_flip_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    flip_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < spin_flip_rate
    if _return_move_codes:
        move_codes = torch.where(
            flip_branch,
            torch.full_like(move_codes, _MOVE_SPIN_FLIP),
            move_codes,
        )
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.up) | (target == encoding.down)
    flip = flip_branch & valid
    proposed[flip_branch] = configs[flip_branch]
    if not torch.any(flip):
        changed = changed & ~flip_branch
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    flipped = torch.where(
        target == encoding.up,
        torch.as_tensor(encoding.down, device=device),
        torch.as_tensor(encoding.up, device=device),
    )
    proposed[flip & choose_i, i] = flipped[flip & choose_i]
    proposed[flip & ~choose_i, j] = flipped[flip & ~choose_i]
    changed = torch.where(flip_branch, flip, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves that preserve spinful total fermion parity.

    The U1-preserving exchange, hopping, and spin-flip moves are augmented by
    an ``empty <-> double`` toggle on a randomly selected endpoint. The latter
    changes particle number by two, allowing the chain to explore the full
    fixed-parity sector rather than remaining in one fixed-number sector.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result

    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < pair_toggle_rate
    if _return_move_codes:
        move_codes = torch.where(
            pair_branch,
            torch.full_like(move_codes, _MOVE_PAIR_TOGGLE),
            move_codes,
        )
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.empty) | (target == encoding.double)
    pair_toggle = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    if not torch.any(pair_toggle):
        changed = changed & ~pair_branch
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    toggled = torch.where(
        target == encoding.empty,
        torch.as_tensor(encoding.double, device=device),
        torch.as_tensor(encoding.empty, device=device),
    )
    proposed[pair_toggle & choose_i, i] = toggled[pair_toggle & choose_i]
    proposed[pair_toggle & ~choose_i, j] = toggled[pair_toggle & ~choose_i]
    changed = torch.where(pair_branch, pair_toggle, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def propose_spinful_z2z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    compile_kernels=False,
    _return_move_codes=False,
):
    """Propose moves preserving spin-resolved parity ``Z2 x Z2``.

    Spin flips are deliberately disabled because they change both resolved
    parities. Exchange, species-preserving hopping, and empty/double toggles
    preserve each parity independently.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposal_result = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=0.0,
        encoding=encoding,
        generator=generator,
        compile_kernels=compile_kernels,
        _return_move_codes=_return_move_codes,
    )
    if _return_move_codes:
        proposed, changed, move_codes = proposal_result
    else:
        proposed, changed = proposal_result
    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        if _return_move_codes:
            return proposed, changed, move_codes
        return proposed, changed

    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=configs.device,
        generator=generator,
    ) < pair_toggle_rate
    if _return_move_codes:
        move_codes = torch.where(
            pair_branch,
            torch.full_like(move_codes, _MOVE_PAIR_TOGGLE),
            move_codes,
        )
    valid_empty_double = (ci == encoding.empty) & (cj == encoding.empty)
    valid_double_empty = (ci == encoding.double) & (cj == encoding.double)
    valid_up_up = (ci == encoding.up) & (cj == encoding.up)
    valid_down_down = (ci == encoding.down) & (cj == encoding.down)
    valid = (
        valid_empty_double
        | valid_double_empty
        | valid_up_up
        | valid_down_down
    )
    pair_move = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    proposed[pair_move & valid_empty_double, i] = encoding.double
    proposed[pair_move & valid_empty_double, j] = encoding.double
    proposed[pair_move & valid_double_empty, i] = encoding.empty
    proposed[pair_move & valid_double_empty, j] = encoding.empty
    proposed[pair_move & valid_up_up, i] = encoding.down
    proposed[pair_move & valid_up_up, j] = encoding.down
    proposed[pair_move & valid_down_down, i] = encoding.up
    proposed[pair_move & valid_down_down, j] = encoding.up
    changed = torch.where(pair_branch, pair_move, changed)
    if _return_move_codes:
        return proposed, changed, move_codes
    return proposed, changed


def _empty_proposal_stats():
    """Create the move-wise counters used by optional sampler diagnostics."""
    return {
        name: {
            "selected": 0,
            "no_op": 0,
            "proposed": 0,
            "accepted": 0,
        }
        for name in _PROPOSAL_MOVE_NAMES
    }


def _accumulate_proposal_stats(stats, move_codes, changed, accepted=None):
    """Accumulate selected, no-op, proposed, and accepted move counts."""
    torch = _require_torch()

    def add_counts(mask, field):
        if not torch.any(mask):
            return
        counts = torch.bincount(
            move_codes[mask],
            minlength=len(_PROPOSAL_MOVE_NAMES),
        ).tolist()
        for name, count in zip(_PROPOSAL_MOVE_NAMES, counts):
            stats[name][field] += int(count)

    add_counts(torch.ones_like(changed, dtype=torch.bool), "selected")
    add_counts(~changed, "no_op")
    add_counts(changed, "proposed")
    if accepted is not None:
        add_counts(accepted, "accepted")
    return stats


def _accumulate_accepted_proposal_stats(stats, move_codes, accepted):
    """Add acceptances after a proposal's Metropolis decision."""
    torch = _require_torch()
    if not torch.any(accepted):
        return stats
    counts = torch.bincount(
        move_codes[accepted],
        minlength=len(_PROPOSAL_MOVE_NAMES),
    ).tolist()
    for name, count in zip(_PROPOSAL_MOVE_NAMES, counts):
        stats[name]["accepted"] += int(count)
    return stats


def _merge_proposal_stats(total, update):
    """Merge independently collected proposal diagnostics."""
    if update is None:
        return total
    if total is None:
        total = _empty_proposal_stats()
    for name in _PROPOSAL_MOVE_NAMES:
        for field in total[name]:
            total[name][field] += int(update[name][field])
    return total


_PROPOSAL_MIX_FAMILIES = {
    "spinful": ("hopping_rate",),
    "hubbard": ("hopping_rate",),
    "spinful_exchange_hopping": ("hopping_rate",),
    "spinful_u1": ("hopping_rate", "spin_flip_rate"),
    "u1_spinful": ("hopping_rate", "spin_flip_rate"),
    "spinful_total": ("hopping_rate", "spin_flip_rate"),
    "spinful_total_exchange_hopping": ("hopping_rate", "spin_flip_rate"),
    "spinful_z2": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "z2_spinful": ("hopping_rate", "spin_flip_rate", "pair_toggle_rate"),
    "spinful_parity": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "spinful_parity_exchange_hopping": (
        "hopping_rate",
        "spin_flip_rate",
        "pair_toggle_rate",
    ),
    "spinful_z2z2": ("hopping_rate", "pair_toggle_rate"),
    "z2z2_spinful": ("hopping_rate", "pair_toggle_rate"),
    "spinful_resolved_parity": ("hopping_rate", "pair_toggle_rate"),
}


def _proposal_move_score(stats, names):
    selected = sum(stats[name]["selected"] for name in names)
    if selected == 0:
        return None
    accepted = sum(stats[name]["accepted"] for name in names)
    return accepted / selected


def _adapt_proposal_mix_rate(
    owner,
    attribute,
    candidate_moves,
    reference_moves,
    stats,
    *,
    adaptation_rate,
    min_probability,
    max_probability,
):
    """Update one conditional move probability from whole-sweep statistics."""
    candidate_score = _proposal_move_score(stats, candidate_moves)
    reference_score = _proposal_move_score(stats, reference_moves)
    current = float(getattr(owner, attribute))
    if (
        candidate_score is None
        or reference_score is None
        or not 0.0 < current < 1.0
    ):
        return False

    # The statistic is the fraction of selections which led to an accepted
    # configuration change, so invalid/no-op branches are penalized too.
    logit = math.log(current / (1.0 - current))
    logit += adaptation_rate * (candidate_score - reference_score)
    if logit >= 0.0:
        updated = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        updated = exp_logit / (1.0 + exp_logit)
    setattr(
        owner,
        attribute,
        min(max(updated, min_probability), max_probability),
    )
    return True


def _proposal_mix_rates(owner):
    return {
        "hopping_rate": float(owner.hopping_rate),
        "spin_flip_rate": float(owner.spin_flip_rate),
        "pair_toggle_rate": float(owner.pair_toggle_rate),
    }


def _warmup_proposal_mix(
    owner,
    *,
    n_sweeps,
    adaptation_rate,
    min_probability,
    max_probability,
    progress,
):
    """Tune symmetric proposal weights between warm-up sweeps only."""
    n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
    adaptation_rate = float(adaptation_rate)
    min_probability = float(min_probability)
    max_probability = float(max_probability)
    if not math.isfinite(adaptation_rate) or adaptation_rate <= 0.0:
        raise ValueError("adaptation_rate must be a finite positive number.")
    if not 0.0 <= min_probability < max_probability <= 1.0:
        raise ValueError(
            "Require 0 <= min_probability < max_probability <= 1."
        )

    supported_rates = _PROPOSAL_MIX_FAMILIES.get(str(owner.proposal), ())
    total_stats = _empty_proposal_stats()
    history = []
    bar = _make_progress(
        progress,
        total=n_sweeps,
        desc="Torch VMC proposal warm-up",
        unit="sweep",
    )
    try:
        for sweep in range(1, n_sweeps + 1):
            result = owner.sample_sweep(
                n_sweeps=1,
                track_proposal_stats=True,
            )
            if result is None:
                raise ValueError("Cannot tune a proposal mix on an empty graph.")
            stats = result.proposal_stats
            _merge_proposal_stats(total_stats, stats)

            if "hopping_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "hopping_rate",
                    ("hopping",),
                    ("exchange",),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            if "spin_flip_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "spin_flip_rate",
                    ("spin_flip",),
                    ("exchange", "hopping"),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            if "pair_toggle_rate" in supported_rates:
                _adapt_proposal_mix_rate(
                    owner,
                    "pair_toggle_rate",
                    ("pair_toggle",),
                    ("exchange", "hopping", "spin_flip"),
                    stats,
                    adaptation_rate=adaptation_rate,
                    min_probability=min_probability,
                    max_probability=max_probability,
                )
            history.append(
                {
                    "sweep": sweep,
                    "rates": _proposal_mix_rates(owner),
                    "proposal_stats": stats,
                }
            )
            if bar is not None:
                bar.update(1)
                set_postfix = getattr(bar, "set_postfix", None)
                if callable(set_postfix):
                    set_postfix(_proposal_mix_rates(owner))
    finally:
        if bar is not None:
            bar.close()

    summary = {
        "n_sweeps": n_sweeps,
        "rates": _proposal_mix_rates(owner),
        "proposal_stats": total_stats,
        "history": tuple(history),
    }
    owner.last_proposal_tuning = summary
    return summary


def _safe_metropolis_ratio(proposed_amps, current_amps):
    torch = _require_torch()
    numerator = proposed_amps.abs().square()
    denominator = current_amps.abs().square()
    zero = torch.zeros_like(numerator)
    inf = torch.full_like(numerator, float("inf"))
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.where(numerator > 0, inf, zero),
    )


def _safe_metropolis_log_ratio(
    proposed_log_abs,
    current_log_abs,
    *,
    proposed_nonzero=None,
    current_nonzero=None,
):
    """Return clipped Metropolis ratios from log magnitudes."""
    torch = _require_torch()
    log_ratio = 2.0 * (proposed_log_abs - current_log_abs)
    # ``inf - inf`` can occur for user-provided log amplitudes. The explicit
    # support masks below decide those zero-amplitude cases, so make the
    # finite-ratio branch harmless instead of propagating NaNs into RNG tests.
    log_ratio = torch.where(
        torch.isnan(log_ratio),
        torch.zeros_like(log_ratio),
        log_ratio,
    )
    ratio = torch.exp(torch.minimum(log_ratio, torch.zeros_like(log_ratio)))
    if proposed_nonzero is None or current_nonzero is None:
        return ratio
    zero = torch.zeros_like(ratio)
    one = torch.ones_like(ratio)
    return torch.where(
        current_nonzero,
        torch.where(proposed_nonzero, ratio, zero),
        torch.where(proposed_nonzero, one, zero),
    )


def metropolis_exchange_sweep(
    configs,
    amplitude_fn,
    graph,
    *,
    current_amplitudes=None,
    current_log_abs=None,
    current_nonzero=None,
    log_amplitude_fn=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    chunk_size=None,
    compile_kernels=False,
    track_proposal_stats=False,
):
    """Run one nearest-neighbor Metropolis sweep.

    ``amplitude_fn`` should accept a ``(batch, n_sites)`` torch integer tensor
    and return a batch of amplitudes. The sampler evaluates only changed
    proposals when possible. ``chunk_size`` caps proposal-amplitude batch size
    without changing the Markov chain. Set ``track_proposal_stats=True`` to
    retain move-wise selected, no-op, proposed, and accepted counts.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs).clone()
    current = (
        _call_amplitude_fn(amplitude_fn, configs, chunk_size=chunk_size)
        if current_amplitudes is None
        else current_amplitudes
    )
    current = torch.as_tensor(current, device=configs.device)
    log_amplitude_fn = _resolve_log_amplitude_fn(
        amplitude_fn,
        log_amplitude_fn,
    )
    if log_amplitude_fn is not None:
        try:
            if current_log_abs is None or current_nonzero is None:
                current_phase, computed_log_abs = _call_log_amplitude_fn(
                    log_amplitude_fn,
                    configs,
                    chunk_size=chunk_size,
                )
                if current_log_abs is None:
                    current_log_abs = computed_log_abs
                if current_nonzero is None:
                    current_nonzero = current_phase.abs() > 0
            current_log_abs = torch.as_tensor(
                current_log_abs,
                dtype=torch.float64,
                device=configs.device,
            ).clone()
            current_nonzero = torch.as_tensor(
                current_nonzero,
                dtype=torch.bool,
                device=configs.device,
            ).clone()
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            # Some approximate sparse contractions expose ``forward_log`` but
            # cannot represent every intermediate charge sector. Keep the
            # raw-amplitude sampler path usable in that case.
            log_amplitude_fn = None
            current_log_abs = None
            current_nonzero = None
    n_proposed = 0
    n_accepted = 0
    proposal_stats = _empty_proposal_stats() if track_proposal_stats else None

    for i, j in _iter_edges(graph):
        if proposal in {"spin", "spin_exchange", "heisenberg"}:
            proposal_result = propose_spin_exchange(
                i,
                j,
                configs,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {"spinful", "hubbard", "spinful_exchange_hopping"}:
            proposal_result = propose_spinful_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_u1",
            "u1_spinful",
            "spinful_total",
            "spinful_total_exchange_hopping",
        }:
            proposal_result = propose_spinful_u1_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_z2",
            "z2_spinful",
            "spinful_parity",
            "spinful_parity_exchange_hopping",
        }:
            proposal_result = propose_spinful_z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        elif proposal in {
            "spinful_z2z2",
            "z2z2_spinful",
            "spinful_resolved_parity",
        }:
            proposal_result = propose_spinful_z2z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
                compile_kernels=compile_kernels,
                _return_move_codes=track_proposal_stats,
            )
        else:
            raise ValueError(
                "proposal must be 'spin', 'spinful_exchange_hopping', or "
                "'spinful_u1', 'spinful_z2', or 'spinful_z2z2'."
            )

        if track_proposal_stats:
            proposed, flags, move_codes = proposal_result
            _accumulate_proposal_stats(proposal_stats, move_codes, flags)
        else:
            proposed, flags = proposal_result

        if not torch.any(flags):
            continue

        n_changed = int(flags.sum().item())
        n_proposed += n_changed
        proposed_amps = current.clone()
        proposal_amplitude_fn = getattr(
            amplitude_fn,
            "proposal_amplitudes",
            None,
        )
        if callable(proposal_amplitude_fn):
            proposed_amps[flags] = proposal_amplitude_fn(
                configs[flags],
                proposed[flags],
                current[flags],
                chunk_size=chunk_size,
            )
        else:
            proposed_amps[flags] = _call_amplitude_fn(
                amplitude_fn,
                proposed[flags],
                chunk_size=chunk_size,
            )
        if log_amplitude_fn is None:
            ratio = _safe_metropolis_ratio(proposed_amps, current)
        else:
            try:
                proposed_phase, proposed_log_abs_values = (
                    _call_log_amplitude_fn(
                        log_amplitude_fn,
                        proposed[flags],
                        chunk_size=chunk_size,
                    )
                )
                proposed_log_abs = current_log_abs.clone()
                proposed_log_abs[flags] = proposed_log_abs_values
                proposed_nonzero = current_nonzero.clone()
                proposed_nonzero[flags] = proposed_phase.abs() > 0
                ratio = _safe_metropolis_log_ratio(
                    proposed_log_abs,
                    current_log_abs,
                    proposed_nonzero=proposed_nonzero,
                    current_nonzero=current_nonzero,
                )
            except (AttributeError, IndexError, KeyError, NotImplementedError,
                    RuntimeError, TypeError, ValueError):
                log_amplitude_fn = None
                current_log_abs = None
                current_nonzero = None
                ratio = _safe_metropolis_ratio(proposed_amps, current)
        accept = flags & (
            torch.rand(configs.shape[0], device=configs.device, generator=generator)
            < ratio
        )
        if track_proposal_stats:
            _accumulate_accepted_proposal_stats(
                proposal_stats,
                move_codes,
                accept,
            )

        if torch.any(accept):
            n_accept = int(accept.sum().item())
            n_accepted += n_accept
            configs[accept] = proposed[accept]
            current[accept] = proposed_amps[accept]
            if log_amplitude_fn is not None:
                current_log_abs[accept] = proposed_log_abs[accept]
                current_nonzero[accept] = proposed_nonzero[accept]

    return TorchMetropolisResult(
        configs=configs,
        amplitudes=current,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
        log_abs_amplitudes=(
            current_log_abs if log_amplitude_fn is not None else None
        ),
        nonzero_amplitudes=(
            current_nonzero if log_amplitude_fn is not None else None
        ),
        proposal_stats=proposal_stats,
    )
