"""Use one ``Fermion`` model for site-native MPS and qMERA workflows."""

from __future__ import annotations

import pepsy
from pepsy.optimizers.mera import QMeraGeometry


def build_workflow(length=3):
    """Build matching site terms, MPS gates, and qMERA mode terms."""
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        t=1.0,
        U=4.0,
        mu=0.0,
    )
    edges = tuple((site, site + 1) for site in range(length - 1))

    # Four-state site-native terms and a canonical stream for MPS evolution.
    site_terms = fermion.local_terms(edges)
    gate_stream = fermion.gate_stream(
        edges,
        dt=0.01,
        sites=range(length),
        order=2,
    )

    # qMERA deliberately expands each physical site into two two-state modes.
    geometry = QMeraGeometry(
        shape=length,
        site_modes=("up", "down"),
    )
    qmera_terms = fermion.local_terms(geometry, layout="qmera")

    return {
        "fermion": fermion,
        "site_terms": site_terms,
        "gate_stream": gate_stream,
        "geometry": geometry,
        "qmera_terms": qmera_terms,
    }


if __name__ == "__main__":
    workflow = build_workflow()
    print(f"site terms: {len(workflow['site_terms'])}")
    print(f"MPS gates: {len(workflow['gate_stream'])}")
    print(f"qMERA mode terms: {len(workflow['qmera_terms'])}")
