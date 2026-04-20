"""Regression tests for :mod:`pepsy.ham` builders."""

import numpy as np
import pytest
import quimb

import pepsy as py
from pepsy.core import OneDMap


def test_build_mpo_single_site_term_works_for_ly1():
    """Single-term MPO build should work for 1D (Ly=1) layouts."""
    builder = py.ham_tn(Lx=2, Ly=1, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    mpo = builder.build_mpo(
        [
            ((z_op,), ((0, 0),)),
        ],
        compress_each=False,
    )

    assert mpo.L == 2


def test_build_mpo_accepts_mapper_override():
    """build_mpo should allow a one-off mapper override for term placement."""
    builder_default = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    mapper = OneDMap(2, 2, mode="row-major")
    z_op = quimb.pauli("Z", dtype="complex128")

    ints = [
        ((z_op,), ((0, 0),)),
        ((z_op,), ((1, 1),), 0.5),
    ]

    mpo_from_override = builder_default.build_mpo(
        ints,
        compress_each=False,
        mapper=mapper,
    )

    builder_row_major = py.ham_tn(Lx=2, Ly=2, data_type="complex128", mapper=mapper)
    mpo_from_mapper_builder = builder_row_major.build_mpo(ints, compress_each=False)

    assert mpo_from_override.L == 4
    assert np.allclose(mpo_from_override.to_dense(), mpo_from_mapper_builder.to_dense())


def test_build_mpo_uses_canonical_ops_sites_coeff_order():
    """build_mpo should accept the canonical (ops, sites, coeff) term order."""
    builder = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    x_op = quimb.pauli("X", dtype="complex128")

    ints = [
        ((z_op,), ((0, 0),), 0.5),
        ((z_op, z_op), ((0, 0), (1, 0)), 1.0),
        ((x_op,), ((1, 1),), -0.25),
    ]

    mpo = builder.build_mpo(ints, compress_each=False)

    assert mpo.L == 4


def test_build_mpo_rejects_legacy_sites_ops_order():
    """build_mpo should reject the old (sites, ops, coeff) term order."""
    builder = py.ham_tn(Lx=2, Ly=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")

    ints_legacy = [
        (((0, 0),), (z_op,), 0.5),
    ]

    with pytest.raises(TypeError, match="Only 2D coordinates are supported"):
        builder.build_mpo(ints_legacy, compress_each=False)


def test_build_itf_lattice_ly1_has_chain_edges():
    """Ly=1 ITF lattice should reduce to a nearest-neighbor 1D chain."""
    out = py.ham_tn.build_itf_lattice(
        Lx=5,
        Ly=1,
        lattice="square",
        J=1.0,
        field=0.5,
        return_edges=True,
    )

    assert out["builder"].Lx == 5
    assert out["builder"].Ly == 1
    assert out["mpo"].L == 5
    assert out["pepo"] is None
    assert all(y == 0 for _, y in out["one_d_to_two_d"].values())

    expected = {frozenset((i, i + 1)) for i in range(4)}
    got = {frozenset(edge) for edge in out["edges_1d"]}
    assert got == expected


def test_build_itf_lattice_ly1_cyclic_drops_degenerate_edges():
    """Ly=1 with cyclic=True should ignore degenerate singleton periodic edges."""
    out = py.ham_tn.build_itf_lattice(
        Lx=5,
        Ly=1,
        lattice="square",
        cyclic=True,
        J=1.0,
        field=0.5,
        return_edges=True,
    )

    assert out["mpo"].L == 5
    assert out["pepo"] is None

    expected = {frozenset((i, i + 1)) for i in range(4)}
    expected.add(frozenset((0, 4)))
    got = {frozenset(edge) for edge in out["edges_1d"]}
    assert got == expected


def test_build_itf_lattice_can_return_pepo_explicitly():
    """build_itf_lattice should only include the PEPO when requested."""
    out = py.ham_tn.build_itf_lattice(
        Lx=3,
        Ly=1,
        lattice="square",
        J=1.0,
        field=0.5,
        return_pepo=True,
    )

    assert out["mpo"].L == 3
    assert out["pepo"].Lx == 3
    assert out["pepo"].Ly == 1


def test_build_itf_lattice_show_returns_schematic_drawing():
    """build_itf_lattice(show=True) should include a schematic MPO drawing."""
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        show=True,
    )

    assert out["mpo"].L == 4
    assert out["pepo"] is None
    assert hasattr(out["drawing"], "fig")
    assert hasattr(out["drawing"], "ax")
    assert "Square ITF MPO" in out["drawing"].ax.get_title()


def test_build_itf_lattice_accepts_mapper_instance():
    """build_itf_lattice should accept a preconfigured OneDMap instance."""
    mapper = OneDMap(2, 2, mode="snake-row-major")
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        mapper=mapper,
        return_edges=True,
    )

    assert out["builder"].mapper is mapper
    assert out["builder"].map_mode == "snake-row-major"
    assert out["one_d_to_lattice"] == mapper.build()[0]
    assert out["mpo"].L == 4
    assert out["pepo"] is None


def test_build_itf_lattice_allows_non_snake_mapper_for_default_mpo():
    """build_itf_lattice should allow non-snake mappers when only MPO is requested."""
    mapper = OneDMap(2, 2, mode="row-major")
    out = py.ham_tn.build_itf_lattice(
        Lx=2,
        Ly=2,
        lattice="square",
        J=1.0,
        field=0.5,
        mapper=mapper,
        return_edges=True,
    )

    assert out["builder"].mapper is mapper
    assert out["mpo"].L == 4
    assert out["pepo"] is None


def test_build_itf_lattice_rejects_non_snake_mapper_for_pepo():
    """build_itf_lattice should reject non-snake mappers when PEPO is requested."""
    mapper = OneDMap(2, 2, mode="row-major")

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        py.ham_tn.build_itf_lattice(
            Lx=2,
            Ly=2,
            lattice="square",
            J=1.0,
            field=0.5,
            mapper=mapper,
            return_edges=True,
            return_pepo=True,
        )


def test_ham_tn_rejects_mapper_shape_mismatch():
    """ham_tn should fail clearly when mapper shape does not match builder shape."""
    with pytest.raises(ValueError, match="mapper shape"):
        py.ham_tn(Lx=2, Ly=3, data_type="complex128", mapper=OneDMap(2, 2, mode="snake"))


def test_map_builder_snake_2d_matches_expected_layout():
    """Default 2D snake traversal should match the legacy ham_tn mapping."""
    map_, map_inv = OneDMap.build(3, 2, mode="snake")
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (2, 0),
        5: (2, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_3d_snake():
    """3D snake mode should return consistent 1D<->3D maps."""
    map_, map_inv = OneDMap.build(2, 2, Lz=2, mode="snake")

    assert len(map_) == 8
    assert len(map_inv) == 8
    assert all(len(coord) == 3 for coord in map_.values())
    assert all(map_inv[coord] == idx for idx, coord in map_.items())


def test_map_builder_supports_row_major_snake_mode():
    """snake-row-major should snake along x within each y row."""
    map_, map_inv = OneDMap.build(3, 2, mode="snake-row-major")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (2, 1),
        4: (1, 1),
        5: (0, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_hilbert_mode():
    """hilbert mode should follow the standard 4x4 Hilbert traversal."""
    map_, map_inv = OneDMap.build(4, 4, mode="hilbert")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (1, 1),
        3: (0, 1),
        4: (0, 2),
        5: (0, 3),
        6: (1, 3),
        7: (1, 2),
        8: (2, 2),
        9: (2, 3),
        10: (3, 3),
        11: (3, 2),
        12: (3, 1),
        13: (2, 1),
        14: (2, 0),
        15: (3, 0),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}
    assert all(
        abs(map_[idx][0] - map_[idx + 1][0]) + abs(map_[idx][1] - map_[idx + 1][1]) == 1
        for idx in range(len(map_) - 1)
    )


def test_map_builder_supports_hilbert_row_major_mode():
    """hilbert-row-major should expose the transposed Hilbert orientation."""
    map_, map_inv = OneDMap.build(4, 4, mode="hilbert-row")
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (2, 0),
        5: (3, 0),
        6: (3, 1),
        7: (2, 1),
        8: (2, 2),
        9: (3, 2),
        10: (3, 3),
        11: (2, 3),
        12: (1, 3),
        13: (1, 2),
        14: (0, 2),
        15: (0, 3),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_hilbert_rejects_unsupported_shapes():
    """hilbert mode should still reject 3D usage clearly."""
    with pytest.raises(NotImplementedError, match="only for 2D lattices"):
        OneDMap.build(2, 2, Lz=2, mode="hilbert")


def test_map_builder_supports_rectangular_hilbert_mode():
    """hilbert mode should cover rectangular 2D lattices via cropped Hilbert order."""
    map_, map_inv = OneDMap.build(3, 5, mode="hilbert")

    assert len(map_) == 15
    assert len(map_inv) == 15
    assert set(map_.values()) == {(x, y) for x in range(3) for y in range(5)}
    assert all(map_inv[coord] == idx for idx, coord in map_.items())
    assert map_[0] == (0, 0)


def test_map_builder_supports_rectangular_hilbert_row_major_mode():
    """Row-major Hilbert should also cover rectangular lattices with swapped orientation."""
    map_, map_inv = OneDMap.build(3, 5, mode="hilbert-row-major")

    assert len(map_) == 15
    assert len(map_inv) == 15
    assert set(map_.values()) == {(x, y) for x in range(3) for y in range(5)}
    assert all(map_inv[coord] == idx for idx, coord in map_.items())
    assert map_[0] == (0, 0)
    assert map_ != OneDMap.build(3, 5, mode="hilbert")[0]


def test_ham_tn_accepts_builtin_mapping_mode_string():
    """ham_tn should accept a OneDMap instance from core builder."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, mode="row-major"),
    )
    assert builder.map == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def test_ham_tn_normalizes_map_mode_aliases_via_onedmap():
    """ham_tn should store the normalized OneDMap mode and mapping helper."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="hilbert-row"),
    )

    assert isinstance(builder.mapper, OneDMap)
    assert builder.map_mode == "hilbert-row-major"
    assert builder.mapper.mode == "hilbert-row-major"
    assert builder.map == builder.mapper.build()[0]


def test_ham_tn_supports_3d_mapping_and_mpo_terms():
    """3D builders should accept (x, y, z) coordinates in interaction terms."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        Lz=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, Lz=2, mode="snake"),
    )
    z_op = quimb.pauli("Z", dtype="complex128")

    assert builder.L == 8
    assert all(len(coord) == 3 for coord in builder.map.values())
    assert builder.map_site((1, 1, 1)) == builder.map_inv[(1, 1, 1)]

    mpo = builder.build_mpo(
        [
            ((z_op,), ((0, 0, 0),)),
            ((z_op, z_op), ((0, 0, 0), (1, 0, 0)), 0.5),
        ],
        compress_each=False,
    )
    assert mpo.L == 8


def test_ham_tn_snake_row_major_supports_pepo_conversion():
    """snake-row-major should remain eligible for 2D MPO->PEPO conversion."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="snake-row-major"),
    )
    pepo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)

    assert pepo.Lx == 4
    assert pepo.Ly == 4
    assert coord_to_chain == builder.map_inv


def test_ham_tn_hilbert_mode_rejects_pepo_conversion():
    """Hilbert mode should be available for mapping, but not for PEPO builds."""
    builder = py.ham_tn(
        Lx=4,
        Ly=4,
        data_type="complex128",
        mapper=OneDMap(4, 4, mode="hilbert"),
    )

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)


def test_ham_tn_row_major_mode_rejects_pepo_conversion():
    """Non-snake maps should fail clearly when PEPO conversion is requested."""
    builder = py.ham_tn(
        Lx=2,
        Ly=2,
        data_type="complex128",
        mapper=OneDMap(2, 2, mode="row-major"),
    )

    with pytest.raises(NotImplementedError, match="snake-style 2D mapping"):
        builder.mpo_itf(J=1.0, field=0.5, as_pepo=True)


def test_ham_tn_3d_rejects_pepo_conversion():
    """3D builders should raise a clear error for 2D-only PEPO conversion."""
    builder = py.ham_tn(Lx=2, Ly=2, Lz=2, data_type="complex128")
    z_op = quimb.pauli("Z", dtype="complex128")
    mpo = builder.build_mpo([((z_op,), ((0, 0, 0),))], compress_each=False)

    try:
        builder.mpo_to_pepo(mpo)
    except NotImplementedError as exc:
        assert "only available for 2D builders" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for 3D mpo_to_pepo.")


def test_mpo_itf_works_for_2d_builder():
    """mpo_itf should build the default 2D square-lattice ITF MPO."""
    builder = py.ham_tn(Lx=3, Ly=2, data_type="complex128")
    mpo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5)

    assert mpo.L == 6
    assert coord_to_chain == builder.map_inv


def test_mpo_itf_works_for_3d_builder_and_blocks_pepo():
    """mpo_itf should support 3D builders and reject PEPO output there."""
    builder = py.ham_tn(Lx=2, Ly=2, Lz=2, data_type="complex128")
    mpo, coord_to_chain = builder.mpo_itf(J=1.0, field=0.5)

    assert mpo.L == 8
    assert coord_to_chain == builder.map_inv
    assert all(len(coord) == 3 for coord in coord_to_chain)

    try:
        builder.mpo_itf(as_pepo=True)
    except NotImplementedError as exc:
        assert "only available for 2D builders" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for 3D mpo_itf(as_pepo=True).")


def test_map_builder_supports_col_major_mode():
    """col_major mode should enumerate x fastest within each y row."""
    map_, map_inv = OneDMap.build(2, 3, mode="col_major")
    expected = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
        4: (0, 2),
        5: (1, 2),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_supports_instance_style_build():
    """OneDMap should support object-style construction then build()."""
    mapper = OneDMap(3, 2, mode="row-major")

    assert mapper.shape == (3, 2)
    assert mapper.mode == "row-major"

    map_, map_inv = mapper.build()
    expected = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
        4: (2, 0),
        5: (2, 1),
    }
    assert map_ == expected
    assert map_inv == {coord: idx for idx, coord in expected.items()}


def test_map_builder_instance_style_can_override_mode_per_call():
    """Instance build/show calls should allow temporary mode overrides."""
    mapper = OneDMap(3, 2, mode="row-major")

    map_, _ = mapper.build(mode="snake")
    assert map_[0] == (0, 0)
    assert map_[1] == (0, 1)
    assert map_[2] == (1, 1)


def test_map_builder_instance_style_supports_3d_build():
    """Instance-style build() should preserve the 3D mapping modes."""
    mapper = OneDMap(2, 2, Lz=2, mode="col-major")
    map_, map_inv = mapper.build()

    assert mapper.shape == (2, 2, 2)
    assert len(map_) == 8
    assert all(len(coord) == 3 for coord in map_.values())
    assert all(map_inv[coord] == idx for idx, coord in map_.items())


def test_map_builder_show_returns_schematic_drawing():
    """show() should now return a schematic drawing object for 2D maps."""
    drawing = OneDMap.show(2, 2, mode="snake")
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawing.ax.get_title() == "OneDMap snake (2x2)"


def test_map_builder_instance_show_returns_schematic_drawing():
    """Instance-style show() should return a drawing and honor override kwargs."""
    mapper = OneDMap(2, 2, mode="row-major")
    drawing = mapper.show(mode="snake", title="Instance Mapper")

    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawing.ax.get_title() == "Instance Mapper"


def test_map_builder_show_rejects_3d():
    """show() should fail clearly for 3D maps until a schematic 3D view exists."""
    with pytest.raises(NotImplementedError, match="only available for 2D lattices"):
        OneDMap.show(2, 2, Lz=2, mode="snake")
