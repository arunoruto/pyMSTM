"""Tests for pymstm.MstmProblem/MstmResult -- the unified bindings/CLI
interface used for benchmarking and cross-checking the two backends."""

import pydantic
import pytest

from pymstm import MstmNotFoundError, MstmProblem, find_mstm_binary


def _mstm_available() -> bool:
    try:
        find_mstm_binary()
    except MstmNotFoundError:
        return False
    return True


_SINGLE_SPHERE = dict(
    radii=[5.0],
    positions=[(0.0, 0.0, 0.0)],
    ref_re=[1.5],
    ref_im=[0.0],
)

_TWO_SPHERE = dict(
    radii=[3.0, 3.0],
    positions=[(-5.0, 0.3, 0.1), (5.0, -0.4, 0.2)],
    ref_re=[1.5, 1.5],
    ref_im=[0.01, 0.01],
    solution_eps=1e-8,
    max_iterations=2000,
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_positions_length_raises():
    with pytest.raises(pydantic.ValidationError, match="positions"):
        MstmProblem(radii=[5.0, 3.0], positions=[(0, 0, 0)], ref_re=[1.5], ref_im=[0.0])


def test_mismatched_ref_index_length_raises():
    with pytest.raises(pydantic.ValidationError, match="ref_re"):
        MstmProblem(
            radii=[5.0], positions=[(0, 0, 0)], ref_re=[1.5, 1.5], ref_im=[0.0]
        )


def test_negative_radius_raises():
    with pytest.raises(pydantic.ValidationError, match="positive"):
        MstmProblem(radii=[-5.0], positions=[(0, 0, 0)], ref_re=[1.5], ref_im=[0.0])


def test_no_spheres_raises():
    with pytest.raises(pydantic.ValidationError, match="at least one sphere"):
        MstmProblem(radii=[], positions=[], ref_re=[], ref_im=[])


def test_cli_mueller_tilted_incidence_raises():
    with pytest.raises(pydantic.ValidationError, match="incident_beta_deg"):
        MstmProblem(
            **_SINGLE_SPHERE,
            cli=True,
            compute_mueller=True,
            incident_beta_deg=30.0,
        )


def test_cli_mueller_zero_incidence_ok():
    # Should NOT raise -- beta=0 is fine for cli + compute_mueller.
    MstmProblem(**_SINGLE_SPHERE, cli=True, compute_mueller=True, incident_beta_deg=0.0)


# ---------------------------------------------------------------------------
# Bindings backend (always available)
# ---------------------------------------------------------------------------


def test_solve_bindings_single_sphere():
    result = MstmProblem(**_SINGLE_SPHERE).solve()
    assert result.backend == "bindings"
    assert result.q_ext_total > 0
    assert result.wall_time_seconds > 0
    assert len(result.per_sphere) == 1
    assert result.mueller is None


def test_solve_bindings_mueller():
    result = MstmProblem(
        **_SINGLE_SPHERE, compute_mueller=True, n_theta=11
    ).solve()
    assert result.mueller is not None
    assert len(result.mueller) == 11
    assert result.mueller[0].theta_deg == 0.0
    assert result.mueller[-1].theta_deg == 180.0
    assert all(p.s11 >= 0 for p in result.mueller)


def test_solve_bindings_mueller_integrates_to_4pi_for_single_sphere():
    """Spherically-symmetric single sphere: the 1D theta cut IS the full
    angular distribution, so this must hold tightly (see n_theta's own
    docstring for why a multi-sphere cluster wouldn't)."""
    import numpy as np

    result = MstmProblem(
        radii=[5.0], positions=[(0.0, 0.0, 0.0)], ref_re=[1.5], ref_im=[0.01],
        compute_mueller=True, n_theta=181,
    ).solve()
    theta = np.array([p.theta_deg for p in result.mueller])
    s11 = np.array([p.s11 for p in result.mueller])
    theta_rad = np.radians(theta)
    integral = 2 * np.pi * np.trapezoid(s11 * np.sin(theta_rad), theta_rad)
    assert integral == pytest.approx(4 * np.pi, rel=0.01)


# ---------------------------------------------------------------------------
# CLI backend (skipped if mstm binary not on PATH)
# ---------------------------------------------------------------------------


pytestmark_cli = pytest.mark.skipif(
    not _mstm_available(),
    reason="MSTM CLI binary not found on PATH. Build it (`make cli`) and add it "
    "to PATH, or set PYMSTM_MSTM_BIN.",
)


@pytestmark_cli
def test_solve_cli_single_sphere():
    result = MstmProblem(**_SINGLE_SPHERE, cli=True).solve()
    assert result.backend == "cli"
    assert result.q_ext_total > 0
    assert result.wall_time_seconds > 0
    assert len(result.per_sphere) == 1


@pytestmark_cli
def test_bindings_and_cli_cross_sections_agree():
    problem = MstmProblem(**_TWO_SPHERE)
    r_bindings = problem.solve()
    r_cli = problem.model_copy(update={"cli": True}).solve()

    assert r_bindings.backend == "bindings"
    assert r_cli.backend == "cli"
    assert r_bindings.q_ext_total == pytest.approx(r_cli.q_ext_total, rel=1e-3)
    assert r_bindings.q_abs_total == pytest.approx(r_cli.q_abs_total, rel=1e-3)
    assert r_bindings.q_sca_total == pytest.approx(r_cli.q_sca_total, rel=1e-3)


@pytestmark_cli
def test_bindings_and_cli_mueller_agree_on_same_grid():
    """The whole point of this class: flip cli=True/False and get a
    directly comparable result on the exact same theta grid."""
    problem = MstmProblem(**_TWO_SPHERE, compute_mueller=True, n_theta=21)
    r_bindings = problem.solve()
    r_cli = problem.model_copy(update={"cli": True}).solve()

    assert len(r_bindings.mueller) == len(r_cli.mueller) == 21
    for pb, pc in zip(r_bindings.mueller, r_cli.mueller):
        assert pb.theta_deg == pc.theta_deg
        assert pb.s11 == pytest.approx(pc.s11, rel=0.02)


@pytestmark_cli
def test_benchmark_both_backends_report_positive_wall_time():
    """Not a performance assertion (too environment-dependent) -- just
    confirms both backends' timing is actually measured, which is the
    whole point of having both in one comparable result shape."""
    problem = MstmProblem(**_SINGLE_SPHERE)
    r_bindings = problem.solve()
    r_cli = problem.model_copy(update={"cli": True}).solve()
    assert r_bindings.wall_time_seconds > 0
    assert r_cli.wall_time_seconds > 0
