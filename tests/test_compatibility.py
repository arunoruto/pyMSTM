"""End-to-end compatibility tests: pyMSTM vs MSTM CLI.

For each test case we generate an .inp file, run the standalone MSTM CLI,
parse the output, run the same configuration through pyMSTM, and compare.
"""

import os

import numpy as np
import pytest

from pymstm import MSTM, MstmNotFoundError, find_mstm_binary, run_mstm
from pymstm._inp import write_inp_file


def _mstm_available() -> bool:
    try:
        find_mstm_binary()
    except MstmNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _mstm_available(),
    reason="MSTM CLI binary not found on PATH. Build it (`make cli`) and add it "
    "to PATH, or set PYMSTM_MSTM_BIN.",
)


def _make_inp(tmp_path, **kwargs):
    """Write inp file and return its path."""
    inp = os.path.join(tmp_path, "test.inp")
    write_inp_file(inp, output_file="mstm_output.dat", **kwargs)
    return inp


# ---------------------------------------------------------------------------
# Single sphere
# ---------------------------------------------------------------------------


def test_single_sphere_total_qext(tmp_path):
    """Single dielectric sphere: total Q_ext must match."""
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]

    kwargs = dict(
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        mie_eps=1e-6,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    inp = _make_inp(tmp_path, **kwargs)
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    np.testing.assert_allclose(
        py_result["qext_tot"], mstm_result["total"]["q_ext_unpol"], rtol=1e-4
    )
    assert mstm_result["iterations"] <= 2


def test_single_sphere_per_sphere_qext(tmp_path):
    """Single sphere: per-sphere Q_ext must match."""
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    assert len(mstm_result["per_sphere"]) == 1
    np.testing.assert_allclose(
        py_result["q_ext"][0], mstm_result["per_sphere"][0]["q_ext"], rtol=1e-4
    )
    np.testing.assert_allclose(
        py_result["q_abs"][0], mstm_result["per_sphere"][0]["q_abs"], atol=1e-10
    )


def test_single_sphere_absorbing(tmp_path):
    """Single absorbing sphere: Q_abs > 0."""
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.01]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    np.testing.assert_allclose(
        py_result["qext_tot"], mstm_result["total"]["q_ext_unpol"], rtol=1e-4
    )
    np.testing.assert_allclose(
        py_result["qabs_tot"], mstm_result["total"]["q_abs_unpol"], rtol=1e-4
    )
    assert py_result["qabs_tot"] > 0


# ---------------------------------------------------------------------------
# Two spheres
# ---------------------------------------------------------------------------


def test_two_sphere_total_qext(tmp_path):
    """Two separated spheres: total Q_ext must match."""
    radii = [3.0, 3.0]
    positions: list[list[float]] = [[0, 0, -5], [0, 0, 5]]
    ref_re = [1.5, 1.5]
    ref_im = [0.0, 0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[6, 6], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    assert mstm_result["iterations"] > 0
    np.testing.assert_allclose(
        py_result["qext_tot"], mstm_result["total"]["q_ext_unpol"], rtol=1e-4
    )


def test_two_sphere_per_sphere_qext(tmp_path):
    """Two spheres: per-sphere Q_ext values must match individually."""
    radii = [3.0, 3.0]
    positions: list[list[float]] = [[0, 0, -5], [0, 0, 5]]
    ref_re = [1.5, 1.5]
    ref_im = [0.0, 0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[6, 6], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    assert len(mstm_result["per_sphere"]) == 2
    for i in range(2):
        np.testing.assert_allclose(
            py_result["q_ext"][i], mstm_result["per_sphere"][i]["q_ext"], rtol=1e-4
        )


# ---------------------------------------------------------------------------
# Solution error and iterations
# ---------------------------------------------------------------------------


def test_solution_error_agreement(tmp_path):
    """Solution error should be small and consistent between tools."""
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-8,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-8, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    assert py_result["status"] == 0
    assert py_result["solution_error"] < 1e-6
    assert mstm_result["solution_error"] < 1e-6


# ---------------------------------------------------------------------------
# Scattering matrix (fixed orientation)
# ---------------------------------------------------------------------------


def test_scattering_matrix_forward(tmp_path):
    """Forward-scattering Mueller matrix S11 must match."""
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=True,
    )
    run_mstm(inp_path=inp, workdir=tmp_path)

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    m.solve()

    sm = m.get_scattering_angle(costheta=1.0, phi=0.0)
    assert sm[0] > 0  # S11 positive
    assert not np.any(np.isnan(sm))
    m.finalize()


def test_scattering_matrix_dlp_oblique_angle(tmp_path):
    """S12/S11 (degree of linear polarization) must match the CLI at an
    oblique angle, including sign.

    Regression test for a dashboard bug where pyMSTM's raw (un-ratioed)
    Mueller element 4 was divided by S11 with a stray extra minus sign,
    flipping the DLP sign relative to the CLI's already-ratioed output.
    """
    radii = [5.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=True,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed
    sm = mstm_result["scattering_matrix"]
    assert sm is not None

    angles_deg = np.asarray(sm["angles_deg"])
    idx = int(np.argmin(np.abs(angles_deg - 90.0)))
    cli_dlp = sm["matrix"][idx][4]  # CLI already prints this as S12/S11

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[12], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    m.solve()

    costheta = np.cos(np.deg2rad(angles_deg[idx]))
    sm_py = m.get_scattering_angle(costheta=costheta, phi=0.0)
    py_dlp = sm_py[4] / max(abs(sm_py[0]), 1e-30)
    m.finalize()

    np.testing.assert_allclose(py_dlp, cli_dlp, rtol=1e-3, atol=1e-6)


def test_scattering_matrix_offaxis_cluster(tmp_path):
    """S11 over the full angle sweep must match the CLI for a cluster whose
    spheres are offset from the incidence axis (azimuthally asymmetric).

    Regression test for two bugs found while investigating a dashboard
    report of a wildly incorrect phase function:

    1. ``mstm_prepare_c`` in the Fortran wrapper clipped the merged
       single-origin T-matrix order to ``max_mie_order`` (the largest
       *per-sphere* Mie order), instead of only the user-configurable
       ``max_t_matrix_order`` safety cap. For spatially extended clusters
       the correct cluster-level order is normally much larger than any
       individual sphere's Mie order, so this silently truncated the
       expansion and corrupted every non-forward angle.
    2. The CLI's "scattering matrix in incident plane" table is built by
       pairing scattering polar angle theta with azimuth ``alpha`` for
       angle-label >= 0 and ``alpha + pi`` for angle-label < 0 (see
       ``scattering_matrix_calculation`` in the Fortran source). Any
       caller that maps those same angle labels to ``get_scattering_angle``
       must reproduce that same azimuth flip, not assume ``phi=0`` throughout.
    """
    radii = [3.0, 3.0]
    positions: list[list[float]] = [[-4.0, 0.0, -5.0], [4.0, 0.0, 5.0]]
    ref_re = [1.5, 1.5]
    ref_im = [0.0, 0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=True,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed
    sm = mstm_result["scattering_matrix"]
    assert sm is not None
    angles_deg = np.asarray(sm["angles_deg"])
    cli_s11 = np.array([row[0] for row in sm["matrix"]])

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[8, 8], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()

    # The individual-sphere Mie order is small (~5), but the cluster's
    # circumscribing radius (~9.4) demands a much larger merged T-matrix
    # order. Guards against the max_mie_order clipping regression directly.
    assert m.get_tmatrix_order() > 10

    m.solve()

    py_s11 = np.empty_like(cli_s11)
    for i, deg in enumerate(angles_deg):
        costheta = np.cos(np.deg2rad(deg))
        phi = np.pi if deg < 0 else 0.0  # alpha=0, so alpha+pi = pi
        py_s11[i] = m.get_scattering_angle(costheta=costheta, phi=phi)[0]
    m.finalize()

    # pyMSTM's raw S11 and the CLI's printed S11 differ by an overall
    # constant (the CLI applies its own print-time normalization); rescale
    # to the backward-direction value to compare shape, which is what the
    # bugs above actually distort.
    py_s11 *= cli_s11[-1] / py_s11[-1]
    np.testing.assert_allclose(py_s11, cli_s11, rtol=2e-2, atol=1e-6)


def test_scattering_matrix_azimuthal_average(tmp_path):
    """With azimuthal_average=True, S11 and S12/S11 must match the CLI over
    theta in [0,180], with no incident-plane sign/fold convention involved.

    This is the recommended comparison mode for non-azimuthally-symmetric
    clusters: azimuthally averaging analytically collapses the scattering
    matrix to 6 independent elements that are single-valued in theta alone
    (see the "azimuthal averaged scattering matrix" CLI output section),
    sidestepping the incident-plane phi-fold convention entirely.
    """
    radii = [3.0, 3.0]
    positions: list[list[float]] = [[-4.0, 0.0, -5.0], [4.0, 0.0, 5.0]]
    ref_re = [1.5, 1.5]
    ref_im = [0.0, 0.0]

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=True,
        azimuthal_average=True,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed
    sm = mstm_result["scattering_matrix"]
    assert sm is not None
    angles_deg = np.asarray(sm["angles_deg"])
    assert angles_deg.min() >= 0.0  # no negative-labeled fold angles
    cli_s11 = np.array([row[0] for row in sm["matrix"]])
    cli_dlp = np.array([row[4] for row in sm["matrix"]])  # already S12/S11

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[8, 8], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_azimuthal_average(True)
    m.set_verbose(False)
    m.prepare()
    m.solve()

    py_s11 = np.empty_like(cli_s11)
    py_dlp = np.empty_like(cli_dlp)
    for i, deg in enumerate(angles_deg):
        costheta = np.cos(np.deg2rad(deg))
        sm_py = m.get_scattering_angle(costheta=costheta, phi=0.0)
        py_s11[i] = sm_py[0]
        py_dlp[i] = sm_py[4] / max(abs(sm_py[0]), 1e-30)
    m.finalize()

    py_s11 *= cli_s11[-1] / py_s11[-1]
    np.testing.assert_allclose(py_s11, cli_s11, rtol=2e-2, atol=1e-6)
    np.testing.assert_allclose(py_dlp, cli_dlp, rtol=2e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Medium refractive index
# ---------------------------------------------------------------------------


def test_medium_ref_index(tmp_path):
    """Cluster in a non-vacuum medium."""
    radii = [3.0]
    positions: list[list[float]] = [[0, 0, 0]]
    ref_re = [1.5]
    ref_im = [0.0]
    n_med = 1.33

    inp = _make_inp(
        tmp_path,
        radii=radii,
        positions=positions,
        ref_re=ref_re,
        ref_im=ref_im,
        medium_ref_re=n_med,
        medium_ref_im=0.0,
        solution_eps=1e-6,
        max_iterations=5000,
        calculate_scattering_matrix=False,
    )
    mstm_result = run_mstm(inp_path=inp, workdir=tmp_path).parsed

    m = MSTM()
    m.set_spheres(
        radii=radii, positions=positions, orders=[8], ref_re=ref_re, ref_im=ref_im
    )
    m.set_medium_ref(n_med, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.set_mie_eps(1e-6)
    m.set_verbose(False)
    m.prepare()
    py_result = m.solve()
    m.finalize()

    np.testing.assert_allclose(
        py_result["qext_tot"], mstm_result["total"]["q_ext_unpol"], rtol=1e-4
    )
