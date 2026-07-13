"""Smoke test for the in-progress ctypes -> f2py migration.

Not a comparison test (see test_compatibility.py for that) -- this only
proves the f2py extension (built via `make f2py-ext`, see
src/pymstm/_fortran/mstm_f2py.f90 and the Makefile's f2py-ext target)
still imports and behaves correctly as more subroutines get ported in
Stage 2. Skipped entirely if the extension hasn't been built.
"""

import glob
import os

import numpy as np
import pytest

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_GLOB = os.path.join(_PROJ_ROOT, "src", "pymstm", "_mstm_ext*.so")

pytestmark = pytest.mark.skipif(
    not glob.glob(_EXT_GLOB),
    reason="f2py extension not built. Run 'make f2py-ext' first.",
)


def _import_ext():
    import importlib
    import sys

    sys.path.insert(0, os.path.join(_PROJ_ROOT, "src"))
    import pymstm._mstm_ext as ext

    importlib.reload(ext)  # avoid stale module-level state across tests
    return ext


def test_init_sets_defaults():
    # t_matrix_order/number_spheres are NOT reset by mstm_init() -- they're
    # only (re)computed by set_spheres()/prepare(), so they're not asserted
    # here. This is module-level Fortran state shared across the whole
    # process (same as the pre-migration ctypes .so), so a prior test's
    # set_spheres()/prepare() call can leave them non-zero; that's expected,
    # not a bug.
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    assert ext.inputinterface.mie_epsilon == pytest.approx(1e-6)
    assert ext.inputinterface.azimuthal_average == 0
    assert ext.spheredata.max_t_matrix_order == 100
    ext.mstm_f2py_bindings.mstm_finalize()


def test_pure_attribute_read_write():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.inputinterface.mie_epsilon = 1e-9
    assert ext.inputinterface.mie_epsilon == pytest.approx(1e-9)
    ext.spheredata.max_t_matrix_order = 50
    assert ext.spheredata.max_t_matrix_order == 50
    ext.random_sphere_configuration.target_shape = 2
    assert ext.random_sphere_configuration.target_shape == 2
    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_verbose():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_verbose(1)
    assert ext.numconstants.print_intermediate_results == 1
    assert ext.numconstants.light_up == 1
    ext.mstm_f2py_bindings.mstm_set_verbose(0)
    assert ext.numconstants.print_intermediate_results == 0
    assert ext.numconstants.light_up == 0
    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_medium_ref():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_medium_ref(1.33, 0.0)
    assert ext.inputinterface.medium_ref_index == pytest.approx(1.33 + 0j)
    assert ext.surface_subroutines.layer_ref_index[0] == pytest.approx(1.33 + 0j)
    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_incident():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_incident(30.0, 45.0)
    assert ext.inputinterface.incident_alpha_deg == pytest.approx(30.0)
    assert ext.inputinterface.incident_beta_deg == pytest.approx(45.0)
    assert ext.inputinterface.incident_beta_specified == 1
    ext.mstm_f2py_bindings.mstm_set_incident(0.0, 0.0)
    assert ext.inputinterface.incident_beta_specified == 0
    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_solver_params():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_solver_params(1e-7, 2000, 1)
    assert ext.inputinterface.solution_epsilon == pytest.approx(1e-7)
    assert ext.inputinterface.max_iterations == 2000
    assert bytes(ext.inputinterface.solution_method).strip()[:1] == b"d"
    ext.mstm_f2py_bindings.mstm_set_solver_params(1e-7, 2000, 0)
    assert bytes(ext.inputinterface.solution_method).strip()[:1] == b"i"
    ext.mstm_f2py_bindings.mstm_finalize()


def test_layer_and_lattice_setters():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()

    ext.mstm_f2py_bindings.mstm_set_layer_count(2)
    assert ext.surface_subroutines.number_plane_boundaries == 2
    assert ext.surface_subroutines.plane_surface_present == 1

    ext.mstm_f2py_bindings.mstm_set_layer_thickness(1, 2.5)
    assert ext.surface_subroutines.layer_thickness[0] == pytest.approx(2.5)

    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(1, 1.5, 0.02)
    assert ext.surface_subroutines.layer_ref_index[1] == pytest.approx(1.5 + 0.02j)

    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(0, 1.1, 0.0)
    assert ext.surface_subroutines.layer_ref_index[0] == pytest.approx(1.1 + 0j)
    assert ext.inputinterface.medium_ref_index == pytest.approx(1.1 + 0j)

    ext.mstm_f2py_bindings.mstm_set_lattice(5.0, 5.0, 1, 0)
    assert ext.periodic_lattice_subroutines.periodic_lattice == 1

    np.testing.assert_allclose(ext.periodic_lattice_subroutines.cell_width, [5.0, 5.0])

    ext.mstm_f2py_bindings.mstm_clear_lattice()
    assert ext.periodic_lattice_subroutines.periodic_lattice == 0

    ext.mstm_f2py_bindings.mstm_finalize()


def _prepare_two_sphere(ext):

    radii = np.array([3.0, 3.0])
    pos = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]).T
    orders = np.array([6, 6], dtype=np.int32)
    ref_re = np.array([1.5, 1.5])
    ref_im = np.array([0.0, 0.0])
    ext.mstm_f2py_bindings.mstm_set_spheres(orders, radii, pos, ref_re, ref_im)
    ext.mstm_f2py_bindings.mstm_set_medium_ref(1.0, 0.0)
    ext.mstm_f2py_bindings.mstm_set_incident(0.0, 0.0)
    ext.mstm_f2py_bindings.mstm_set_solver_params(1e-6, 5000, 0)


def test_prepare_normal_incidence():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    _prepare_two_sphere(ext)
    ext.mstm_f2py_bindings.mstm_prepare()

    assert ext.spheredata.t_matrix_order > 0
    assert ext.spheredata.cross_section_radius > 0
    np.testing.assert_allclose(ext.spheredata.cluster_origin, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(ext.spheredata.host_sphere, [0, 0])
    assert ext.inputinterface.incident_sin_beta == pytest.approx(0.0)
    assert ext.inputinterface.incident_direction == 1

    ext.mstm_f2py_bindings.mstm_finalize()


def test_prepare_oblique_incidence_snells_law():
    """Regression test for the mstm_set_incident_c latent bug found during
    this migration: incident_sin_beta/incident_direction must be derived
    from beta_deg via Snell's law using the incidence-side layer's
    refractive index, not a naive sin(beta_deg) with a caller-supplied
    direction.
    """
    import math

    ext = _import_ext()

    ext.mstm_f2py_bindings.mstm_init()
    radii = np.array([2.0])
    pos = np.array([[0.0, 0.0, 0.0]]).T
    orders = np.array([6], dtype=np.int32)
    ref_re = np.array([1.5])
    ref_im = np.array([0.0])
    ext.mstm_f2py_bindings.mstm_set_spheres(orders, radii, pos, ref_re, ref_im)
    ext.mstm_f2py_bindings.mstm_set_medium_ref(1.33, 0.0)
    ext.mstm_f2py_bindings.mstm_set_layer_count(1)
    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(0, 1.33, 0.0)
    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(1, 1.0, 0.0)
    ext.mstm_f2py_bindings.mstm_set_incident(0.0, 30.0)  # beta <= 90 -> direction 1
    ext.mstm_f2py_bindings.mstm_set_solver_params(1e-6, 5000, 0)
    ext.mstm_f2py_bindings.mstm_prepare()

    expected = math.sin(math.radians(30.0)) / 1.33
    assert ext.inputinterface.incident_sin_beta == pytest.approx(expected, rel=1e-5)
    assert ext.inputinterface.incident_direction == 1
    ext.mstm_f2py_bindings.mstm_finalize()

    # beta > 90 -> direction switches to 2, uses the far-boundary layer's index
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_spheres(orders, radii, pos, ref_re, ref_im)
    ext.mstm_f2py_bindings.mstm_set_medium_ref(1.33, 0.0)
    ext.mstm_f2py_bindings.mstm_set_layer_count(1)
    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(0, 1.33, 0.0)
    ext.mstm_f2py_bindings.mstm_set_layer_ref_index(1, 2.0, 0.0)
    ext.mstm_f2py_bindings.mstm_set_incident(0.0, 120.0)
    ext.mstm_f2py_bindings.mstm_set_solver_params(1e-6, 5000, 0)
    ext.mstm_f2py_bindings.mstm_prepare()

    expected2 = math.sin(math.radians(120.0)) / 2.0
    assert ext.inputinterface.incident_sin_beta == pytest.approx(expected2, rel=1e-5)
    assert ext.inputinterface.incident_direction == 2
    ext.mstm_f2py_bindings.mstm_finalize()


def test_solve_matches_ctypes_path():
    """The f2py mstm_solve port must be numerically identical to the
    existing ctypes-based MSTM.solve() for the same configuration.
    """
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    _prepare_two_sphere(ext)
    ext.mstm_f2py_bindings.mstm_prepare()

    (q_ext, q_abs, q_sca, qext_tot, qabs_tot, qsca_tot, sol_err, niter, status) = (
        ext.mstm_f2py_bindings.mstm_solve(2)
    )
    ext.mstm_f2py_bindings.mstm_finalize()

    assert status == 0
    assert not np.any(np.isnan(q_ext))

    from pymstm import MSTM

    m = MSTM()
    m.set_spheres(
        radii=[3.0, 3.0],
        positions=[[0, 0, -5], [0, 0, 5]],
        orders=[6, 6],
        ref_re=[1.5, 1.5],
        ref_im=[0.0, 0.0],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(0, 0, 1)
    m.set_solver_params(eps=1e-6, max_iter=5000)
    m.prepare()
    r = m.solve()
    m.finalize()

    np.testing.assert_allclose(q_ext, r["q_ext"], rtol=1e-10)
    np.testing.assert_allclose(q_sca, r["q_sca"], atol=1e-10)
    assert qext_tot == pytest.approx(r["qext_tot"], rel=1e-10)
    assert qsca_tot == pytest.approx(r["qsca_tot"], rel=1e-10)


def test_scattering_angle_and_scat_mat_access():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    _prepare_two_sphere(ext)
    ext.mstm_f2py_bindings.mstm_prepare()
    ext.mstm_f2py_bindings.mstm_solve(2)

    sm = ext.mstm_f2py_bindings.mstm_scattering_angle(1.0, 0.0)
    assert sm.shape == (16,)
    assert sm[0] > 0  # S11 positive at forward direction
    assert not np.any(np.isnan(sm))

    # scat_mat and its bounds are directly accessible module attributes --
    # no dedicated get-smatrix subroutine needed (unlike the ctypes era's
    # manual flatten-in-Fortran/reshape-in-Python dance).
    scat_mat = ext.inputinterface.scat_mat
    assert scat_mat.shape == (ext.inputinterface.scat_mat_mdim,
        ext.inputinterface.scat_mat_udim - ext.inputinterface.scat_mat_ldim + 1)

    ext.mstm_f2py_bindings.mstm_finalize()


def test_compute_and_ranorient_smatrix():
    """Also a regression test for two real bugs found in
    mstm_ranorient_smatrix_c (present pre-migration, not introduced by
    this port): a fixed-length character*30 dummy argument in
    ranorientscatmatrix read past the end of a shorter unpadded filename
    buffer (the source of a stray garbled "tmatrix_temp.dat<garbage>" file
    once left in this project's working tree), and an unconditionally
    passed `override_order=0` silently forced the T-matrix order to zero
    regardless of the file's real content, making every coefficient
    trivially zero.
    """
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    _prepare_two_sphere(ext)
    ext.mstm_f2py_bindings.mstm_set_verbose(0)
    ext.mstm_f2py_bindings.mstm_prepare()

    t_order = int(ext.spheredata.t_matrix_order)
    nentries = sum(2 * (2 * l + 1) * 2 * l * (l + 2) for l in range(1, t_order + 1))
    array_len = 2 * nentries

    tmatrix_data, tmat_order, q_ext, q_abs, status = (
        ext.mstm_f2py_bindings.mstm_compute_tmatrix(n=2, array_len=array_len)
    )
    assert status == 0
    assert tmat_order > 0
    assert not np.any(np.isnan(tmatrix_data))
    norm2 = np.sum(tmatrix_data[0::2] ** 2 + tmatrix_data[1::2] ** 2)
    assert norm2 > 1.0

    sm_coef, cm_coef, tmat_order_out = ext.mstm_f2py_bindings.mstm_ranorient_smatrix(
        "tmatrix_temp.dat", tmat_order=tmat_order
    )
    assert tmat_order_out > 0
    n_used = 16 * (2 * tmat_order_out + 1)
    assert np.count_nonzero(sm_coef[:n_used]) > 0  # regression: was all-zero

    for ct in (1.0, 0.5, 0.0, -0.5, -1.0):
        sm = ext.mstm_f2py_bindings.mstm_ranorient_smatrix_at_angle(
            sm_coef[:n_used], ct, tmat_order_in=tmat_order_out
        )
        assert sm.shape == (16,)
        assert not np.any(np.isnan(sm))
    # S11 (index 0) should be forward-peaked for this dimer
    sm_forward = ext.mstm_f2py_bindings.mstm_ranorient_smatrix_at_angle(
        sm_coef[:n_used], 1.0, tmat_order_in=tmat_order_out
    )
    sm_backward = ext.mstm_f2py_bindings.mstm_ranorient_smatrix_at_angle(
        sm_coef[:n_used], -1.0, tmat_order_in=tmat_order_out
    )
    assert sm_forward[0] > sm_backward[0] > 0

    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_scattering_map():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    ext.mstm_f2py_bindings.mstm_set_scattering_map(10, -90.0, 90.0)
    assert ext.inputinterface.scattering_map_dimension == 10
    assert ext.inputinterface.scat_mat_amin == pytest.approx(-90.0)
    assert ext.inputinterface.scat_mat_amax == pytest.approx(90.0)
    assert ext.inputinterface.scat_mat_ldim == -10
    assert ext.inputinterface.scat_mat_udim == 10
    assert ext.inputinterface.scat_mat_mdim == 32
    assert ext.inputinterface.calculate_scattering_matrix == 1
    ext.mstm_f2py_bindings.mstm_finalize()


def test_set_excitation_switch():
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    radii = np.array([3.0, 3.0, 3.0])
    pos = np.array([[0, 0, -6], [0, 0, 0], [0, 0, 6]], dtype=float).T
    orders = np.array([6, 6, 6], dtype=np.int32)
    ref_re = np.array([1.5, 1.5, 1.5])
    ref_im = np.array([0.0, 0.0, 0.0])
    ext.mstm_f2py_bindings.mstm_set_spheres(orders, radii, pos, ref_re, ref_im)
    ext.mstm_f2py_bindings.mstm_set_excitation_switch([1, 0, 1])
    np.testing.assert_array_equal(
        ext.inputinterface.sphere_excitation_switch, [1, 0, 1]
    )
    ext.mstm_f2py_bindings.mstm_finalize()


def test_length_scale_and_azimuthal_average_pure_attributes():
    """No dedicated subroutines needed for these -- confirms they're
    reachable as plain module attributes."""
    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()
    assert ext.inputinterface.length_scale_factor == pytest.approx(1.0)
    ext.inputinterface.length_scale_factor = 2.5
    assert ext.inputinterface.length_scale_factor == pytest.approx(2.5)
    ext.inputinterface.azimuthal_average = True
    ext.inputinterface.numerical_azimuthal_average = True
    assert ext.inputinterface.azimuthal_average == 1
    assert ext.inputinterface.numerical_azimuthal_average == 1
    ext.mstm_f2py_bindings.mstm_finalize()


def test_derived_type_variables_hidden():
    ext = _import_ext()
    assert not hasattr(ext.spheredata, "sphere_links")
    assert not hasattr(ext.random_sphere_configuration, "cell_list")
    assert not hasattr(ext.random_sphere_configuration, "coll_data")


def test_set_spheres():

    ext = _import_ext()
    ext.mstm_f2py_bindings.mstm_init()

    radii = np.array([5.0, 3.0])
    pos = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]).T  # shape (3, n)
    orders = np.array([12, 8], dtype=np.int32)
    ref_re = np.array([1.5, 1.5])
    ref_im = np.array([0.01, 0.0])

    ext.mstm_f2py_bindings.mstm_set_spheres(orders, radii, pos, ref_re, ref_im)

    assert ext.spheredata.number_spheres == 2
    np.testing.assert_allclose(ext.spheredata.sphere_radius, radii)
    np.testing.assert_allclose(ext.spheredata.sphere_position, pos)
    np.testing.assert_allclose(
        ext.spheredata.sphere_ref_index[:, 1:], [[1.5 + 0.01j, 1.5], [1.5 + 0.01j, 1.5]]
    )
    np.testing.assert_array_equal(ext.spheredata.sphere_order, orders)
    np.testing.assert_array_equal(ext.spheredata.host_sphere, [0, 0])

    ext.mstm_f2py_bindings.mstm_finalize()
