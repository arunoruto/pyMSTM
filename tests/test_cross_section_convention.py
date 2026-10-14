"""cross_section_convention: "physical" vs MSTM's own normalisation.

MSTM normalises efficiencies in a medium of index n by the vacuum
irradiance of the incident amplitude, so they are Re(n) times the physical
ones. "Physical" must therefore equal the relative setup (vacuum medium,
sphere index m/n, lengths scaled by n), and a single sphere must give the
Lorenz-Mie efficiencies in the host.
"""

import numpy as np
import pytest

from pymstm import MSTM, MstmBindings, MstmNotFoundError, MstmTask, find_mstm_binary, solve_task

N_HOST = 1.33
M = 1.5 + 0.1j
CLUSTER = dict(radii=[1.0, 0.8], positions=[[0, 0, 0], [0, 0, 2.5]])


def _solve(radii, positions, m, n_host, scale=1.0, cell=None, convention="physical"):
    b = MstmBindings(cross_section_convention=convention)
    b.set_length_scale(scale)
    k = len(radii)
    b.set_spheres(radii, positions, [10] * k, [m.real] * k, [m.imag] * k)
    b.set_medium_ref(n_host.real, n_host.imag)
    if cell is not None:
        b.set_lattice(*cell)
    b.prepare()
    r = b.solve()
    r["mie"] = b.get_mie_efficiencies()
    b.finalize()
    return r


def _medium_and_relative(**kw):
    medium = _solve(m=M, n_host=N_HOST, **CLUSTER, **kw)
    relative = _solve(m=M / N_HOST, n_host=1.0, scale=N_HOST, **CLUSTER, **kw)
    return medium, relative


KEYS = ["qext_tot", "qabs_tot", "qsca_tot", "q_ext", "q_abs", "q_sca"]


@pytest.mark.parametrize("cell", [None, (3.0, 3.0)], ids=["cluster", "lattice"])
def test_physical_equals_relative_setup(cell):
    medium, relative = _medium_and_relative(cell=cell)
    for k in KEYS:
        np.testing.assert_allclose(medium[k], relative[k], rtol=1e-8, err_msg=k)


def test_mstm_is_re_n_times_physical():
    for n_host in (N_HOST, 1.33 + 0.02j):
        phys = _solve(m=M, n_host=n_host, **CLUSTER)
        raw = _solve(m=M, n_host=n_host, convention="mstm", **CLUSTER)
        for k in KEYS:
            np.testing.assert_allclose(raw[k], n_host.real * phys[k], rtol=1e-12, err_msg=k)
        np.testing.assert_allclose(raw["mie"], n_host.real**2 * np.array(phys["mie"]), rtol=1e-12)


def test_vacuum_unchanged():
    phys = _solve(m=M, n_host=1.0, **CLUSTER)
    raw = _solve(m=M, n_host=1.0, convention="mstm", **CLUSTER)
    for k in KEYS:
        np.testing.assert_array_equal(raw[k], phys[k])


def test_single_sphere_is_mie_in_the_host():
    """Reference: MSTM's own Lorenz-Mie routine for the sphere in vacuum
    with relative index m/n and size parameter n*x."""
    x = 3.0
    phys = _solve([x], [[0, 0, 0]], M, N_HOST)
    mie_ext, mie_abs = _solve([N_HOST * x], [[0, 0, 0]], M / N_HOST, 1.0)["mie"]
    assert phys["qext_tot"] == pytest.approx(mie_ext[0], rel=1e-6)
    assert phys["qabs_tot"] == pytest.approx(mie_abs[0], rel=1e-6)
    np.testing.assert_allclose(phys["mie"], (mie_ext, mie_abs), rtol=1e-12)


def test_large_sphere_extinction_paradox():
    """Physical Q_ext of a large sphere in a medium tends to 2 (from above,
    ~x**(-2/3) edge term); MSTM's own value tends to 2 Re(n)."""
    q = _solve([60.0], [[0, 0, 0]], 1.5 + 0.1j, N_HOST)["qext_tot"]
    assert 2.0 < q < 2.1


def test_solve_task_and_mstm_model_follow_the_convention():
    ref = _solve(m=M, n_host=N_HOST, **CLUSTER)
    for conv, f in (("physical", 1.0), ("mstm", N_HOST)):
        task = MstmTask(**CLUSTER, ref_index=M, medium_ref_index=N_HOST,
                        orders=[10, 10], cross_section_convention=conv)
        assert solve_task(task)["qext_tot"] == pytest.approx(f * ref["qext_tot"], rel=1e-8)
        model = MSTM(radii=CLUSTER["radii"], positions=CLUSTER["positions"],
                     ref_re=[M.real] * 2, ref_im=[M.imag] * 2, medium_ref_re=N_HOST,
                     cross_section_convention=conv)
        assert model.solve().q_ext_total == pytest.approx(f * ref["qext_tot"], rel=1e-5)


def test_bad_convention_rejected():
    with pytest.raises(ValueError):
        MstmBindings(cross_section_convention="vacuum")


def _cli_available():
    try:
        find_mstm_binary()
    except MstmNotFoundError:
        return False
    return True


@pytest.mark.skipif(not _cli_available(), reason="MSTM CLI binary not found")
@pytest.mark.parametrize("convention", ["physical", "mstm"])
def test_bindings_and_cli_agree_in_a_medium(convention):
    problem = MSTM(radii=[2.0, 1.5], positions=[(0, 0, -2.0), (0, 0, 2.0)],
                   ref_re=[1.5, 1.5], ref_im=[0.05, 0.05], medium_ref_re=N_HOST,
                   cross_section_convention=convention, compute_mueller=True,
                   solution_eps=1e-8)
    b = problem.solve()
    c = problem.model_copy(update={"cli": True}).solve()
    for k in ("q_ext_total", "q_abs_total", "q_sca_total"):
        assert getattr(b, k) == pytest.approx(getattr(c, k), rel=1e-3), k
    for pb, pc in zip(b.per_sphere, c.per_sphere):
        assert pb.q_ext == pytest.approx(pc.q_ext, rel=1e-3)
    # The phase function does not depend on the convention.
    for pb, pc in zip(b.mueller, c.mueller):
        assert pb.s11 == pytest.approx(pc.s11, rel=0.02)


@pytest.mark.skipif(not _cli_available(), reason="MSTM CLI binary not found")
def test_mstm_convention_is_the_raw_cli_output():
    from pymstm import run_mstm

    kw = dict(radii=[2.0, 1.5], positions=[(0, 0, -2.0), (0, 0, 2.0)],
              ref_re=[1.5, 1.5], ref_im=[0.05, 0.05])
    raw = run_mstm(inp_kwargs=dict(**kw, medium_ref_re=N_HOST, solution_eps=1e-8,
                                   calculate_scattering_matrix=False)).parsed["total"]
    r = MSTM(**kw, medium_ref_re=N_HOST, cross_section_convention="mstm",
             solution_eps=1e-8, cli=True).solve()
    assert r.q_ext_total == raw["q_ext_unpol"]
    assert r.q_abs_total == raw["q_abs_unpol"]
