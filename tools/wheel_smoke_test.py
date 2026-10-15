"""Post-build smoke test for a freshly built pyMSTM wheel.

Run by cibuildwheel's `test-command` (see [tool.cibuildwheel] in
pyproject.toml) inside the isolated venv it creates for each wheel --
i.e. against the *installed wheel*, not the source tree. Actually solving
a tiny two-sphere cluster (rather than a bare `import pymstm`) is the
point: it exercises the compiled _mstm_ext extension end to end, catching
a wheel that imports fine but whose Fortran extension is missing,
mislinked, or miscompiled for the target Python/platform.

Lives in a real file instead of an inline `python -c "..."` in the
workflow specifically because YAML folded scalars turn newlines into
spaces, which repeatedly corrupted the one-liner (a stray leading space
became `IndentationError: unexpected indent`). A script sidesteps that
entire class of quoting/whitespace bug.
"""

import pymstm


def main() -> None:
    m = pymstm.MstmBindings()
    m.set_spheres(
        radii=[1.0, 1.0],
        positions=[[-1.5, 0, 0], [1.5, 0, 0]],
        orders=[6, 6],
        ref_re=[1.5, 1.5],
        ref_im=[0.01, 0.01],
    )
    m.set_medium_ref(1.0, 0.0)
    m.set_incident(alpha_deg=0.0, beta_deg=0.0)
    m.set_solver_params(eps=1e-3, max_iter=200)
    m.set_verbose(False)
    m.prepare()
    result = m.solve()
    m.finalize()

    qext = result["qext_tot"]
    assert qext > 0, f"expected positive Q_ext, got {qext!r}"

    # The high-level interface, in a medium (exercises the convention).
    r = pymstm.MSTM(
        radii=[1.0], positions=[(0.0, 0.0, 0.0)], ref_re=[1.5], ref_im=[0.0],
        medium_ref_re=1.33,
    ).solve()
    assert r.q_ext_total > 0

    # A lattice solve on 2 OpenMP threads (exercises the bundled libgomp):
    # bit-identical to the serial one.
    res = []
    for n_threads in (1, 2):
        m = pymstm.MstmBindings(n_threads=n_threads)
        m.set_spheres([0.3, 0.3], [[-0.4, 0, 0], [0.4, 0.1, 0.2]], [4, 4], [1.6] * 2, [0.1] * 2)
        m.set_lattice(1.5, 1.5)
        m.prepare()
        res.append(m.solve()["qabs_tot"])
        m.finalize()
    assert res[0] == res[1] > 0, res

    print(f"pyMSTM {pymstm.__version__} wheel smoke test OK (Q_ext = {qext:.6g})")


if __name__ == "__main__":
    main()
