"""Tests for pymstm._cli.run_mstm() / find_mstm_binary()."""

import os

import pytest

from pymstm import (
    MstmExecutionError,
    MstmNotFoundError,
    find_mstm_binary,
    run_mstm,
)
from pymstm._inp import write_inp_file


def _mstm_available() -> bool:
    try:
        find_mstm_binary()
    except MstmNotFoundError:
        return False
    return True


def _mstm_mpi_available() -> bool:
    try:
        find_mstm_binary(mpi=True)
    except MstmNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _mstm_available(),
    reason="MSTM CLI binary not found on PATH. Build it (`make cli`) and add it "
    "to PATH, or set PYMSTM_MSTM_BIN.",
)

_SPHERE_KWARGS = dict(
    radii=[5.0],
    positions=[[0.0, 0.0, 0.0]],
    ref_re=[1.5],
    ref_im=[0.0],
    solution_eps=1e-6,
    max_iterations=5000,
    calculate_scattering_matrix=False,
)


# ---------------------------------------------------------------------------
# Input modes
# ---------------------------------------------------------------------------


def test_run_mstm_inp_kwargs_single_sphere():
    result = run_mstm(inp_kwargs=_SPHERE_KWARGS)
    assert result.returncode == 0
    assert result.parsed["total"]["q_ext_unpol"] > 0
    assert len(result.runs) == 1


def test_run_mstm_inp_text(tmp_path):
    inp_path = os.path.join(tmp_path, "test.inp")
    write_inp_file(inp_path, **_SPHERE_KWARGS)
    with open(inp_path) as f:
        text = f.read()

    result = run_mstm(inp_text=text)
    assert result.returncode == 0
    assert result.parsed["total"]["q_ext_unpol"] > 0


def test_run_mstm_inp_path_different_dir(tmp_path):
    """inp_path in a different directory than workdir exercises the
    copy-into-workdir logic (MSTM writes output relative to CWD)."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    work_dir = tmp_path / "work"
    inp_path = src_dir / "test.inp"
    write_inp_file(inp_path, **_SPHERE_KWARGS)

    result = run_mstm(inp_path=inp_path, workdir=work_dir, keep_workdir=True)
    assert result.returncode == 0
    assert result.parsed["total"]["q_ext_unpol"] > 0
    assert result.inp_path == work_dir / "run.inp"
    assert result.inp_path.is_file()


def test_run_mstm_inp_path_same_dir_no_copy(tmp_path):
    """inp_path already inside workdir shouldn't need copying -- the
    original file is used directly (named 'test.inp', not 'run.inp')."""
    inp_path = tmp_path / "test.inp"
    write_inp_file(inp_path, **_SPHERE_KWARGS)

    result = run_mstm(inp_path=inp_path, workdir=tmp_path)
    assert result.returncode == 0
    assert result.inp_path == inp_path


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_run_mstm_no_args_raises():
    with pytest.raises(ValueError, match="Exactly one of"):
        run_mstm()


def test_run_mstm_multiple_args_raises():
    with pytest.raises(ValueError, match="Exactly one of"):
        run_mstm(inp_kwargs=_SPHERE_KWARGS, inp_text="whatever")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_run_mstm_missing_binary_override():
    with pytest.raises(MstmNotFoundError):
        run_mstm(inp_kwargs=_SPHERE_KWARGS, binary_path="/no/such/mstm")


def test_run_mstm_malformed_inp_raises(tmp_path):
    """A malformed .inp produces no output file even if MSTM itself
    exits 0 -- run_mstm() must still surface this as an error rather
    than silently returning a bogus result."""
    bad_inp = tmp_path / "bad.inp"
    bad_inp.write_text("this is not a valid mstm input file\n")

    with pytest.raises(MstmExecutionError) as excinfo:
        run_mstm(inp_path=bad_inp)
    assert excinfo.value.cmd


def test_run_mstm_keep_workdir(tmp_path):
    work_dir = tmp_path / "work"
    result = run_mstm(inp_kwargs=_SPHERE_KWARGS, workdir=work_dir, keep_workdir=True)
    assert work_dir.is_dir()
    assert result.output_path.is_file()


# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _mstm_mpi_available(),
    reason="mstm-mpi binary not found on PATH. Build it (`make mpi` or the Nix "
    "flake's mstm-mpi package) and add it to PATH, or set PYMSTM_MSTM_MPI_BIN.",
)
def test_run_mstm_mpi():
    serial = run_mstm(inp_kwargs=_SPHERE_KWARGS)
    mpi_result = run_mstm(inp_kwargs=_SPHERE_KWARGS, mpi_processes=2)
    assert mpi_result.returncode == 0
    assert mpi_result.parsed["total"]["q_ext_unpol"] == pytest.approx(
        serial.parsed["total"]["q_ext_unpol"], rel=1e-3
    )


def test_run_mstm_mpi_binary_not_found_without_mpi_install():
    """mpi_processes= must look for mstm-mpi specifically, not silently
    fall back to the serial binary."""
    if _mstm_mpi_available():
        pytest.skip("mstm-mpi is available in this environment")
    with pytest.raises(MstmNotFoundError):
        run_mstm(inp_kwargs=_SPHERE_KWARGS, mpi_processes=2)
