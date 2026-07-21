"""
First-class invocation of the standalone MSTM CLI binary (serial or MPI).

Unifies write_inp_file() -> subprocess -> parse_mstm_output() into one
call, with PATH-based binary discovery (overridable), explicit error
handling, and optional MPI invocation via ``mpiexec``. This is the
canonical way to drive the MSTM CLI from Python; downstream consumers
should use this instead of reimplementing the same three steps
themselves.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ._inp import write_inp_file
from ._inp_parser import parse_inp_text
from ._parser import parse_all_runs


class MstmNotFoundError(RuntimeError):
    """Raised when the mstm/mstm-mpi/mpiexec binary cannot be located."""


class MstmExecutionError(RuntimeError):
    """Raised when the MSTM CLI process exits non-zero, times out, or
    doesn't produce the expected output file.

    Attributes
    ----------
    returncode : int or None
        Process exit code, or None if it never started/timed out.
    stdout, stderr : str
        Captured process output.
    cmd : list of str
        The exact argv that was invoked.
    """

    def __init__(self, message, *, returncode=None, stdout="", stderr="", cmd=None):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.cmd = cmd or []


@dataclasses.dataclass
class MstmRunResult:
    """Result of a single :func:`run_mstm` invocation.

    Attributes
    ----------
    returncode : int
        Process exit code (0 on success).
    stdout, stderr : str
        Captured process output.
    output_path : Path
        Path to the parsed output (.dat) file.
    parsed : dict
        Result of ``parse_mstm_output()`` on *output_path* (first run, for
        the common single-run case) -- same shape as ``runs[0]``.
    runs : list of dict
        Result of ``parse_all_runs()`` -- more than one element when the
        .inp used ``loop_variable``/``new_run`` and produced multiple
        result blocks; a single element for ordinary single-run
        invocations (in which case it's identical to ``[parsed]``).
    inp_path : Path
        Path to the generated (or caller-supplied) .inp file, for
        debugging/reproducing a failing run.
    workdir : Path
        Directory the process was run in (a temp dir unless the caller
        passed *workdir* explicitly).
    """

    returncode: int
    stdout: str
    stderr: str
    output_path: Path
    parsed: dict[str, Any]
    runs: list[dict[str, Any]]
    inp_path: Path
    workdir: Path


# Real .inp files in the wild use either of these two output-file names;
# probed in order when introspection of the .inp's own `output_file` line
# fails or wasn't attempted (e.g. a hand-written .inp using an unusual
# construct pymstm._inp_parser can't parse).
_FALLBACK_OUTPUT_NAMES = ("mstm_output.dat", "mstmtest.dat")


def find_mstm_binary(
    *, mpi: bool = False, override: str | os.PathLike[str] | None = None
) -> Path:
    """Locate the mstm / mstm-mpi CLI binary.

    Resolution order:

    1. *override*, if given (must exist, else :class:`MstmNotFoundError`).
    2. ``PYMSTM_MSTM_BIN`` / ``PYMSTM_MSTM_MPI_BIN`` env var (serial/mpi
       respectively), if set.
    3. ``shutil.which("mstm")`` / ``shutil.which("mstm-mpi")`` (PATH).

    Deliberately does NOT fall back to a repo-relative ``build/mstm``
    path (the pattern this replaces, previously used ad hoc by this
    project's own dashboard) -- PATH/env/override are the only supported
    discovery mechanisms, so behavior is identical whether pyMSTM is
    installed as a package, run from a checkout, or used via the Nix
    flake's ``mstm``/``mstm-mpi`` packages (which install to a normal
    ``$out/bin``, i.e. end up on PATH in a devShell/``nix run``).

    Parameters
    ----------
    mpi : bool
        If True, look for ``mstm-mpi`` instead of ``mstm``.
    override : path-like, optional
        Explicit path to use, bypassing all other discovery.

    Returns
    -------
    Path

    Raises
    ------
    MstmNotFoundError
        If no binary can be found by any of the above.
    """
    name = "mstm-mpi" if mpi else "mstm"
    env_var = "PYMSTM_MSTM_MPI_BIN" if mpi else "PYMSTM_MSTM_BIN"

    if override is not None:
        p = Path(override)
        if not p.is_file():
            raise MstmNotFoundError(f"Explicit {name} binary not found: {p}")
        return p

    env_val = os.environ.get(env_var)
    if env_val:
        p = Path(env_val)
        if not p.is_file():
            raise MstmNotFoundError(f"{env_var}={env_val} does not exist")
        return p

    found = shutil.which(name)
    if found is None:
        raise MstmNotFoundError(
            f"Could not find '{name}' on PATH. Build it (e.g. `make cli` or the "
            f"Nix flake's `{name}` package output) and ensure it's on PATH, or "
            f"pass binary_path= / set {env_var}."
        )
    return Path(found)


def _find_mpiexec(mpiexec_path: str) -> str:
    p = Path(mpiexec_path)
    if p.is_absolute() or os.sep in mpiexec_path:
        if not p.is_file():
            raise MstmNotFoundError(f"mpiexec not found at {mpiexec_path}")
        return str(p)
    found = shutil.which(mpiexec_path)
    if found is None:
        raise MstmNotFoundError(
            f"Could not find '{mpiexec_path}' on PATH -- required for "
            "mpi_processes= invocation. Install an MPI implementation "
            "(e.g. OpenMPI) or pass mpiexec_path= explicitly."
        )
    return found


def _resolve_output_filename(
    *,
    inp_kwargs: dict[str, Any] | None,
    inp_text: str | None,
    inp_path: str | os.PathLike[str] | None,
    output_filename: str,
) -> str:
    """Figure out the .inp's declared ``output_file`` name.

    Prefers introspecting the actual .inp content/kwargs over trusting
    the caller-supplied *output_filename* default, but falls back to it
    whenever introspection isn't possible or fails for any reason (never
    hard-errors just because introspection failed).
    """
    if inp_kwargs is not None:
        return str(inp_kwargs.get("output_file", output_filename))
    text = None
    if inp_text is not None:
        text = inp_text
    elif inp_path is not None:
        try:
            text = Path(inp_path).read_text()
        except OSError:
            text = None
    if text is not None:
        try:
            return parse_inp_text(text).output_file
        except Exception:
            pass
    return output_filename


def run_mstm(
    *,
    inp_kwargs: dict[str, Any] | None = None,
    inp_path: str | os.PathLike[str] | None = None,
    inp_text: str | None = None,
    binary_path: str | os.PathLike[str] | None = None,
    mpi_processes: int | None = None,
    mpiexec_path: str = "mpiexec",
    workdir: str | os.PathLike[str] | None = None,
    keep_workdir: bool = False,
    timeout: float | None = 1800.0,
    output_filename: str = "mstm_output.dat",
    env: dict[str, str] | None = None,
) -> MstmRunResult:
    """Run the standalone MSTM CLI end to end: write .inp, invoke the
    binary, parse the output.

    Exactly one of *inp_kwargs*, *inp_path*, *inp_text* must be given:

    - *inp_kwargs*: forwarded to :func:`pymstm._inp.write_inp_file` to
      generate the .inp file in the run's workdir. Most convenient for
      simple, single-run cases.
    - *inp_path*: an already-written .inp file (e.g. hand-authored, or
      produced by :func:`pymstm._config.config_to_inp` and written to
      disk by the caller). Copied into *workdir* if it isn't already
      there (MSTM writes its output relative to CWD, so the run always
      happens in a controlled directory).
    - *inp_text*: raw .inp text (e.g. the direct return value of
      ``config_to_inp()``, not yet written to disk) -- written to
      ``<workdir>/run.inp`` before invocation.

    Parameters
    ----------
    binary_path : path-like, optional
        Explicit path to the mstm/mstm-mpi binary, bypassing PATH
        discovery (see :func:`find_mstm_binary`).
    mpi_processes : int, optional
        If set, invokes ``mpiexec -n <mpi_processes> <mstm-mpi binary>
        <inp_file>`` instead of ``<mstm binary> <inp_file>`` directly.
        Binary discovery in this mode looks for ``mstm-mpi`` (or
        ``PYMSTM_MSTM_MPI_BIN`` / the flake's ``mstm-mpi`` package), not
        ``mstm``. If None (default), runs the serial ``mstm`` binary
        directly with no MPI wrapper. MPI is a distinct argv shape (not
        just "a different binary"), so this is a separate parameter
        rather than folded into *binary_path*.
    mpiexec_path : str
        Name/path of the mpiexec launcher; only used when *mpi_processes*
        is set.
    workdir : path-like, optional
        Directory to run in. Defaults to a fresh temp dir (auto-cleaned
        unless *keep_workdir*). MSTM always writes its output relative
        to CWD, so this is also where *output_filename* ends up.
    keep_workdir : bool
        If True, do not delete the temp workdir after the run (ignored
        if *workdir* was explicitly given -- caller-supplied dirs are
        never deleted). Useful for debugging a failing run.
    timeout : float, optional
        Subprocess timeout in seconds. None disables the timeout.
    output_filename : str
        Fallback expected output file name inside *workdir*, used only
        when the .inp's own declared ``output_file`` can't be
        introspected (see :func:`_resolve_output_filename`).
    env : dict, optional
        Extra environment variables merged into the subprocess env
        (e.g. ``OMP_NUM_THREADS``). Overlays, does not replace, the
        inherited ``os.environ``.

    Returns
    -------
    MstmRunResult

    Raises
    ------
    MstmNotFoundError
        Binary (mstm, mstm-mpi, or mpiexec) not found.
    MstmExecutionError
        Non-zero exit code, timeout, or missing output file after a
        reported-successful run.
    ValueError
        If zero or more than one of inp_kwargs/inp_path/inp_text is given.
    """
    given = [
        name
        for name, val in (
            ("inp_kwargs", inp_kwargs),
            ("inp_path", inp_path),
            ("inp_text", inp_text),
        )
        if val is not None
    ]
    if len(given) != 1:
        raise ValueError(
            "Exactly one of inp_kwargs, inp_path, inp_text must be given "
            f"(got: {given or 'none'})"
        )

    owns_workdir = workdir is None
    if workdir is None:
        workdir_path = Path(tempfile.mkdtemp(prefix="pymstm_cli_"))
    else:
        workdir_path = Path(workdir)
        workdir_path.mkdir(parents=True, exist_ok=True)

    try:
        effective_output_filename = _resolve_output_filename(
            inp_kwargs=inp_kwargs,
            inp_text=inp_text,
            inp_path=inp_path,
            output_filename=output_filename,
        )

        inp_file_path = workdir_path / "run.inp"
        if inp_kwargs is not None:
            write_inp_file(inp_file_path, **inp_kwargs)
        elif inp_text is not None:
            inp_file_path.write_text(inp_text)
        else:
            src = Path(inp_path)  # type: ignore[arg-type]
            if src.resolve().parent != workdir_path.resolve():
                shutil.copy(src, inp_file_path)
            else:
                inp_file_path = src

        if mpi_processes is None:
            mstm_bin = find_mstm_binary(mpi=False, override=binary_path)
            argv = [str(mstm_bin), str(inp_file_path)]
        else:
            mstm_bin = find_mstm_binary(mpi=True, override=binary_path)
            mpiexec = _find_mpiexec(mpiexec_path)
            argv = [
                mpiexec,
                "-n",
                str(mpi_processes),
                str(mstm_bin),
                str(inp_file_path),
            ]

        run_env = {**os.environ, **(env or {})}
        try:
            proc = subprocess.run(
                argv,
                cwd=workdir_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
        except subprocess.TimeoutExpired as exc:
            raise MstmExecutionError(
                f"MSTM CLI timed out after {timeout}s: {' '.join(argv)}",
                stdout=str(exc.stdout or ""),
                stderr=str(exc.stderr or ""),
                cmd=argv,
            ) from exc

        if proc.returncode != 0:
            raise MstmExecutionError(
                f"MSTM CLI exited with code {proc.returncode}: {' '.join(argv)}\n"
                f"{proc.stderr[:2000]}",
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                cmd=argv,
            )

        output_path = workdir_path / effective_output_filename
        if not output_path.is_file():
            for cand in _FALLBACK_OUTPUT_NAMES:
                alt = workdir_path / cand
                if alt.is_file():
                    output_path = alt
                    break
            else:
                raise MstmExecutionError(
                    f"MSTM CLI reported success but no output file found at "
                    f"{output_path} (or fallback names {_FALLBACK_OUTPUT_NAMES})",
                    returncode=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    cmd=argv,
                )

        runs = parse_all_runs(output_path)
        return MstmRunResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_path=output_path,
            parsed=runs[0],
            runs=runs,
            inp_path=inp_file_path,
            workdir=workdir_path,
        )
    finally:
        if owns_workdir and not keep_workdir:
            shutil.rmtree(workdir_path, ignore_errors=True)
