"""
Parser for MSTM .out (output) files.

Extracts comparable numerical values from the tagged text output
produced by the standalone MSTM CLI.
"""

from __future__ import annotations

import os
import re
from typing import Any


def parse_mstm_output(filepath: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse an MSTM output file and return a dict of key numerical results.

    Parsed keys
    -----------
    iterations : int
        Number of solver iterations.
    solution_error : float
        Final solution residual.
    solution_time : float
        Wall-clock solution time.
    tmatrix_order : int or None
        T-matrix truncation order (only for random-orientation runs).
    per_sphere : list of dict
        Per-sphere ``{q_ext, q_abs, q_vabs}`` (only if ``print_sphere_data``
        was enabled in the input file).
    total : dict
        Total efficiencies with keys ``q_ext_unpol, q_abs_unpol, q_sca_unpol,
        q_ext_par, q_abs_par, q_sca_par, q_ext_perp, q_abs_perp, q_sca_perp``.
    scattering_matrix : dict or None
        ``{angles_deg, matrix}`` where *angles_deg* is a list of scattering
        angles in degrees and *matrix* is a nested list of shape
        ``(n_angles, 16)`` with the 16 Mueller matrix elements.
        Present only when ``calculate_scattering_matrix`` was enabled.
    """
    text = _slurp(filepath)

    result: dict[str, Any] = {}

    result.update(_parse_solver_stats(text))
    result["per_sphere"] = _parse_per_sphere(text)
    result["total"] = _parse_total_efficiencies(text)
    result["tmatrix_order"] = _parse_tmatrix_order(text)

    sm = _parse_scattering_matrix(text)
    result["scattering_matrix"] = sm

    return result


def parse_all_runs(filepath: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Parse every run in a multi-run MSTM output file.

    Returns a list of per-run dicts with the same keys as
    :func:`parse_mstm_output`.  Suitable for ``loop_variable`` sweeps.
    """
    text = _slurp(filepath)
    parts = re.split(r"^\s*calculation results for run\s*\n", text, flags=re.M)
    if len(parts) <= 1:
        return [parse_mstm_output(filepath)]

    header = parts[0]
    runs: list[dict[str, Any]] = []
    import tempfile

    for chunk in parts[1:]:
        block = header + "\n calculation results for run \n" + chunk
        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as tf:
            tf.write(block)
            tpath = tf.name
        try:
            runs.append(parse_mstm_output(tpath))
        finally:
            os.unlink(tpath)
    return runs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?"


def _slurp(path: str | os.PathLike[str]) -> str:
    with open(path) as f:
        return f.read()


def _parse_solver_stats(text: str) -> dict[str, Any]:
    """Extract 'number iterations, error, solution time' line."""
    pat = re.compile(
        r"number\s+iterations,\s+error,\s+solution\s+time\s*\n"
        rf"\s*(\d+)\s+({_FLOAT_RE})\s+({_FLOAT_RE})"
    )
    m = pat.search(text)
    if not m:
        return {"iterations": 0, "solution_error": 0.0, "solution_time": 0.0}
    return {
        "iterations": int(m.group(1)),
        "solution_error": float(m.group(2).replace("D", "E").replace("d", "E")),
        "solution_time": float(m.group(3).replace("D", "E").replace("d", "E")),
    }


def _parse_per_sphere(text: str) -> list[dict[str, float]]:
    """Extract per-sphere Q_ext, Q_abs, Q_vabs lines."""
    header_pat = re.compile(
        r"sphere\s+extinction,\s+absorption,\s+volume\s+absorption\s+efficiencies.*\n"
        r"\s+sphere\s+Qext\s+Qabs\s+Qvabs\s*\n"
    )
    m = header_pat.search(text)
    if not m:
        return []

    start = m.end()
    lines = text[start:].splitlines()
    results: list[dict[str, float]] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            break
        if not parts[0].lstrip("-").isdigit():
            break
        q_ext = float(parts[1].replace("D", "E").replace("d", "E"))
        q_abs = float(parts[2].replace("D", "E").replace("d", "E"))
        q_vabs = float(parts[3].replace("D", "E").replace("d", "E"))
        results.append({"q_ext": q_ext, "q_abs": q_abs, "q_vabs": q_vabs})

    return results


def _parse_total_efficiencies(text: str) -> dict[str, float]:
    """Extract the 9-value total efficiencies line."""
    pat = re.compile(
        r"total\s+extinction,\s+absorption,\s+scattering\s+efficiencies.*\n"
        rf"\s*((?:{_FLOAT_RE}\s+){{8}}{_FLOAT_RE})"
    )
    m = pat.search(text)
    if not m:
        return {}

    vals = [float(v.replace("D", "E").replace("d", "E")) for v in m.group(1).split()]
    keys = [
        "q_ext_unpol",
        "q_abs_unpol",
        "q_sca_unpol",
        "q_ext_par",
        "q_abs_par",
        "q_sca_par",
        "q_ext_perp",
        "q_abs_perp",
        "q_sca_perp",
    ]
    return dict(zip(keys, vals))


def _parse_tmatrix_order(text: str) -> int | None:
    """Extract T-matrix order if present (random-orientation runs)."""
    m = re.search(r"calculated t matrix order:\s*\n\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None


# Column-major flat index (0-based) of each Mueller element label, matching
# the Fortran ``smlabel`` array order used by ``print_scat_mat_header``.
_SM_LABELS = [
    "11", "21", "31", "41", "12", "22", "32", "42",
    "13", "23", "33", "43", "14", "24", "34", "44",
]
_SM_LABEL_TO_INDEX = {label: i for i, label in enumerate(_SM_LABELS)}

# Matches both the full incident-plane cut and the azimuthally-averaged
# ("azimuthal_average=true") scattering matrix table.
_SM_HEADER = re.compile(
    r"(?:scattering matrix in incident plane.*|azimuthal averaged scattering matrix)\n"
)


def _parse_scattering_matrix(text: str) -> dict[str, Any] | None:
    """Extract scattering matrix from the output.

    Handles both the full 16-element "incident plane" table and the
    reduced 6-element table produced when ``azimuthal_average=true``
    (columns ``11,12,22,33,34,44`` -- the only non-zero elements once a
    scattering matrix has been averaged over the full azimuth). Rows are
    always expanded to 16 elements in the standard column-major order
    (missing elements, which are analytically zero when averaged, are
    filled with 0.0).
    """
    m = _SM_HEADER.search(text)
    if not m:
        return None

    tail = text[m.end() :]
    lines = tail.splitlines()

    # The column-label line (e.g. "theta   11   21   31 ...") tells us
    # exactly which of the 16 elements are present and in what order.
    labels: list[str] | None = None
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "theta":
            labels = parts[1:]
            break
    if not labels:
        return None
    label_indices = [_SM_LABEL_TO_INDEX[lbl] for lbl in labels]
    ncols = len(labels)

    rows: list[list[float]] = []
    col_idx: list[float] = []

    for line in lines:
        parts = line.split()
        if len(parts) != ncols + 1:
            # Skip metadata lines
            if parts and (parts[0].startswith("number") or parts[0] == "theta"):
                continue
            if rows:
                break
            continue

        try:
            idx = float(parts[0].replace("D", "E").replace("d", "E"))
        except ValueError:
            if rows:
                break
            continue

        try:
            raw_vals = [
                float(v.replace("D", "E").replace("d", "E")) for v in parts[1:]
            ]
        except ValueError:
            if rows:
                break
            continue

        full_row = [0.0] * 16
        for pos, val in zip(label_indices, raw_vals):
            full_row[pos] = val

        col_idx.append(idx)
        rows.append(full_row)

    if not rows:
        return None

    n = len(rows)
    if abs(col_idx[0]) > n * 2:
        angles_deg = [i * 180.0 / max(n - 1, 1) for i in range(n)]
    else:
        angles_deg = col_idx

    return {
        "angles_deg": angles_deg,
        "matrix": rows,
    }
