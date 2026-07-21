"""
pyMSTM Compatibility Dashboard.

Dual mode: quick setup with in-app controls, or upload a ``.toml``
sweep config.  Cluster files from ``tests/data/`` are auto-discovered.
Missing references prompt the user to upload the required file.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pymstm import MstmBindings, MstmNotFoundError, find_mstm_binary, run_mstm
from pymstm._config import (
    IncidentConfig,
    MediumConfig,
    OutputConfig,
    ParticlesConfig,
    SolverConfig,
    SweepConfig,
    WavelengthsConfig,
    config_to_inp,
)
from pymstm._inp_parser import MstmInpConfig, parse_inp_text

# ---------------------------------------------------------------------------
# Session keys
# ---------------------------------------------------------------------------

_KEY_TOML_RAW = "toml_raw"
_KEY_CFG = "sweep_config"
_KEY_CLUSTER_FILES = "cluster_files"
_KEY_UPLOADS = "temp_uploads"  # dict: filename -> bytes

# ---------------------------------------------------------------------------
# Helpers: file discovery / resolution
# ---------------------------------------------------------------------------

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DATA = _PROJ_ROOT / "tests" / "data"


def _cli_available() -> bool:
    try:
        find_mstm_binary()
    except MstmNotFoundError:
        return False
    return True


def _scan_cluster_files() -> list[str]:
    """Return relative paths of all ``.dat`` files in tests/data/."""
    files: list[str] = []
    if _TESTS_DATA.is_dir():
        for p in sorted(_TESTS_DATA.glob("*.dat")):
            files.append(str(p.relative_to(_PROJ_ROOT)))
    return files


def _resolve_positions(positions_file: str) -> Path:
    """Resolve a *positions_file* string to an absolute path.

    - Absolute paths are used as-is.
    - Paths starting with ``./`` or ``../`` are resolved relative to the
      config file's directory (or project root if config was uploaded).
    - Bare filenames are looked up in *project_root / tests / data/* first,
      then relative to project root.
    """
    p = Path(positions_file)
    if p.is_absolute():
        return p
    # Try tests/data/<name> first
    candidate = _PROJ_ROOT / "tests" / "data" / p.name
    if candidate.is_file():
        return candidate
    # Fallback: relative to project root
    return (_PROJ_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Helpers: build config from UI values
# ---------------------------------------------------------------------------


def _default_sweep_config() -> SweepConfig:
    return SweepConfig(
        particles=ParticlesConfig(
            positions_file="fractal_N128_Df2.0.dat",
            scale=1.0,
            gap_factor=1.0,
            refractive_index=(1.5, 0.01),
        ),
        wavelengths=WavelengthsConfig(start=0.5, stop=1.0, num=7, scale=1e-6),
        medium=MediumConfig(),
        incident=IncidentConfig(),
        solver=SolverConfig(tolerance=1e-4, max_iterations=2000, mie_epsilon=1e-4),
        output=OutputConfig(calculate_scattering_matrix=True),
    )


def _apply_toml(raw: str) -> SweepConfig | None:
    """Parse TOML into a SweepConfig, returning None on failure."""
    try:
        data = tomllib.loads(raw)
        return SweepConfig(**data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers: execution
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _run_cli_sweep(inp_text: str) -> list[dict]:
    # MstmExecutionError/MstmNotFoundError are both RuntimeError subclasses,
    # so the caller's `except Exception` already handles them the same way
    # it handled the old ad-hoc RuntimeError this replaces.
    return run_mstm(inp_text=inp_text, timeout=1800).runs


def _run_pymstm_sweep(
    cfg: SweepConfig, cli_angles: list[float] | None, pos_path: Path
) -> list[dict]:
    positions = cfg.particles.load_positions(str(pos_path))
    n_re, n_im = cfg.particles.refractive_index
    med_re, med_im = cfg.medium.refractive_index
    sol = cfg.solver
    ls_vals = cfg.wavelengths.get_linear_length_scales()  # linear, matches CLI loop
    wl_um = cfg.wavelengths.get_wavelengths_m() * 1e6
    n_iters = len(ls_vals)

    progress = st.progress(0, "pyMSTM sweep…")
    results: list[dict] = []
    for idx, ls in enumerate(ls_vals):
        m = MstmBindings()
        m.set_length_scale(float(ls))
        m.set_spheres(
            radii=list(positions[:, 3]),
            positions=[list(p) for p in positions[:, :3]],
            orders=[
                int(
                    max(
                        4,
                        2 * np.pi * r * abs(complex(n_re, n_im)) / 0.4
                        + 4 * (2 * np.pi * r) ** (1 / 3)
                        + 2,
                    )
                )
                for r in positions[:, 3]
            ],
            ref_re=[n_re] * len(positions),
            ref_im=[n_im] * len(positions),
        )
        m.set_medium_ref(med_re, med_im)
        m.set_incident(
            cfg.incident.azimuthal_angle_deg,
            cfg.incident.polar_angle_deg,
            cfg.incident.direction,
        )
        m.set_solver_params(eps=sol.tolerance, max_iter=sol.max_iterations)
        m.set_mie_eps(sol.mie_epsilon)
        m.set_translation_eps(sol.translation_epsilon)
        m.set_max_tmatrix_order(sol.max_tmatrix_order)
        m.set_azimuthal_average(cfg.output.azimuthal_average)
        m.set_verbose(False)
        m.prepare()
        tord = m.get_tmatrix_order()
        raw = m.solve()
        sm = _pymstm_smatrix(m, cli_angles, cfg.incident.azimuthal_angle_deg)
        m.finalize()

        total = {
            "q_ext_unpol": raw["qext_tot"],
            "q_abs_unpol": raw["qabs_tot"],
            "q_sca_unpol": raw["qsca_tot"],
        }
        results.append(
            {
                "iterations": int(raw["iterations"]),
                "total": total,
                "scattering_matrix": sm,
                "wavelength_um": float(wl_um[idx]),
                "tord": int(tord),
            }
        )
        progress.progress(
            (idx + 1) / n_iters, f"pyMSTM {idx + 1}/{n_iters}  λ={wl_um[idx]:.2f}µm"
        )
    progress.empty()
    return results


def _pymstm_smatrix(
    m: MstmBindings, target_angles: list[float] | None = None, alpha_deg: float = 0.0
) -> dict | None:
    # Always loop get_scattering_angle() rather than calling
    # get_scattering_matrix() (retired from this dashboard entirely -- see
    # its docstring for the confirmed non-deterministic memory-safety
    # bug). When no target_angles are given, fall back to the CLI's own
    # default resolution (-180..180deg, 361 points at 1deg) so the two
    # code paths still produce comparably-shaped output.
    if target_angles is None:
        target_angles = list(np.linspace(-180, 180, 361))
    try:
        # With azimuthal_average enabled (the default), the CLI reports
        # theta in [0,180] only and phi is irrelevant. This phi selection
        # only matters for the (now unused) non-averaged "incident plane"
        # cut, where the CLI's table covers a full -180..180 sweep by
        # pairing theta in [0,180] with azimuth alpha for non-negative
        # angle labels and alpha+pi for negative ones (see
        # scattering_matrix_calculation in the Fortran source).
        alpha_rad = np.deg2rad(alpha_deg)
        rows = []
        for deg in target_angles:
            ct = np.cos(np.deg2rad(deg))
            phi = alpha_rad + np.pi if deg < 0 else alpha_rad
            sm = m.get_scattering_angle(costheta=ct, phi=phi)
            rows.append(sm.tolist())
        return {"angles_deg": target_angles, "matrix": rows}
    except Exception:
        return None


def _norm_smatrix(
    sm: dict, positions: np.ndarray, q_sca: float, length_scale: float = 1.0
) -> dict:
    """Normalize pyMSTM S-matrix to CLI convention.

    The CLI computes ``cross_section_radius`` on the *scaled* radii
    (which include ``length_scale_factor``).  We must do the same.
    """
    r_cs = float(np.sum(positions[:, 3] ** 3) ** (1.0 / 3.0)) * length_scale
    norm = 1.0 / (r_cs**2 * np.pi * max(abs(q_sca), 1e-12))
    return {
        "angles_deg": sm["angles_deg"],
        "matrix": [[v * norm for v in row] for row in sm["matrix"]],
    }


def _plot_sweep(cli_runs, py_runs, wl_um, elem_idx, y_label, title):
    angles = cli_runs[0]["scattering_matrix"]["angles_deg"]
    n = len(wl_um)
    colors = [f"hsl({int(260 * (1 - i / max(1, n - 1)))},80%,50%)" for i in range(n)]

    def _extract(runs, i):
        sm = runs[i].get("scattering_matrix") or {}
        mat = sm.get("matrix")
        if not mat:
            return [float("nan")] * len(angles)
        if elem_idx == 4:
            if runs is py_runs:
                return [-r[4] / max(abs(r[0]), 1e-30) for r in mat]
            return [-r[4] for r in mat]
        return [r[elem_idx] for r in mat]

    fig = go.Figure()
    for i in range(n):
        fig.add_scatter(
            x=angles,
            y=_extract(cli_runs, i),
            mode="lines",
            line=dict(color=colors[i], width=1.5),
            name=f"CLI {wl_um[i]:.1f}µm",
            legendgroup=f"w{i}",
            showlegend=(i % max(1, n // 5) == 0),
        )
        fig.add_scatter(
            x=angles,
            y=_extract(py_runs, i),
            mode="markers",
            marker=dict(color=colors[i], size=2, symbol="x"),
            name=f"py  {wl_um[i]:.1f}µm",
            legendgroup=f"w{i}",
            showlegend=(i % max(1, n // 5) == 0),
        )
    fig.update_layout(
        title=title,
        xaxis_title="θ (deg)",
        yaxis_title=y_label,
        height=500,
        hovermode="x unified",
    )
    return fig


def _single_run(inp_text: str, config: MstmInpConfig, positions: np.ndarray):
    """Run single .inp comparison (existing mode)."""

    # --- helpers for single run ---
    def _run_cli(inp_text, output_filename):
        # output_filename comes from the already-parsed MstmInpConfig, but
        # run_mstm() introspects the same value from inp_text itself, so
        # it's redundant to pass through here (kept as a parameter only to
        # avoid touching this function's call site below).
        del output_filename
        return run_mstm(inp_text=inp_text, timeout=120).parsed

    def _run_py(config):
        m = MstmBindings()
        s = config.spheres
        orders = [
            max(
                4,
                int(
                    2 * np.pi * r * abs(complex(s[i].ref_re, s[i].ref_im))
                    + 4.3 * (2 * np.pi * r) ** (1 / 3)
                    + 2
                ),
            )
            for i, r in enumerate(s.radius for s in s)
        ]
        m.set_spheres(
            radii=[s.radius for s in config.spheres],
            positions=[[s.x, s.y, s.z] for s in config.spheres],
            orders=orders,
            ref_re=[s.ref_re for s in config.spheres],
            ref_im=[s.ref_im for s in config.spheres],
        )
        m.set_medium_ref(config.medium_ref_re, config.medium_ref_im)
        m.set_incident(config.alpha_deg, config.beta_deg, config.incident_direction)
        m.set_solver_params(eps=config.solution_eps, max_iter=config.max_iterations)
        m.set_mie_eps(config.mie_eps)
        m.set_translation_eps(config.translation_eps)
        m.set_max_tmatrix_order(config.max_tmatrix_order)
        m.set_verbose(False)
        if config.number_plane_boundaries > 0:
            m.set_layers(
                config.layer_thicknesses,
                [(lr.re, lr.im) for lr in config.layer_ref_indices],
            )
        m.prepare()
        raw = m.solve()
        total = {
            "q_ext_unpol": raw["qext_tot"],
            "q_abs_unpol": raw["qabs_tot"],
            "q_sca_unpol": raw["qsca_tot"],
            "q_ext_par": raw["qext_tot"],
            "q_abs_par": raw["qabs_tot"],
            "q_sca_par": raw["qsca_tot"],
            "q_ext_perp": raw["qext_tot"],
            "q_abs_perp": raw["qabs_tot"],
            "q_sca_perp": raw["qsca_tot"],
        }
        ps = [
            {"q_ext": float(raw["q_ext"][i]), "q_abs": float(raw["q_abs"][i])}
            for i in range(config.nspheres)
        ]
        sm = _pymstm_smatrix(m)
        return m, {
            "iterations": int(raw["iterations"]),
            "solution_error": float(raw["solution_error"]),
            "total": total,
            "per_sphere": ps,
            "scattering_matrix": sm,
        }

    cli_ok = _cli_available()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("MSTM Fortran CLI")
        if cli_ok:
            try:
                cli_res = _run_cli(inp_text, config.output_file)
                st.success("Done")
                st.metric("Q_ext", f"{cli_res['total'].get('q_ext_unpol', 0):.6f}")
                st.metric("Iters", cli_res.get("iterations", "?"))
            except Exception as e:
                st.error(str(e))
                cli_res = None
        else:
            st.warning("CLI not built")
            cli_res = None
    with col_right:
        st.subheader("pyMSTM")
        try:
            mi, py_res = _run_py(config)
            st.success("Done")
            st.metric("Q_ext", f"{py_res['total'].get('q_ext_unpol', 0):.6f}")
            st.metric("Iters", py_res.get("iterations", "?"))
        except Exception as e:
            st.error(str(e))
            py_res = None

    if cli_res and py_res and cli_res.get("scattering_matrix"):
        ca = cli_res["scattering_matrix"]["angles_deg"]
        ps = _pymstm_smatrix(mi, ca, config.alpha_deg)
        if ps:
            py_res["scattering_matrix"] = ps
    if py_res:
        try:
            mi.finalize()
        except Exception:
            pass

    if not cli_res and not py_res:
        return

    st.divider()
    st.header("Comparison")

    # Total efficiencies table
    st.subheader("Total Efficiencies")
    if cli_res and py_res:
        df = pd.DataFrame(
            {
                "Quantity": ["Q_ext", "Q_abs", "Q_sca"],
                "CLI": [
                    cli_res["total"][k]
                    for k in ("q_ext_unpol", "q_abs_unpol", "q_sca_unpol")
                ],
                "pyMSTM": [
                    py_res["total"][k]
                    for k in ("q_ext_unpol", "q_abs_unpol", "q_sca_unpol")
                ],
            }
        )
        df["Rel.diff"] = (df["CLI"] - df["pyMSTM"]).abs() / df[["CLI", "pyMSTM"]].max(
            axis=1
        ).replace(0, 1)
        st.dataframe(
            df.style.format(
                {"CLI": "{:.6f}", "pyMSTM": "{:.6f}", "Rel.diff": "{:.2e}"}
            ),
            width='stretch',
        )
    elif cli_res:
        st.dataframe(pd.DataFrame([cli_res["total"]]))
    elif py_res:
        st.dataframe(pd.DataFrame([py_res["total"]]))

    # Per-sphere
    if cli_res and py_res and config.nspheres <= 20:
        st.subheader("Per-Sphere")
        rows = []
        for i in range(config.nspheres):
            cq = (
                cli_res["per_sphere"][i]["q_ext"]
                if i < len(cli_res.get("per_sphere", []))
                else float("nan")
            )
            pq = (
                py_res["per_sphere"][i]["q_ext"]
                if i < len(py_res.get("per_sphere", []))
                else float("nan")
            )
            rows.append({"Sphere": i + 1, "CLI": cq, "pyMSTM": pq})
        if rows:
            df2 = pd.DataFrame(rows)
            df2["Diff"] = (df2["CLI"] - df2["pyMSTM"]).abs()
            st.dataframe(
                df2.style.format(
                    {"CLI": "{:.6f}", "pyMSTM": "{:.6f}", "Diff": "{:.2e}"}
                ),
                width='stretch',
            )

    # S11
    st.subheader("S₁₁ vs θ")
    sc = cli_res.get("scattering_matrix") if cli_res else None
    sp = py_res.get("scattering_matrix") if py_res else None
    if sp and py_res:
        sp = _norm_smatrix(
            sp, positions, py_res["total"].get("q_sca_unpol", 1), config.length_scale
        )
    if sc and sp:
        fig = go.Figure()
        fig.add_scatter(
            x=sc["angles_deg"],
            y=[r[0] for r in sc["matrix"]],
            mode="lines",
            name="CLI S₁₁",
            line=dict(color="blue", width=2),
        )
        fig.add_scatter(
            x=sp["angles_deg"],
            y=[r[0] for r in sp["matrix"]],
            mode="markers",
            name="pyMSTM S₁₁",
            marker=dict(color="red", size=4, symbol="circle-open"),
        )
        fig.update_layout(xaxis_title="θ (deg)", yaxis_title="S₁₁", height=450)
        st.plotly_chart(fig, width='stretch')
    elif sc:
        fig = go.Figure()
        fig.add_scatter(
            x=sc["angles_deg"],
            y=[r[0] for r in sc["matrix"]],
            mode="lines",
            name="CLI S₁₁",
        )
        fig.update_layout(xaxis_title="θ (deg)", yaxis_title="S₁₁", height=450)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Enable scattering matrix in the config.")

    with st.expander("Raw"):
        ca, cb = st.columns(2)
        ca.json(cli_res)
        cb.json(py_res)


# ======================================================================
# PAGE
# ======================================================================

st.set_page_config(page_title="pyMSTM Dashboard", page_icon="🔬", layout="wide")
st.title("pyMSTM Compatibility Dashboard")

# --- init session ---
if _KEY_CFG not in st.session_state:
    st.session_state[_KEY_CFG] = _default_sweep_config()
if _KEY_CLUSTER_FILES not in st.session_state:
    st.session_state[_KEY_CLUSTER_FILES] = _scan_cluster_files()
if _KEY_UPLOADS not in st.session_state:
    st.session_state[_KEY_UPLOADS] = {}
if _KEY_TOML_RAW not in st.session_state:
    st.session_state[_KEY_TOML_RAW] = ""

cfg: SweepConfig = st.session_state[_KEY_CFG]

# ---- Mode: single .inp vs sweep ----
tab_config, tab_single = st.tabs(["🌊 Sweep", "📄 Single .inp"])

# ======================================================================
# SWEEP TAB
# ======================================================================
with tab_config:
    c1, c2 = st.columns([2, 1])
    with c2:
        toml_file = st.file_uploader(
            "Load .toml config",
            type=["toml"],
            key="toml_up",
            help="Upload a TOML sweep config to populate all fields.",
        )
        if toml_file is not None:
            raw = toml_file.read().decode()
            if raw != st.session_state[_KEY_TOML_RAW]:
                new_cfg = _apply_toml(raw)
                if new_cfg is not None:
                    st.session_state[_KEY_CFG] = new_cfg
                    cfg = new_cfg
                    st.session_state[_KEY_TOML_RAW] = raw
                    st.success("Loaded!")
                else:
                    st.error("Invalid TOML")

    with c1:
        st.subheader("Cluster")
        cluster_cols = st.columns([3, 1])
        with cluster_cols[0]:
            available = st.session_state[_KEY_CLUSTER_FILES]
            current_pos = cfg.particles.positions_file
            idx = 0
            for ii, f in enumerate(available):
                if f == current_pos or Path(f).name == current_pos:
                    idx = ii
                    break
            selected = st.selectbox(
                "Select cluster file",
                available + ["📁 Upload custom…"],
                index=min(idx, len(available)),
                key="cluster_select",
            )
            if selected == "📁 Upload custom…":
                up_file = st.file_uploader(
                    "Upload cluster (.dat, .csv, .txt)",
                    type=["dat", "csv", "txt", "pos"],
                    key="cluster_upload",
                )
                if up_file is not None:
                    data = up_file.read()
                    fname = up_file.name.replace(" ", "_")
                    st.session_state[_KEY_UPLOADS][fname] = data
                    cfg.particles.positions_file = fname
            else:
                cfg.particles.positions_file = selected

        st.subheader("Quick Settings")
        q1, q2, q3 = st.columns(3)
        with q1:
            cfg.particles.scale = st.number_input(
                "Scale",
                value=cfg.particles.scale,
                format="%.1e",
                help="Physical unit of cluster coordinates",
                key="cfg_scale",
            )
            cfg.particles.gap_factor = st.number_input(
                "Gap factor",
                value=cfg.particles.gap_factor,
                min_value=1.0,
                step=0.5,
                key="cfg_gap",
            )
        with q2:
            wl_m = cfg.wavelengths.get_wavelengths_m() * 1e6
            wl_start = st.number_input(
                "λ start (µm)", value=float(wl_m[0]), step=0.1, key="wl_start"
            )
            wl_stop = st.number_input(
                "λ stop (µm)", value=float(wl_m[-1]), step=0.1, key="wl_stop"
            )
            n_steps = len(wl_m)
            wl_num = st.number_input(
                "Steps", value=n_steps, min_value=1, max_value=100, key="wl_num"
            )
            cfg.wavelengths = WavelengthsConfig(
                start=wl_start, stop=wl_stop, num=wl_num, scale=1e-6
            )
        with q3:
            cfg.particles.refractive_index = (
                st.number_input(
                    "n (real)",
                    value=cfg.particles.refractive_index[0],
                    step=0.1,
                    key="nre",
                ),
                st.number_input(
                    "n (imag)",
                    value=cfg.particles.refractive_index[1],
                    step=0.001,
                    format="%.3f",
                    key="nim",
                ),
            )
            cfg.medium.refractive_index = (
                st.number_input(
                    "Medium n (real)",
                    value=cfg.medium.refractive_index[0],
                    step=0.1,
                    key="mre",
                ),
                st.number_input(
                    "Medium n (imag)",
                    value=cfg.medium.refractive_index[1],
                    step=0.001,
                    format="%.3f",
                    key="mim",
                ),
            )

        with st.expander("Advanced Solver Settings"):
            s1, s2, s3 = st.columns(3)
            cfg.solver.tolerance = s1.number_input(
                "Tolerance", value=cfg.solver.tolerance, format="%.0e", key="sol_tol"
            )
            cfg.solver.max_iterations = s2.number_input(
                "Max iterations", value=cfg.solver.max_iterations, key="sol_maxit"
            )
            cfg.solver.mie_epsilon = s3.number_input(
                "Mie ε", value=cfg.solver.mie_epsilon, format="%.0e", key="sol_mie"
            )
            cfg.solver.translation_epsilon = s1.number_input(
                "Translation ε",
                value=cfg.solver.translation_epsilon,
                format="%.0e",
                key="sol_tran",
            )
            cfg.solver.max_tmatrix_order = s2.number_input(
                "Max T-matrix order", value=cfg.solver.max_tmatrix_order, key="sol_tord"
            )
            cfg.output.calculate_scattering_matrix = st.checkbox(
                "Calculate scattering matrix",
                value=cfg.output.calculate_scattering_matrix,
                key="calc_sm",
            )

        # Resolve positions file
        resolved_path = _resolve_positions(cfg.particles.positions_file)
        file_exists = resolved_path.is_file()

        if not file_exists:
            st.warning(
                f"Cluster file `{cfg.particles.positions_file}` not found. Upload it below:"
            )
            up_dat = st.file_uploader(
                "Upload cluster data",
                type=["dat", "csv", "txt", "pos"],
                key="fallback_upload",
            )
            if up_dat is not None:
                fname = up_dat.name.replace(" ", "_")
                st.session_state[_KEY_UPLOADS][fname] = up_dat.read()
                cfg.particles.positions_file = fname
                file_exists = True

    # --- Config summary ---
    # Inline (not st.sidebar) -- a sidebar is a single global container
    # regardless of which tab is active, so a per-tab summary placed
    # there stays visible (and stacks with the other tab's own sidebar
    # content) no matter which tab you're actually looking at. An inline
    # row above the Run button stays correctly scoped to this tab.
    st.subheader("Config Summary")
    wl = cfg.wavelengths.get_wavelengths_m() * 1e6
    n = len(wl)
    n_re, n_im = cfg.particles.refractive_index
    mr, mi = cfg.medium.refractive_index
    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    sc1.metric("Cluster", f"{cfg.particles.positions_file}")
    sc2.metric("Wavelengths", f"{wl[0]:.1f} – {wl[-1]:.1f} µm  ({n} steps)")
    sc3.metric("n_particle", f"{n_re:.2f} + {n_im:.3f}i")
    sc4.metric("n_medium", f"{mr:.2f} + {mi:.3f}i")
    sc5.metric(
        "Solver", f"ε={cfg.solver.tolerance:.0e}  maxit={cfg.solver.max_iterations}"
    )
    sc6.metric(
        "Scattering matrix", "✓" if cfg.output.calculate_scattering_matrix else "✗"
    )

    # --- Run ---
    if not st.button(
        "▶ Run Sweep", type="primary", width='stretch', key="run_sweep"
    ):
        st.info("Configure and click **Run Sweep**.")
        st.stop()

    # Resolve cluster file (with uploaded files)
    cls_path = _resolve_positions(cfg.particles.positions_file)
    cls_data: bytes | None = None
    if cls_path.is_file():
        cls_data = cls_path.read_bytes()
    elif cfg.particles.positions_file in st.session_state[_KEY_UPLOADS]:
        cls_data = st.session_state[_KEY_UPLOADS][cfg.particles.positions_file]

    if cls_data is None:
        st.error(f"Cannot find cluster file: `{cfg.particles.positions_file}`")
        st.stop()

    # Write cluster data to temp file for CLI
    tmp_dir = tempfile.mkdtemp(prefix="pymstm_")
    cls_tmp = os.path.join(tmp_dir, "cluster.dat")
    with open(cls_tmp, "wb") as f:
        f.write(cls_data)
    # Override positions_file to absolute temp path
    cfg.particles.positions_file = cls_tmp

    # Load for pyMSTM  (gap_factor already applied in load_positions)
    positions = cfg.particles.load_positions(str(Path(cls_tmp).parent))

    # Generate inp for CLI
    inp_text = config_to_inp(cfg, output_filename="mstm_output.dat")

    cli_ok = _cli_available()
    cli_runs: list[dict] = []

    if cli_ok:
        with st.spinner("Running MSTM CLI sweep…"):
            try:
                cli_runs = _run_cli_sweep(inp_text)
                st.success(f"CLI: {len(cli_runs)} runs completed")
            except Exception as exc:
                st.error(f"CLI failed: {exc}")
    else:
        st.warning("CLI not built. Only pyMSTM results shown.")

    cli_angles = cli_runs[0]["scattering_matrix"]["angles_deg"] if cli_runs else None

    with st.spinner("Running pyMSTM sweep…"):
        py_runs = _run_pymstm_sweep(cfg, cli_angles, Path(cls_tmp))
    st.success(f"pyMSTM: {len(py_runs)} runs completed")

    # Normalize pyMSTM S-matrices (per-wavelength r_cs scaling)
    ls_vals = cfg.wavelengths.get_linear_length_scales()
    for i, py in enumerate(py_runs):
        sm = py.get("scattering_matrix")
        if sm:
            py_runs[i]["scattering_matrix"] = _norm_smatrix(
                sm, positions, py["total"].get("q_sca_unpol", 1), float(ls_vals[i])
            )

    wl_um = cfg.wavelengths.get_wavelengths_m() * 1e6

    st.divider()
    st.header("Comparison")

    if cli_runs:
        colA, colB = st.columns(2)
        with colA:
            fig_phase = _plot_sweep(
                cli_runs,
                py_runs,
                wl_um,
                elem_idx=0,
                y_label="Phase function S₁₁",
                title="Phase Function",
            )
            st.plotly_chart(fig_phase, width='stretch')
        with colB:
            fig_dlp = _plot_sweep(
                cli_runs,
                py_runs,
                wl_um,
                elem_idx=4,
                y_label="DLP = −S₁₂",
                title="Degree of Linear Polarization",
            )
            st.plotly_chart(fig_dlp, width='stretch')

        # Q_ext vs λ
        fig_eff = go.Figure()
        fig_eff.add_scatter(
            x=wl_um,
            y=[r["total"]["q_ext_unpol"] for r in cli_runs],
            mode="lines+markers",
            name="CLI Q_ext",
            line=dict(color="blue"),
        )
        fig_eff.add_scatter(
            x=wl_um,
            y=[r["total"]["q_ext_unpol"] for r in py_runs],
            mode="markers",
            name="pyMSTM Q_ext",
            marker=dict(color="red", symbol="x", size=8),
        )
        fig_eff.update_layout(
            title="Q_ext vs Wavelength",
            xaxis_title="λ (µm)",
            yaxis_title="Q_ext",
            height=350,
            hovermode="x unified",
        )
        st.plotly_chart(fig_eff, width='stretch')
    else:
        # Only pyMSTM
        angles = py_runs[0]["scattering_matrix"]["angles_deg"]
        n = len(wl_um)
        colors = [
            f"hsl({int(260 * (1 - i / max(1, n - 1)))},80%,50%)" for i in range(n)
        ]
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure()
            for i, py in enumerate(py_runs):
                sm = py.get("scattering_matrix") or {}
                if sm.get("matrix"):
                    fig.add_scatter(
                        x=angles,
                        y=[r[0] for r in sm["matrix"]],
                        mode="lines",
                        line=dict(color=colors[i]),
                        name=f"{wl_um[i]:.1f}µm",
                    )
            fig.update_layout(
                title="Phase Function S₁₁",
                height=500,
                xaxis_title="θ (deg)",
                yaxis_title="S₁₁",
            )
            st.plotly_chart(fig, width='stretch')
        with cb:
            fig2 = go.Figure()
            for i, py in enumerate(py_runs):
                sm = py.get("scattering_matrix") or {}
                if sm.get("matrix"):
                    fig2.add_scatter(
                        x=angles,
                        y=[-r[4] / max(abs(r[0]), 1e-30) for r in sm["matrix"]],
                        mode="lines",
                        line=dict(color=colors[i]),
                        name=f"{wl_um[i]:.1f}µm",
                    )
            fig2.update_layout(
                title="DLP = −S₁₂",
                height=500,
                xaxis_title="θ (deg)",
                yaxis_title="−S₁₂",
            )
            st.plotly_chart(fig2, width='stretch')

        fig_eff = go.Figure()
        fig_eff.add_scatter(
            x=wl_um,
            y=[r["total"]["q_ext_unpol"] for r in py_runs],
            mode="lines+markers",
            name="pyMSTM Q_ext",
        )
        fig_eff.update_layout(title="Q_ext vs λ", xaxis_title="λ (µm)", height=350)
        st.plotly_chart(fig_eff, width='stretch')

    # Cleanup temp
    import shutil

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Diagnostics
    with st.expander("Diagnostics", expanded=False):
        ls_vals = cfg.wavelengths.get_linear_length_scales()
        wl_um = cfg.wavelengths.get_wavelengths_m() * 1e6
        rows = []
        for i, (cli, py) in enumerate(zip(cli_runs, py_runs)):
            cq = cli["total"]
            pq = py["total"]
            dq = (
                abs(cq["q_ext_unpol"] - pq["q_ext_unpol"])
                / max(abs(cq["q_ext_unpol"]), 1e-12)
                * 100
            )
            py_tord = py.get("tord", "?")
            rows.append(
                {
                    "λ (µm)": round(wl_um[i], 2),
                    "ls": round(float(ls_vals[i]), 4),
                    "CLI Q_ext": round(cq["q_ext_unpol"], 4),
                    "py Q_ext": round(pq["q_ext_unpol"], 4),
                    "ΔQ%": round(dq, 2),
                    "CLI Q_abs": round(cq.get("q_abs_unpol", 0), 4),
                    "py Q_abs": round(pq.get("q_abs_unpol", 0), 4),
                    "CLI Q_sca": round(cq.get("q_sca_unpol", 0), 4),
                    "py Q_sca": round(pq.get("q_sca_unpol", 0), 4),
                    "py tord": py_tord,
                    "CLI iters": cli.get("iterations", "?"),
                    "py iters": py.get("iterations", "?"),
                }
            )
        st.dataframe(pd.DataFrame(rows), width='stretch')

        min_tord = min((py.get("tord", 999) for py in py_runs), default=999)
        if min_tord < 10:
            st.warning(
                f"Low Mie order (tord={min_tord}). "
                "Angular distributions (S11, DLP) are approximate. "
                "Q_ext should match within ~5%."
            )

    st.stop()

# ======================================================================
# SINGLE .INP TAB
# ======================================================================
with tab_single:
    st.markdown("Upload a ``.inp`` file for single-wavelength comparison.")
    uploaded = st.file_uploader("Upload .inp", type=["inp", "txt"], key="inp_upload")
    if uploaded is None:
        st.info("Upload an ``.inp`` file.")
        st.stop()
    try:
        inp_text = uploaded.read().decode("utf-8")
        config = parse_inp_text(inp_text)
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    st.subheader("Config")
    ic1, ic2, ic3, ic4, ic5 = st.columns(5)
    ic1.metric("Spheres", config.nspheres)
    ic2.metric("Medium", f"{config.medium_ref_re:.3f}+{config.medium_ref_im:.3f}i")
    ic3.metric("Solver ε", f"{config.solution_eps:.1e}")
    ic4.metric("Max iters", config.max_iterations)
    ic5.metric("SM", "✓" if config.calculate_scattering_matrix else "✗")

    if st.button(
        "▶ Run Comparison", type="primary", width='stretch', key="run_single"
    ):
        positions = np.array([[s.x, s.y, s.z, s.radius] for s in config.spheres])
        _single_run(inp_text, config, positions)
