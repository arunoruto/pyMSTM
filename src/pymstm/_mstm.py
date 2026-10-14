"""
Python bindings for the MSTM (Multiple Sphere T-Matrix) Fortran library.

Uses f2py to call into the compiled extension built from the legacy MSTM
Fortran sources plus src/pymstm/_fortran/mstm_f2py.f90 (see the Makefile's
f2py-ext target). Most simple state (solver tolerances, T-matrix order,
etc.) is read/written directly as attributes on Fortran modules exposed by
f2py; only subroutines with real orchestration logic (allocation, defaults
fan-out, multi-step calculations) go through dedicated wrapper calls.

Usage:
    import pymstm

    # Create a solver instance
    m = pymstm.MstmBindings()

    # Set up a cluster of spheres
    radii = [0.5, 0.5]
    positions = [[0., 0., 0.], [0., 0., 2.]]
    orders = [5, 5]
    ref_re = [1.5, 1.5]
    ref_im = [0.01, 0.01]

    m.set_spheres(radii, positions, orders, ref_re, ref_im)
    m.set_incident(alpha_deg=0.0, beta_deg=0.0)
    m.set_medium_ref(1.0, 0.0)
    m.set_solver_params(eps=1e-6, max_iter=5000)

    m.prepare()
    result = m.solve()
    print("Q_ext:", result['q_ext'])
    print("Q_abs:", result['q_abs'])
    print("Q_sca:", result['q_sca'])
    print("Total Q_ext:", result['qext_tot'])
    print("Total Q_abs:", result['qabs_tot'])
    print("Total Q_sca:", result['qsca_tot'])
"""

import os
import shutil
import tempfile
import threading

import numpy as np

from . import _mstm_ext as _ext

# All MSTM state lives in Fortran module variables, so the extension is not
# reentrant: one set->prepare->solve sequence per process at a time. The
# long Fortran calls release the GIL (``!f2py threadsafe``), so without this
# lock two threads would interleave on that shared state. The high-level
# entry points (MSTM.solve(), solve_task()) hold it for the whole sequence;
# direct MstmBindings users sharing a process across threads must do the
# same (``with MSTM_LOCK: ...``). Reentrant so those entry points can
# construct/finalize MstmBindings inside it.
# ponytail: one global lock = one solve per process; run more processes
# (Dask workers with nthreads=1) for parallelism.
MSTM_LOCK = threading.RLock()


def get_tmatrix_size(tmatrix_order):
    """Number of complex T-matrix entries for a given truncation order.

    Pure Python -- has no Fortran-side dependency beyond the order itself
    (formerly mstm_get_tmatrix_size_c).
    """
    return sum(
        2 * (2 * l + 1) * 2 * l * (l + 2) for l in range(1, tmatrix_order + 1)
    )


class MstmBindings:
    """Direct f2py bindings to the MSTM T-Matrix solver -- a low-level,
    stateful, builder-pattern interface (set_spheres/set_incident/.../
    prepare()/solve()/finalize()) giving access to every individual MSTM
    capability. For a higher-level, pydantic-validated interface that
    also supports the standalone CLI binary as an alternate backend, see
    :class:`pymstm.MSTM`.

    Only one instance can be used at a time (the most recently created
    one): MSTM's state is process-global. Threads sharing a process must
    hold :data:`MSTM_LOCK` around a whole set->prepare->solve sequence.

    Parameters
    ----------
    cross_section_convention : {"physical", "mstm"}
        Normalisation of the efficiencies returned by :meth:`solve`,
        :meth:`compute_tmatrix` and :meth:`get_mie_efficiencies`. They
        differ only when the medium is not vacuum.

        - ``"physical"`` (default): cross sections per incident irradiance
          in the medium, e.g. a single sphere gives the Lorenz-Mie
          efficiencies in the host (``Q_ext -> 2`` for large spheres).
        - ``"mstm"``: MSTM's own numbers, as the MSTM CLI prints them.
          With a medium refractive index ``n`` (``set_medium_ref``, or
          layer 0 of ``set_layers``) MSTM normalises by the irradiance of
          the same field amplitude in vacuum, so its extinction,
          absorption and scattering efficiencies (totals and per sphere,
          and with a lattice the unit-cell values) are ``Re(n)`` times the
          physical ones; :meth:`get_mie_efficiencies` are ``Re(n)**2``
          times. ``"physical"`` divides by these factors, using ``Re(n)``
          of layer 0 as MSTM itself does when it normalises the
          scattering matrix.

        The scattering matrix (:meth:`get_scattering_angle`) is MSTM's
        either way; see its Notes.
    """

    # MSTM keeps all state in process-global Fortran module variables, so
    # only one instance can own it: the most recently created one. Older
    # instances raise instead of silently using (or freeing) its arrays.
    _owner = None

    def __init__(self, cross_section_convention="physical"):
        if cross_section_convention not in ("physical", "mstm"):
            raise ValueError(
                "cross_section_convention must be 'physical' or 'mstm', "
                f"got {cross_section_convention!r}"
            )
        self.cross_section_convention = cross_section_convention
        self._nspheres = 0
        self._prepared = False
        self._solved = False
        self._token = object()
        with MSTM_LOCK:  # never reset defaults under a running solve
            MstmBindings._owner = self._token
            _ext.mstm_f2py_bindings.mstm_init()

    @property
    def _ext(self):
        if MstmBindings._owner is not self._token:
            raise RuntimeError(
                "this MstmBindings instance was finalized or superseded by a "
                "newer one: MSTM's state is process-global, so only the most "
                "recently created MstmBindings can be used"
            )
        return _ext

    def _efficiency_factor(self):
        """What MSTM's efficiencies are divided by under the selected
        convention: Re(n) of the incident medium (layer 0), or 1."""
        if self.cross_section_convention == "mstm":
            return 1.0
        return float(self._ext.surface_subroutines.layer_ref_index[0].real)

    def set_spheres(self, radii, positions, orders, ref_re, ref_im):
        """Set sphere data for the cluster.

        Parameters
        ----------
        radii : array-like, shape (n,)
            Sphere radii in arbitrary units (e.g., size parameter x=2*pi*r/lambda).
        positions : array-like, shape (n, 3)
            Sphere center positions (x, y, z).
        orders : array-like, shape (n,)
            Advisory per-sphere Mie expansion order used only to size the
            internal allocation -- it does *not* govern the truncation of
            the result. The actual per-sphere truncation is controlled by
            ``mie_epsilon`` (``set_mie_eps``): a positive value is an
            adaptive convergence tolerance, a negative value a fixed number
            of orders. (Confirmed empirically: at a fixed ``mie_epsilon``,
            the cross sections are identical whether ``orders`` is 4 or 12.)
            ``max(4, int(x + 4*x**(1/3) + 2))`` where x is the size
            parameter is a reasonable value to pass.
        ref_re : array-like, shape (n,)
            Real part of refractive index for each sphere.
        ref_im : array-like, shape (n,)
            Imaginary part of refractive index for each sphere.
        """
        radii = np.asarray(radii, dtype=np.float64)
        pos = np.asarray(positions, dtype=np.float64)
        orders = np.asarray(orders, dtype=np.int32)
        ref_re = np.asarray(ref_re, dtype=np.float64)
        ref_im = np.asarray(ref_im, dtype=np.float64)

        n = len(radii)
        if pos.shape != (n, 3):
            raise ValueError(
                f"positions must have shape ({n}, 3), got {pos.shape}"
            )
        if len(orders) != n:
            raise ValueError(f"orders must have length {n}")
        if len(ref_re) != n or len(ref_im) != n:
            raise ValueError(f"ref_re and ref_im must have length {n}")

        self._nspheres = n
        # mstm_set_spheres expects pos as (3, n) (Fortran-natural layout).
        self._ext.mstm_f2py_bindings.mstm_set_spheres(
            orders, radii, pos.T, ref_re, ref_im
        )
        self._prepared = False
        self._solved = False

    def set_medium_ref(self, ref_re, ref_im=0.0):
        """Set the surrounding medium refractive index (default: vacuum, 1+0i).

        Sphere refractive indices stay absolute, and lengths stay in units
        of ``1 / k0`` (vacuum wavenumber) after ``set_length_scale``. The
        equivalent "relative" setup is a vacuum medium with sphere indices
        ``m / n`` and lengths scaled by ``n``; for a non-absorbing medium
        it gives the same efficiencies as this one under
        ``cross_section_convention="physical"``.
        """
        self._ext.mstm_f2py_bindings.mstm_set_medium_ref(ref_re, ref_im)

    def set_incident(self, alpha_deg=0.0, beta_deg=0.0):
        """Set the incident plane wave direction.

        Parameters
        ----------
        alpha_deg : float
            Azimuthal angle in degrees (rotation about y-axis before beta).
        beta_deg : float
            Polar angle in degrees (0 = forward along +z, 90 = perpendicular).
            Above 90 the wave comes from the far side (+z towards -z), as in
            the MSTM CLI, which derives its ``incident_direction`` from
            ``incident_beta_deg`` the same way (in ``prepare()``, once the
            medium and layer refractive indices are known).
        """
        self._ext.mstm_f2py_bindings.mstm_set_incident(alpha_deg, beta_deg)

    def set_solver_params(self, eps=1e-6, max_iter=5000, method="iterative"):
        """Set solver parameters.

        Parameters
        ----------
        eps : float
            Solution convergence tolerance.
        max_iter : int
            Maximum number of BiCG iterations.
        method : str
            'iterative' (BiCG) or 'direct' (LU decomposition).
        """
        method_code = 0 if method == "iterative" else 1
        self._ext.mstm_f2py_bindings.mstm_set_solver_params(eps, max_iter, method_code)

    def set_mie_eps(self, eps=1e-6):
        """Set Mie coefficient convergence tolerance."""
        self._ext.inputinterface.mie_epsilon = eps

    def set_length_scale(self, scale=1.0):
        """Set length scale factor applied to all positions and radii."""
        self._ext.inputinterface.length_scale_factor = scale

    def set_verbose(self, verbose=True):
        """Enable or disable diagnostic output."""
        self._ext.mstm_f2py_bindings.mstm_set_verbose(1 if verbose else 0)

    def set_scattering_map(self, half_range=15, angle_min=0.0, angle_max=180.0):
        """Configure scattering matrix output.

        Parameters
        ----------
        half_range : int
            Half-range of scattering angles. Total number of angles = 2*half_range + 1.
        angle_min : float
            Minimum scattering angle in degrees.
        angle_max : float
            Maximum scattering angle in degrees.
        """
        self._ext.mstm_f2py_bindings.mstm_set_scattering_map(
            half_range, angle_min, angle_max
        )

    def set_azimuthal_average(self, enabled=True):
        """Average the scattering matrix analytically over azimuth.

        When enabled, S(theta) is single-valued over theta in [0,180],
        independent of azimuth phi -- this is the conventional "phase
        function" for a non-spherically-symmetric target, as opposed to
        a single incident-plane cut. Must be called before ``prepare()``.
        """
        self._ext.inputinterface.azimuthal_average = bool(enabled)
        self._ext.inputinterface.numerical_azimuthal_average = bool(enabled)

    def set_excitation_switch(self, excited):
        """Set which spheres are excited by the incident field.

        Parameters
        ----------
        excited : array-like of bool or int, shape (n,)
            True/1 means sphere is excited, False/0 means not.
        """
        exc = np.asarray(excited, dtype=np.int32)
        if len(exc) != self._nspheres:
            raise ValueError(
                f"excited must have length {self._nspheres} (number of spheres)"
            )
        self._ext.mstm_f2py_bindings.mstm_set_excitation_switch(exc, n=self._nspheres)

    def prepare(self):
        """Prepare the calculation: compute host spheres, Mie coefficients,
        translation orders, and allocate result arrays.

        Must be called after set_spheres() and before solve().
        """
        if self._nspheres == 0:
            raise RuntimeError("No spheres set. Call set_spheres() first.")
        self._ext.mstm_f2py_bindings.mstm_prepare()
        self._prepared = True

    def solve(self):
        """Run the fixed-orientation T-matrix calculation.

        Efficiencies are cross sections divided by ``pi * r**2``, with ``r``
        the volume-equivalent radius of the external spheres for the totals
        (:meth:`get_cross_section_radius`), each sphere's radius per sphere,
        and ``sqrt(cell area / pi)`` (projected on the incident direction)
        with a periodic lattice, where ``qabs_tot`` is the unit-cell
        absorptance. Normalised per ``cross_section_convention``.

        Returns
        -------
        dict with keys:
            q_ext : ndarray, shape (n,)
                Per-sphere extinction efficiency.
            q_abs : ndarray, shape (n,)
                Per-sphere absorption efficiency.
            q_sca : ndarray, shape (n,)
                Per-sphere scattering efficiency.
            qext_tot : float
                Total cluster extinction efficiency.
            qabs_tot : float
                Total cluster absorption efficiency.
            qsca_tot : float
                Total cluster scattering efficiency.
            solution_error : float
                Final solution residual.
            iterations : int
                Number of solver iterations.
            status : int
                Solution status (0 = converged, 1 = did not converge).
        """
        if not self._prepared:
            raise RuntimeError("Not prepared. Call prepare() first.")

        (q_ext, q_abs, q_sca, qext_tot, qabs_tot, qsca_tot, sol_err, niter, status) = (
            self._ext.mstm_f2py_bindings.mstm_solve(n=self._nspheres)
        )

        self._solved = True
        f = self._efficiency_factor()

        return {
            "q_ext": q_ext / f,
            "q_abs": q_abs / f,
            "q_sca": q_sca / f,
            "qext_tot": qext_tot / f,
            "qabs_tot": qabs_tot / f,
            "qsca_tot": qsca_tot / f,
            "solution_error": sol_err,
            "iterations": niter,
            "status": status,
        }

    def get_scattering_angle(self, costheta, phi=0.0):
        """Get the 4x4 Mueller scattering matrix elements at a specific angle.

        Parameters
        ----------
        costheta : float
            Cosine of the scattering angle, in the LAB frame (see Notes --
            this is NOT relative to the incident direction).
        phi : float
            Azimuthal scattering angle in radians, in the LAB frame.

        Returns
        -------
        ndarray, shape (16,)
            The 16 elements of the 4x4 scattering (Mueller) matrix,
            flattened in ROW-major order: S11, S12, S13, S14, S21, S22,
            ..., S44 (a prior version of this docstring incorrectly
            labeled this "column-major" while listing the row-major
            sequence -- the sequence was always correct, only the label
            was wrong). Note this differs from
            ``pymstm._parser.parse_mstm_output()``'s
            ``['scattering_matrix']['matrix']``, whose per-angle
            16-element rows are assembled in COLUMN-major order (S11,
            S21, S31, S41, S12, ...) to match the CLI's own printed
            column layout -- the two are not directly comparable
            index-for-index without re-permuting one to match the other.

        Notes
        -----
        **Angles are lab-frame, not incident-relative.** ``costheta``/
        ``phi`` are spherical coordinates in the lab (simulation) frame,
        matching MSTM's own incident-wave convention
        ``k_hat = Rz(alpha_deg).Ry(beta_deg).z_hat`` (see
        ``incident_field_initialization`` in the MSTM Fortran source).
        The forward-scattering peak is therefore at
        ``(theta=beta_deg, phi=alpha_deg)`` as set via ``set_incident()``,
        NOT at ``(theta=0, phi=anything)`` -- getting this wrong for a
        tilted incident direction silently produces plausible-looking but
        physically wrong values (confirmed empirically: results were off
        by more than 1000x at some angles) rather than an obvious error.
        To sweep angles relative to the incident direction instead,
        rotate the desired ``(theta_rel, phi_rel)`` into the lab frame via
        the same ``R = Rz(alpha_deg).Ry(beta_deg)`` before calling this
        method.

        **Raw S11 convention.** The returned S11 is MSTM's own (it does not
        depend on ``cross_section_convention``): ``integral(S11 dOmega)
        over 4*pi steradians == Csca``, with ``Csca = qsca * pi * r**2`` in
        MSTM's length units (in which the vacuum wavenumber is 1) and
        ``qsca`` from ``cross_section_convention="mstm"``. In vacuum this
        is the Bohren-Huffman convention ``dCsca/dOmega = S11 / k**2``; in
        a medium of real index ``n``, multiply S11 by ``n`` to get the
        Bohren-Huffman S11 for the wavenumber in the medium. It is NOT the
        radiative-transfer "phase function" convention
        (``integral(S11 dOmega) == 4*pi``); see
        :func:`pymstm._convert.s11_to_phase_function`.
        """
        if not self._solved:
            raise RuntimeError("Not solved. Call solve() first.")

        return self._ext.mstm_f2py_bindings.mstm_scattering_angle(costheta, phi)

    def get_number_of_equations(self):
        """Return the number of equations in the linear system."""
        return int(self._ext.spheredata.number_eqns)

    def get_number_of_spheres(self):
        """Return the number of spheres."""
        return int(self._ext.spheredata.number_spheres)

    def get_tmatrix_order(self):
        """Return the T-matrix expansion order."""
        return int(self._ext.spheredata.t_matrix_order)

    def get_cross_section_radius(self):
        """Return the effective cross-section radius of the cluster."""
        return float(self._ext.spheredata.cross_section_radius)

    def compute_tmatrix(self, tmatrix_file=None):
        """Compute the T-matrix for the current cluster configuration.

        Parameters
        ----------
        tmatrix_file : path-like, optional
            Where MSTM writes the T-matrix file, e.g. to pass it to
            :meth:`ranorient_smatrix` later (at most 128 characters).
            Default: a temporary file, deleted before returning.

        Returns
        -------
        dict with keys:
            tmatrix : ndarray, shape (2 * n_entries,)
                Flattened complex T-matrix entries as (real, imag) pairs.
            tmatrix_order : int
                T-matrix truncation order.
            q_ext : ndarray, shape (n,)
                Per-sphere extinction efficiency.
            q_abs : ndarray, shape (n,)
                Per-sphere absorption efficiency.
            status : int
                0 = converged, 1 = not converged.
        """
        if not self._prepared:
            raise RuntimeError("Not prepared. Call prepare() first.")

        ntot = get_tmatrix_size(self.get_tmatrix_order())
        array_len = 2 * ntot

        with tempfile.TemporaryDirectory(prefix="pymstm-") as tmp:
            path = (
                os.path.join(tmp, "tmatrix.dat")
                if tmatrix_file is None
                else os.fspath(tmatrix_file)
            )
            if len(path) > 128:
                raise ValueError(f"tmatrix_file path longer than 128 characters: {path}")
            (tdata, tord, q_ext, q_abs, status) = (
                self._ext.mstm_f2py_bindings.mstm_compute_tmatrix(
                    tmatrix_file=path, n=self._nspheres, array_len=array_len
                )
            )

        f = self._efficiency_factor()
        return {
            "tmatrix": tdata,
            "tmatrix_order": tord,
            "tmatrix_size": ntot,
            "q_ext": q_ext / f,
            "q_abs": q_abs / f,
            "status": status,
        }

    def ranorient_smatrix(self, tmatrix_file):
        """Compute random-orientation scattering matrix expansion from a T-matrix file.

        Parameters
        ----------
        tmatrix_file : path-like
            Path to a T-matrix file, e.g. written by
            ``compute_tmatrix(tmatrix_file=...)``.

        Returns
        -------
        dict with keys:
            sm_coef : ndarray, shape (16 * (2*tm_order+1),)
                Total scattering matrix GSF expansion coefficients.
            cm_coef : ndarray, shape (16 * (2*tm_order+1),)
                Coherent-field scattering matrix GSF expansion coefficients.
            tmatrix_order : int
                T-matrix order read from file.
        """
        # tmat_order sizes the output arrays; the module's own
        # t_matrix_order (set by prepare()/compute_tmatrix()) is the best
        # available upper bound before reading the file.
        tmat_order = self.get_tmatrix_order()
        path = os.fspath(tmatrix_file)

        with tempfile.TemporaryDirectory(prefix="pymstm-") as tmp:
            # MSTM reads the file name into a 30-character variable.
            if len(path) > 30:
                path = shutil.copy(path, os.path.join(tmp, "t.dat"))
                if len(path) > 30:
                    raise ValueError(
                        "MSTM needs a T-matrix file path of at most 30 characters; "
                        f"set TMPDIR to a shorter directory (got {path})"
                    )
            sm_coef, cm_coef, tord_out = (
                self._ext.mstm_f2py_bindings.mstm_ranorient_smatrix(
                    path, tmat_order=tmat_order
                )
            )

        n = 16 * (2 * tord_out + 1)
        return {
            "sm_coef": sm_coef[:n],
            "cm_coef": cm_coef[:n],
            "tmatrix_order": tord_out,
        }

    def ranorient_smatrix_at_angle(self, sm_coef, tmatrix_order, costheta):
        """Evaluate random-orientation scattering matrix at a given scattering angle.

        Parameters
        ----------
        sm_coef : ndarray
            GSF expansion coefficients from ranorient_smatrix().
        tmatrix_order : int
            T-matrix order.
        costheta : float
            Cosine of scattering angle.

        Returns
        -------
        ndarray, shape (16,)
            16 Mueller matrix elements at the given angle.
        """
        sm_coef = np.asarray(sm_coef, dtype=np.float64, order="C")
        return self._ext.mstm_f2py_bindings.mstm_ranorient_smatrix_at_angle(
            sm_coef, costheta, tmat_order_in=tmatrix_order
        )

    def set_layers(self, thicknesses, ref_indices):
        """Set up plane boundaries (layered media).

        Parameters
        ----------
        thicknesses : list of float
            Thickness of each layer (length = number of boundaries - 1).
            The first boundary is at z=0 by convention.
        ref_indices : list of (re, im) tuples
            Refractive index of each layer including the incident medium.
            ref_indices[0] = incident medium, ref_indices[1] = first layer, etc.
            Length = number of boundaries + 1.
        """
        n_layers = len(ref_indices) - 1
        self._ext.mstm_f2py_bindings.mstm_set_layer_count(n_layers)
        for i, (re, im) in enumerate(ref_indices):
            self._ext.mstm_f2py_bindings.mstm_set_layer_ref_index(
                i, float(re), float(im)
            )
        for i, t in enumerate(thicknesses):
            self._ext.mstm_f2py_bindings.mstm_set_layer_thickness(i + 1, float(t))

    def set_lattice(self, cell_width_x, cell_width_y, phase_shift=False, finite=False):
        """Enable periodic lattice mode.

        Parameters
        ----------
        cell_width_x, cell_width_y : float
            Unit cell dimensions in x and y, in the same length units as
            the sphere radii/positions (scaled by ``set_length_scale()``,
            so call that first, as for ``set_spheres()``). With a lattice,
            ``qext_tot``/``qabs_tot``/``qsca_tot`` are per unit-cell area
            (``qabs_tot`` is the CLI's unit-cell absorptance).
        phase_shift : bool
            Use phase-shifted lattice sums.
        finite : bool
            Use finite (truncated) lattice sums.
        """
        self._ext.mstm_f2py_bindings.mstm_set_lattice(
            cell_width_x,
            cell_width_y,
            1 if phase_shift else 0,
            1 if finite else 0,
        )

    def clear_lattice(self):
        """Disable periodic lattice mode."""
        self._ext.mstm_f2py_bindings.mstm_clear_lattice()

    def set_max_tmatrix_order(self, order):
        """Set the maximum T-matrix truncation order."""
        self._ext.spheredata.max_t_matrix_order = order

    def set_tmatrix_convergence_eps(self, eps):
        """Set the T-matrix convergence criterion."""
        self._ext.inputinterface.t_matrix_convergence_epsilon = eps

    def set_translation_eps(self, eps):
        """Set the translation operator convergence tolerance."""
        self._ext.inputinterface.translation_epsilon = eps

    def set_gaussian_beam(self, beam_constant):
        """Set the Gaussian beam width constant (0 = plane wave)."""
        self._ext.spheredata.gaussian_beam_constant = beam_constant

    def get_mie_efficiencies(self):
        """Get per-sphere isolated-sphere (Lorenz-Mie) efficiencies,
        computed during prepare().

        Under ``cross_section_convention="physical"`` these are the Mie
        efficiencies of each sphere alone in a non-absorbing medium (for an
        absorbing medium they are divided by ``Re(n)**2`` as well, but
        extinction in an absorbing host has no unique definition). Spheres
        inside another sphere are computed by MSTM relative to that host
        sphere's material.

        Returns
        -------
        q_ext : ndarray, shape (n,)
        q_abs : ndarray, shape (n,)
        """
        n = self._nspheres
        f2 = self._efficiency_factor() ** 2
        q_ext = np.asarray(self._ext.spheredata.qext_mie[:n]) / f2
        q_abs = np.asarray(self._ext.spheredata.qabs_mie[:n]) / f2
        return q_ext, q_abs

    def finalize(self):
        """Free all allocated memory and finalize the library.

        A no-op for an instance that no longer owns MSTM's state (see
        ``__init__``), so it never frees a newer instance's arrays.
        """
        with MSTM_LOCK:  # never free arrays under a running solve
            if MstmBindings._owner is self._token:
                _ext.mstm_f2py_bindings.mstm_finalize()
                MstmBindings._owner = None
        self._prepared = False
        self._solved = False

    def __del__(self):
        # Only the owner frees anything. GC can run this on any thread (e.g.
        # a Dask worker's event loop): never block it behind someone else's
        # solve -- if the lock is busy, skip; the next prepare() reallocates.
        if MstmBindings._owner is not getattr(self, "_token", None):
            return
        if not MSTM_LOCK.acquire(blocking=False):
            return
        try:
            self.finalize()
        except Exception:
            pass
        finally:
            MSTM_LOCK.release()
