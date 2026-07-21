"""
Scattering-matrix normalization conventions.

MSTM's raw S11 (from both ``MSTM.get_scattering_angle()`` and the CLI's
un-normalized scattering-matrix table, i.e. ``normalize_s11=False`` in
:func:`pymstm._inp.write_inp_file`) follows the Bohren-Huffman convention::

    dCsca/dOmega = S11 / k**2       (i.e. integral(S11 dOmega) == k**2 * Csca)

This is *not* the "phase function" convention used elsewhere in radiative
transfer::

    integral(p(theta, phi) dOmega) == 4*pi

These helpers convert between the two, and undo a separate, unrelated
residual factor left by the CLI's ``normalize_s11=False`` path. See
:meth:`pymstm.MSTM.get_scattering_angle`'s docstring for the underlying
empirical derivation of both conventions.
"""

from __future__ import annotations

import numpy as np


def s11_to_phase_function(s11, k, c_sca):
    """Convert raw (Bohren-Huffman-convention) S11 to the radiative-transfer
    phase function convention (``integral(p dOmega) == 4*pi``).

    Parameters
    ----------
    s11 : array-like
        Raw S11 values, e.g. from ``MSTM.get_scattering_angle()[..., 0]``
        (or the CLI's scattering-matrix table with ``normalize_s11=False``,
        after applying :func:`cli_normalized_s11_to_raw` if needed).
    k : float
        Wavenumber in the surrounding medium, ``2*pi*n_medium/wavelength``,
        in units consistent with S11's own length scale.
    c_sca : float
        Scattering cross section (same length-scale convention as *k*).

    Returns
    -------
    ndarray
        Phase function values, same shape as *s11*, satisfying
        ``integral(p dOmega) == 4*pi`` for the full-sphere angular integral
        of the underlying scatterer.
    """
    s11 = np.asarray(s11, dtype=float)
    return s11 * (4.0 * np.pi) / (k**2 * c_sca)


def cli_normalized_s11_to_raw(s11_normalized_false, *, correction=2.0 * np.pi):
    """Undo the residual CLI ``normalize_s11=False`` factor.

    Setting MSTM's own ``normalize_s11`` keyword to ``False`` (exposed as
    ``write_inp_file(normalize_s11=False)``) corrects the *shape* of the
    CLI's printed S11 to match ``get_scattering_angle()``'s raw convention,
    but leaves a residual, exactly-constant ``2*pi`` factor (confirmed
    empirically to 5 significant figures across multiple angles/cases).
    This divides that factor back out.

    Parameters
    ----------
    s11_normalized_false : array-like
        S11 (or S12, S13, ...) values parsed from a CLI run with
        ``normalize_s11=False``.
    correction : float
        The residual factor to divide by. Exposed as a parameter (rather
        than hardcoded silently) in case a future MSTM version changes
        this; defaults to the empirically-confirmed ``2*pi``.

    Returns
    -------
    ndarray
        Values matching ``get_scattering_angle()``'s raw convention.
    """
    s11_normalized_false = np.asarray(s11_normalized_false, dtype=float)
    return s11_normalized_false / correction
