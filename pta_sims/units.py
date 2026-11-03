"""
Unit and scaling conventions shared with Prometheus.

Prometheus stores every noise-weighted inner product in a renormalized basis in
which Fourier mode amplitudes carry units of nanoseconds rather than seconds.
Concretely, with ``renorm = 1e9``,

    a_[ns] = renorm * a_[s],

and an inner product is rescaled by one factor of ``1/renorm`` for each *basis*
index it carries (a residual index carries none):

    FNFs   = F^T  N^-1 F    / renorm^2      (two basis indices)
    FNrs   = F^T  N^-1 r    / renorm        (one basis index, one residual)
    TNTDs  = F^T  N^-1 F_D  / renorm^2
    TDNTDs = F_D^T N^-1 F_D / renorm^2
    TDNrs  = F_D^T N^-1 r   / renorm

Every quadratic form appearing in the likelihood -- ``a . FNrs``,
``a^T FNFs a``, ``a_det^T TDNTDs a_det`` -- is then invariant, because each
factor of ``renorm`` introduced by an amplitude is cancelled by the
``1/renorm`` carried by the matching basis index.

This module keeps that bookkeeping in one place.  Everything in ``pta_sims``
works in *natural* units (seconds, unscaled inner products) and in float64
until the very last step, where :func:`to_renorm` and a float32 cast produce
the arrays Prometheus consumes.
"""

import numpy as np

#: Renormalization constant.  Must match ``prometheus.utilities.renorm``.
RENORM = 1.0e9

#: The dtype Prometheus stores its data arrays in.
PROMETHEUS_DTYPE = np.float32

#: The dtype every intermediate calculation in ``pta_sims`` uses.
WORK_DTYPE = np.float64


def to_renorm(array, n_basis_axes):
    """
    Convert a natural-unit inner product to Prometheus' renormalized units.

    Parameters
    ----------
    array : array_like
        Inner product in natural units (seconds, unscaled).
    n_basis_axes : int
        Number of design-matrix (basis) indices the object carries.  One for
        ``F^T N^-1 r``, two for ``F^T N^-1 F``, and so on.

    Returns
    -------
    ndarray
        ``array / RENORM ** n_basis_axes``.
    """
    return np.asarray(array, dtype=WORK_DTYPE) / RENORM ** n_basis_axes


def from_renorm(array, n_basis_axes):
    """Inverse of :func:`to_renorm`: renormalized units back to natural units."""
    return np.asarray(array, dtype=WORK_DTYPE) * RENORM ** n_basis_axes


def seconds_to_ns(array):
    """Convert an amplitude in seconds to nanoseconds."""
    return np.asarray(array, dtype=WORK_DTYPE) * RENORM


def ns_to_seconds(array):
    """Convert an amplitude in nanoseconds to seconds."""
    return np.asarray(array, dtype=WORK_DTYPE) / RENORM


def downcast(array):
    """Cast to the float32 dtype Prometheus stores, after all float64 work."""
    return np.asarray(array, dtype=WORK_DTYPE).astype(PROMETHEUS_DTYPE)
