"""
Drawing the Fourier amplitudes that the simulated data is conditioned on.

Prometheus' generative model for a pulsar's residuals is

    r_p = F_p a_p + F_D,p a_det,p + n_p,      n_p ~ N(0, Ntilde_p)

with the stochastic amplitudes ``a`` drawn jointly across pulsars from a
zero-mean Gaussian whose covariance is the ``phi`` of the spectral models,

    a ~ N(0, phi),   phi = phi_intrinsic + phi_GWB.

For the usual case of no inter-frequency correlations, ``phi`` is block
diagonal in *frequency component*: distinct components are independent and each
carries an ``(Np, Np)`` inter-pulsar covariance.  Prometheus stores exactly that
as ``phi_cube`` of shape ``(Nc, Np, Np)``, and the intrinsic (diagonal) and GWB
(``S(f) * Gamma_HD``) pieces simply add.  Sampling then factorizes into ``Nc``
independent ``Np``-dimensional draws.

With inter-frequency correlations switched on the covariance no longer
factorizes over components and the full ``(Np * Nc)`` covariance has to be
assembled; :func:`draw_from_blocks` does that, using the same Kronecker
structure Prometheus itself uses, ``blockdiag_p(phi_p) + kron(Gamma, phi_spec)``
in the row-major ``(pulsar, component)`` ordering of ``a.flatten()``.

All amplitudes here are in **nanoseconds**, matching ``phi`` in ns^2 and
Prometheus' ``renorm`` convention.
"""

import numpy as np

from .linalg import psd_factor, DEFAULT_RTOL


def _as_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def draw_from_phi_cube(phi_cube, rng=None, rtol=DEFAULT_RTOL):
    """
    Draw stochastic Fourier amplitudes from a component-diagonal prior.

    Parameters
    ----------
    phi_cube : array_like, shape (Nc, Np, Np)
        Prior covariance of the Fourier amplitudes in ns^2, as returned by
        ``SpectralModel.get_phi_cube``.  Element ``(c, i, j)`` is the covariance
        of component ``c`` between pulsars ``i`` and ``j``.  May be float32 --
        it is upcast before any factorization.
    rng : numpy Generator, int, or None
    rtol : float
        Eigenvalue truncation for the per-component factorization.  A GWB model
        restricted to fewer frequencies than the data leaves exactly-zero
        components; those are handled by the truncation and simply produce zero
        amplitudes.

    Returns
    -------
    a : ndarray, shape (Np, Nc), float64
        Fourier amplitudes in nanoseconds.
    info : dict
        Per-component factorization diagnostics from :func:`psd_factor`.
    """
    rng = _as_rng(rng)
    phi_cube = np.asarray(phi_cube, dtype=np.float64)
    nc, npsr, _ = phi_cube.shape

    L, info = psd_factor(phi_cube, rtol=rtol)              # (Nc, Np, Np)
    z = rng.standard_normal((nc, npsr))
    a_cp = np.einsum('cij,cj->ci', L, z)                   # (Nc, Np)
    return a_cp.T.copy(), info


def draw_from_blocks(per_pulsar_phi, correlation_matrix=None, common_phi=None,
                     rng=None, rtol=DEFAULT_RTOL):
    """
    Draw amplitudes from a prior with inter-frequency correlations.

    Builds the full ``(Np * Nc, Np * Nc)`` covariance

        blockdiag_p(per_pulsar_phi[p])  +  kron(correlation_matrix, common_phi)

    in the row-major ``(pulsar, component)`` ordering that Prometheus'
    ``a.flatten()`` uses, then factors it once.  For a 67-pulsar, 30-frequency
    array this is a 4020 x 4020 eigendecomposition -- a few seconds, done once.

    Parameters
    ----------
    per_pulsar_phi : array_like, shape (Np, Nc, Nc)
        Dense per-pulsar prior covariance in ns^2.
    correlation_matrix : array_like, shape (Np, Np), optional
        Inter-pulsar correlation of the common process (e.g. Hellings-Downs).
    common_phi : array_like, shape (Nc, Nc), optional
        Spectral covariance of the common process in ns^2.  If the common
        process uses fewer components than the data, pad with zeros.
    rng, rtol
        As in :func:`draw_from_phi_cube`.

    Returns
    -------
    a : ndarray, shape (Np, Nc), float64
    info : dict
    """
    rng = _as_rng(rng)
    per_pulsar_phi = np.asarray(per_pulsar_phi, dtype=np.float64)
    npsr, nc, _ = per_pulsar_phi.shape

    cov = np.zeros((npsr * nc, npsr * nc), dtype=np.float64)
    for p in range(npsr):
        sl = slice(p * nc, (p + 1) * nc)
        cov[sl, sl] += per_pulsar_phi[p]

    if correlation_matrix is not None and common_phi is not None:
        cov += np.kron(np.asarray(correlation_matrix, dtype=np.float64),
                       np.asarray(common_phi, dtype=np.float64))

    L, info = psd_factor(cov, rtol=rtol)
    a_flat = L @ rng.standard_normal(npsr * nc)
    return a_flat.reshape(npsr, nc), info


def deterministic_coefficients(det_model, det_params, psr_phases=None,
                               psr_dists=None):
    """
    Evaluate a Prometheus deterministic model's Fourier amplitudes.

    This calls the model's own ``get_coeffs_func``, so the injection is
    *exactly* the template the sampler will later evaluate -- including the
    Tukey window, the extended-baseline FFT and the float32 arithmetic
    Prometheus runs in.  Using a higher-precision or differently-discretized
    version of the same waveform would inject a signal the model cannot
    represent, and would show up as a biased recovery.

    Parameters
    ----------
    det_model : prometheus.deterministic_models.DeterministicModel
    det_params : array_like, shape (nparams,)
    psr_phases : array_like, shape (Np,), optional
        Required when ``det_model.with_psr_params``.
    psr_dists : array_like, shape (Np,), optional
        Pulsar distances in kpc.  Required when ``det_model.with_psr_params``.

    Returns
    -------
    a_det : ndarray, shape (Np, Nd), float64
        Amplitudes in nanoseconds.
    """
    import jax.numpy as jnp

    if det_model.with_psr_params and (psr_phases is None or psr_dists is None):
        raise ValueError(
            f"Deterministic model '{det_model.name}' has with_psr_params=True, "
            'so psr_phases and psr_dists must both be supplied.')

    a_det = det_model.get_coeffs_func(
        jnp.asarray(det_params),
        None if psr_phases is None else jnp.asarray(psr_phases),
        None if psr_dists is None else jnp.asarray(psr_dists))
    return np.asarray(a_det, dtype=np.float64)
