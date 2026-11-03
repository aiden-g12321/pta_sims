"""
The simulation itself: drawing noise-weighted inner products directly.

Derivation
----------
Prometheus' generative model for pulsar ``p`` is

    r = F a + F_D a_det + n,        n ~ N(0, N),

with ``N`` the white noise covariance and the likelihood taken in the metric
``Ntilde^-1``, the white-noise inverse projected orthogonal to the linearized
timing model.  Writing ``Ntilde^-1 = N^-1/2 P N^-1/2`` with ``P`` the
orthogonal projector that annihilates ``N^-1/2 M``, the compressed data are

    FNr  = F^T   Ntilde^-1 r = FNF   a + FNFD  a_det + xi_F
    FDNr = F_D^T Ntilde^-1 r = FNFD^T a + FDNFD a_det + xi_D

where the noise projections are ``xi_F = F^T Ntilde^-1 n`` and
``xi_D = F_D^T Ntilde^-1 n``.  Their joint covariance follows from the
idempotence of ``P``:

    Cov(xi_F, xi_F) = F^T Ntilde^-1 N Ntilde^-1 F
                    = F^T N^-1/2 P P N^-1/2 F
                    = F^T Ntilde^-1 F  =  FNF,

and identically for the other blocks.  So

    (xi_F, xi_D) ~ N(0, C),   C = [[FNF, FNFD], [FNFD^T, FDNFD]],

**exactly** -- the covariance of the compressed noise is the compressed Gram
matrix itself.  That is the whole trick: no TOAs, no residuals, no refit, and
nothing approximate.  It also means the deterministic and stochastic channels
share their noise, through the off-diagonal ``FNFD``, and must be drawn
jointly; drawing ``xi_F`` and ``xi_D`` independently would misstate the
correlation Prometheus' likelihood explicitly accounts for.

Rank deficiency
---------------
``C`` is severely singular on real data, for reasons set out in
:mod:`pta_sims.linalg`.  The factorization used here truncates the null space
rather than regularizing it, which is what preserves the exact linear
identities the real data satisfies.

Residual norm
-------------
``rNr = r^T Ntilde^-1 r`` is not part of Prometheus 0.1.x's ``Data``, but it
completes the sufficient statistics and later versions carry it.  It is
simulatable in closed form.  With ``v = [a; a_det]`` and ``w = N^-1/2 n``,

    rNr = v^T C v  +  2 v^T xi  +  w^T P w,

and the last term splits into a part determined by ``xi`` and an independent
remainder:

    w^T P w  =  xi^T C^+ xi  +  chi^2(rank(P) - rank(C)),

with ``rank(P) = N_TOA - N_tmpars``.  Only that residual chi-squared is fresh
randomness, and it is exactly the power the compressed basis cannot see.

Precision
---------
Everything above happens in float64 in natural (seconds) units.  The float32
renormalized arrays Prometheus stores are produced once, at the very end.
Simulating in float32 throughout does not work: the Gram matrices span 14
orders of magnitude in eigenvalue on the NG15 array, so the eigendecomposition
that the noise draw depends on has no valid digits left in single precision.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import units
from .coefficients import _as_rng
from .inner_products import InnerProducts
from .linalg import DEFAULT_RTOL, symmetrize


class CompressedNoise:
    """
    The compressed noise distribution of one PTA, factored once.

    Holds the eigendecomposition of the joint Gram matrix
    ``C = [F | F_D]^T Ntilde^-1 [F | F_D]`` -- which *is* the covariance of the
    compressed noise, see the module docstring -- so that many realizations can
    be drawn without repeating it.  The factorization is by far the dominant
    cost of a simulation (67 pulsars x a 180x180 ``eigh``), so an ensemble study
    should build this once and reuse it:

        noise = CompressedNoise.from_inner_products(ip)
        for seed in range(1000):
            sim = simulate_inner_products(ip, a=..., rng=seed, noise=noise)

    Directions whose eigenvalue falls below ``rtol * lambda_max`` are dropped
    rather than regularized; :mod:`pta_sims.linalg` explains why that is the
    statistically correct choice and not merely a numerical convenience.

    Parameters
    ----------
    C : ndarray, shape (Np, n, n)
        Joint Gram matrix in natural units, float64.
    rtol : float
        Eigenvalue truncation tolerance.
    """

    def __init__(self, C, rtol=DEFAULT_RTOL):
        self.C = symmetrize(C)
        self.rtol = rtol

        eigvals, eigvecs = np.linalg.eigh(self.C)
        max_eig = eigvals[..., -1]
        self.keep = eigvals > rtol * np.maximum(max_eig, 0.0)[..., None]
        kept = np.where(self.keep, eigvals, 0.0)

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.L = eigvecs * np.sqrt(kept)[..., None, :]
        self.rank = np.sum(self.keep, axis=-1)

        total = np.sum(np.abs(eigvals), axis=-1)
        self.info = {
            'rank': self.rank,
            'size': self.C.shape[-1],
            'max_eig': max_eig,
            'min_eig': eigvals[..., 0],
            'neg_frac': np.where(max_eig > 0,
                                 -np.minimum(eigvals[..., 0], 0.0)
                                 / np.maximum(max_eig, 1e-300), 0.0),
            'dropped_trace_frac': np.where(
                total > 0,
                np.sum(np.where(self.keep, 0.0, np.abs(eigvals)), axis=-1)
                / np.maximum(total, 1e-300), 0.0),
        }

    @classmethod
    def from_inner_products(cls, inner_products, rtol=DEFAULT_RTOL):
        """Build from an :class:`~pta_sims.inner_products.InnerProducts`."""
        return cls(inner_products.joint_matrix(), rtol=rtol)

    @property
    def size(self):
        return self.C.shape[-1]

    def draw(self, rng):
        """
        Draw one realization of the compressed noise, shape ``(Np, n)``.

        Exact for the singular covariance: the dropped directions receive
        identically zero, which is what preserves the linear identities the
        real compressed data satisfies.
        """
        z = rng.standard_normal(self.C.shape[:-1])
        return np.einsum('pij,pj->pi', self.L, z)

    def pinv_quadform(self, x):
        """``x^T C^+ x`` using the cached decomposition."""
        proj = np.einsum('pij,pi->pj', self.eigvecs, np.asarray(x, dtype=np.float64))
        inv = np.where(self.keep, 1.0 / np.where(self.keep, self.eigvals, 1.0), 0.0)
        return np.sum(proj ** 2 * inv, axis=-1)

    def whiten(self, x):
        """``C^{+/2} x``: standard normal components on the retained subspace."""
        inv_sqrt = np.where(self.keep,
                            1.0 / np.sqrt(np.where(self.keep, self.eigvals, 1.0)), 0.0)
        proj = np.einsum('pij,pi->pj', self.eigvecs, np.asarray(x, dtype=np.float64))
        return proj * inv_sqrt


@dataclass
class InnerProductSimulation:
    """
    Result of :func:`simulate_inner_products`.

    Attributes
    ----------
    inner_products : InnerProducts
        Copy of the input with ``FNr``, ``FDNr`` and (optionally) ``rNr``
        replaced by the simulated values.  Natural units, float64.
    a : ndarray, shape (Np, Nc) or None
        Injected stochastic amplitudes, nanoseconds.
    a_det : ndarray, shape (Np, Nd) or None
        Injected deterministic amplitudes, nanoseconds.
    noise_realization : ndarray, shape (Np, Nc + Nd) or None
        The realized noise projection ``xi``, in natural units -- not to be
        confused with the :class:`CompressedNoise` that generated it.  ``None``
        in noiseless mode.
    diagnostics : dict
        See :func:`simulate_inner_products`.
    """

    inner_products: InnerProducts
    a: Optional[np.ndarray] = None
    a_det: Optional[np.ndarray] = None
    noise_realization: Optional[np.ndarray] = None
    diagnostics: dict = field(default_factory=dict)


def simulate_inner_products(inner_products, a=None, a_det=None, rng=None,
                            include_white_noise=True, rtol=DEFAULT_RTOL,
                            simulate_rNr=True, noise=None):
    """
    Draw ``FNr`` (and ``FDNr``, ``rNr``) consistent with injected amplitudes.

    Parameters
    ----------
    inner_products : InnerProducts
        Supplies the fixed geometry ``FNF``, ``FNFD``, ``FDNFD``.  Its ``FNr``
        and ``FDNr`` are ignored and replaced.
    a : array_like, shape (Np, Nc), optional
        Stochastic Fourier amplitudes in **nanoseconds** (intrinsic red noise
        plus GWB, already summed).  Zero if omitted.
    a_det : array_like, shape (Np, Nd), optional
        Deterministic Fourier amplitudes in **nanoseconds**.  Zero if omitted.
    rng : numpy Generator, int, or None
    include_white_noise : bool
        When False the noise projection ``xi`` is set to its mean, zero, giving
        the noiseless data ``FNr = FNF a + FNFD a_det``.  Useful for checking
        sampler mechanics: note that even then the posterior mode for ``a`` is
        not exactly the injected value, because the prior still shrinks it.
    rtol : float
        Eigenvalue truncation for the joint noise covariance.  Ignored when
        ``noise`` is supplied.
    simulate_rNr : bool
        Also simulate ``r^T Ntilde^-1 r``.  Requires ``inner_products.ntoas``
        and ``.ntmpars``; skipped with a warning flag in the diagnostics
        otherwise.
    noise : CompressedNoise, optional
        A pre-built factorization of the joint Gram matrix.  Supply one when
        drawing many realizations from the same array -- the factorization
        costs far more than everything else combined.

    Returns
    -------
    InnerProductSimulation
    """
    rng = _as_rng(rng)
    ip = inner_products
    npsrs, nc, nd = ip.npsrs, ip.ncomponents, ip.ndet

    a = np.zeros((npsrs, nc)) if a is None else np.asarray(a, dtype=np.float64)
    if a.shape != (npsrs, nc):
        raise ValueError(f'a has shape {a.shape}, expected {(npsrs, nc)}.')

    if ip.has_deterministic:
        a_det = np.zeros((npsrs, nd)) if a_det is None else np.asarray(a_det, dtype=np.float64)
        if a_det.shape != (npsrs, nd):
            raise ValueError(f'a_det has shape {a_det.shape}, expected {(npsrs, nd)}.')
    elif a_det is not None:
        raise ValueError('a_det was supplied but the inner products carry no '
                         'deterministic basis (FNFD / FDNFD are None).')

    # amplitudes are in ns; the inner products are in natural (seconds) units
    v_ns = a if not ip.has_deterministic else np.concatenate([a, a_det], axis=-1)
    v = units.ns_to_seconds(v_ns)                               # (Np, Nc + Nd)

    if noise is None:
        noise = CompressedNoise.from_inner_products(ip, rtol=rtol)
    elif noise.size != ip.ncomponents + ip.ndet:
        raise ValueError(
            f'The supplied CompressedNoise has size {noise.size}, but these '
            f'inner products need {ip.ncomponents + ip.ndet}.')

    signal = np.einsum('pij,pj->pi', noise.C, v)                # (Np, n)
    xi = noise.draw(rng) if include_white_noise else np.zeros_like(signal)
    joint = signal + xi

    FNr_sim = joint[:, :nc]
    FDNr_sim = joint[:, nc:] if ip.has_deterministic else None

    diagnostics = {
        'joint_rank': np.asarray(noise.info['rank']),
        'joint_size': noise.size,
        'joint_neg_frac': np.asarray(noise.info['neg_frac']),
        'joint_dropped_trace_frac': np.asarray(noise.info['dropped_trace_frac']),
        'include_white_noise': include_white_noise,
        'rtol': noise.rtol,
    }

    rNr_sim = None
    if simulate_rNr:
        rNr_sim, rnr_info = _simulate_rNr(
            ip, noise, v, signal, xi, rng,
            include_white_noise=include_white_noise)
        diagnostics.update(rnr_info)

    sim_ip = ip.replace(FNr=FNr_sim, FDNr=FDNr_sim, rNr=rNr_sim)

    return InnerProductSimulation(
        inner_products=sim_ip,
        a=a,
        a_det=a_det if ip.has_deterministic else None,
        noise_realization=None if not include_white_noise else xi,
        diagnostics=diagnostics,
    )


def _simulate_rNr(ip, noise, v, signal, xi, rng, include_white_noise):
    """
    Simulate ``r^T Ntilde^-1 r``; see the module docstring for the decomposition.
    """
    info = {}
    signal_quad = np.einsum('pi,pi->p', v, signal)               # v^T C v
    cross = 2.0 * np.einsum('pi,pi->p', v, xi)

    if not include_white_noise:
        info['rNr_simulated'] = True
        info['rNr_chi2_dof'] = np.zeros(ip.npsrs, dtype=int)
        return signal_quad, info

    if ip.ntoas is None or ip.ntmpars is None:
        info['rNr_simulated'] = False
        info['rNr_note'] = ('ntoas / ntmpars unavailable, so the unobservable '
                            'chi-squared component of rNr could not be drawn; '
                            'rNr is left as None.')
        return None, info

    seen = noise.pinv_quadform(xi)                               # xi^T C^+ xi
    dof = np.asarray(ip.ntoas) - np.asarray(ip.ntmpars) - np.asarray(noise.rank)
    if np.any(dof < 0):
        raise ValueError(
            'Negative degrees of freedom for the unobserved part of rNr: the '
            'numerical rank of the compressed basis exceeds N_TOA - N_tmpars. '
            'Check ntoas / ntmpars.')
    unseen = rng.chisquare(dof.astype(np.float64))

    info['rNr_simulated'] = True
    info['rNr_chi2_dof'] = dof
    return signal_quad + cross + seen + unseen, info


def whiten(inner_products, reference=None, rtol=DEFAULT_RTOL, subtract=None,
           noise=None):
    """
    Whiten a compressed data vector against its own noise covariance.

    With ``x = [FNr; FDNr]`` and the injected signal ``s = C v`` removed,
    ``C^{+/2} (x - s)`` is a standard normal vector on the retained
    ``rank(C)`` directions.  This is the sharpest available check that a
    simulated dataset is statistically what it claims to be, and applied to
    *real* data with ``subtract=None`` it measures how far the real residuals
    depart from pure white noise (i.e. how much red signal is present).

    Parameters
    ----------
    inner_products : InnerProducts
    reference : InnerProducts, optional
        Supplies the Gram matrices, if different from ``inner_products``.
    rtol : float
    subtract : array_like, shape (Np, Nc + Nd), optional
        Signal term to remove before whitening, in natural units.
    noise : CompressedNoise, optional
        Pre-built factorization, as in :func:`simulate_inner_products`.

    Returns
    -------
    ndarray, shape (Np, n)
        Whitened components; entries on dropped directions are exactly zero.
    """
    if noise is None:
        noise = CompressedNoise.from_inner_products(reference or inner_products,
                                                    rtol=rtol)
    x = inner_products.joint_vector()
    if subtract is not None:
        x = x - np.asarray(subtract, dtype=np.float64)
    return noise.whiten(x)


def deterministic_metric(inner_products, phi=None):
    """
    The metric a deterministic template is measured against.

    Without ``phi`` this is just ``FDNFD``, the white noise floor.  With it, the
    stochastic processes are marginalized:

        M = FDNFD - FNFD^T (FNF + phi^-1)^-1 FNFD

    The metric depends only on the array and the stochastic prior, not on the
    template, so a sky scan or an amplitude solve should build it once and pass
    it to :func:`matched_filter_snr` -- rebuilding it per evaluation costs a
    batched ``(Np, Nc, Nc)`` solve every time and dominates everything else.

    Parameters
    ----------
    inner_products : InnerProducts
    phi : array_like, shape (Np, Nc) or (Np, Nc, Nc), optional
        Per-pulsar prior covariance of the stochastic amplitudes in ns^2, as a
        diagonal or a dense block.  Pass the per-pulsar diagonal of the total
        ``phi_cube`` for the usual CURN-equivalent approximation, which drops
        the inter-pulsar correlation of the background.  On the NG15 array with
        an NG15-amplitude Hellings-Downs background that approximation
        overstates the SNR of a 4 nHz continuous wave by about 10 per cent
        (8.7 against 7.9 for the exactly correlated calculation), so it is fine
        for tuning an injection and not for quoting a significance.

    Returns
    -------
    ndarray, shape (Np, Nd, Nd)
    """
    ip = inner_products
    if not ip.has_deterministic:
        raise ValueError('No deterministic basis in these inner products.')

    M = np.asarray(ip.FDNFD, dtype=np.float64)
    if phi is None:
        return M

    # phi is in ns^2 while FNF is in natural units: one factor of renorm^2
    phi = np.asarray(phi, dtype=np.float64) / units.RENORM ** 2
    if phi.ndim == 2:
        # A component with zero prior variance is pinned to zero, so its
        # phi^-1 is infinite and it contributes nothing to be marginalized;
        # a large finite value stands in without tripping the solve.
        floor = 1e-30 * np.max(phi)
        phi_inv = np.einsum('pi,ij->pij', 1.0 / np.maximum(phi, floor),
                            np.eye(phi.shape[-1]))
    else:
        phi_inv = np.linalg.inv(phi)

    Sigma = np.asarray(ip.FNF, dtype=np.float64) + phi_inv
    return M - np.einsum('pij,pjk->pik',
                         np.swapaxes(ip.FNFD, -1, -2),
                         np.linalg.solve(Sigma, ip.FNFD))


def deterministic_statistic(inner_products, phi=None, metric=None):
    """
    The marginalized matched-filter statistic for a deterministic signal.

    Marginalizing the stochastic amplitudes out of the likelihood leaves a
    quadratic in the deterministic amplitudes,

        ln L(a_det) = a_det . y  -  (1/2) a_det^T M a_det  +  const,

    with ``M`` from :func:`deterministic_metric` and

        y = FDNr - FNFD^T (FNF + phi^-1)^-1 FNr.

    So ``y`` is what the data has to say about a deterministic template, once
    the red noise and background have been accounted for, and
    :func:`marginal_delta_ln_likelihood` evaluates the quadratic.  At the true
    amplitudes the expectation of that quadratic is ``SNR^2 / 2``.

    Parameters
    ----------
    inner_products : InnerProducts
    phi : array_like, optional
        Stochastic prior covariance in ns^2; see :func:`deterministic_metric`.
    metric : ndarray, optional
        Unused here, accepted so that callers can pass the same arguments to
        both functions.

    Returns
    -------
    ndarray, shape (Np, Nd)
        In natural units, matching ``FDNr``.
    """
    ip = inner_products
    if not ip.has_deterministic:
        raise ValueError('No deterministic basis in these inner products.')
    if ip.FDNr is None:
        raise ValueError('These inner products carry no FDNr.')

    y = np.asarray(ip.FDNr, dtype=np.float64)
    if phi is None:
        return y

    phi = np.asarray(phi, dtype=np.float64) / units.RENORM ** 2
    if phi.ndim == 2:
        floor = 1e-30 * np.max(phi)
        phi_inv = np.einsum('pi,ij->pij', 1.0 / np.maximum(phi, floor),
                            np.eye(phi.shape[-1]))
    else:
        phi_inv = np.linalg.inv(phi)

    Sigma = np.asarray(ip.FNF, dtype=np.float64) + phi_inv
    # the trailing [..., None] keeps numpy's batched solve reading FNr as one
    # right-hand-side vector per pulsar rather than a single (Np, Nc) matrix
    solved = np.linalg.solve(Sigma, np.asarray(ip.FNr, dtype=np.float64)[..., None])
    correction = np.einsum('pji,pj->pi', ip.FNFD, solved[..., 0])
    return y - correction


def marginal_delta_ln_likelihood(inner_products, a_det, phi=None, metric=None,
                                 statistic=None):
    """
    Log-likelihood of a deterministic template relative to no signal, with the
    stochastic processes marginalized.

    Evaluates ``a_det . y - (1/2) a_det^T M a_det``; see
    :func:`deterministic_statistic`.  Scanning it over a template parameter is
    the cheapest honest look at how visible an injection is -- much more
    informative than a slice through the full posterior at fixed Fourier
    coefficients, which holds the stochastic amplitudes at a preconditioned
    point rather than marginalizing them and can overstate the evidence by an
    order of magnitude.

    Parameters
    ----------
    inner_products : InnerProducts
    a_det : array_like, shape (Np, Nd)
        Deterministic amplitudes in nanoseconds.
    phi : array_like, optional
        Stochastic prior covariance in ns^2.
    metric, statistic : ndarray, optional
        Precomputed ``M`` and ``y``.  Pass both when scanning; they do not
        depend on the template.

    Returns
    -------
    float
    """
    if metric is None:
        metric = deterministic_metric(inner_products, phi=phi)
    if statistic is None:
        statistic = deterministic_statistic(inner_products, phi=phi)

    a_det_s = units.ns_to_seconds(a_det)
    linear = float(np.einsum('pi,pi->', a_det_s, statistic))
    quad = float(np.einsum('pi,pij,pj->', a_det_s, metric, a_det_s))
    return linear - 0.5 * quad


def matched_filter_snr(inner_products, a_det, phi=None, metric=None,
                       per_pulsar=False):
    """
    Matched-filter SNR of an injected deterministic signal.

    Parameters
    ----------
    inner_products : InnerProducts
    a_det : array_like, shape (Np, Nd)
        Deterministic amplitudes in nanoseconds.
    phi : array_like, optional
        Stochastic prior covariance in ns^2; see :func:`deterministic_metric`.
        Without it the SNR is measured against the white noise floor alone.
    metric : ndarray, shape (Np, Nd, Nd), optional
        A metric already built by :func:`deterministic_metric`.  Takes
        precedence over ``phi``; use it for scans.
    per_pulsar : bool
        Return the per-pulsar contributions rather than the array total.  The
        total is the root-sum-square of these.

    Returns
    -------
    float, or ndarray of shape (Np,) when ``per_pulsar``
    """
    if metric is None:
        metric = deterministic_metric(inner_products, phi=phi)

    a_det_s = units.ns_to_seconds(a_det)
    snr_sq = np.einsum('pi,pij,pj->p', a_det_s, metric, a_det_s)
    if per_pulsar:
        return np.sqrt(np.maximum(snr_sq, 0.0))
    return float(np.sqrt(max(float(np.sum(snr_sq)), 0.0)))
