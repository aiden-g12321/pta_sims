"""
What a deterministic signal is worth to the analysis that will look for it.

The matched-filter SNR of an injected continuous wave overstates its Bayesian
detectability for two reasons that this module quantifies:

* **pulsar-term phases.**  Half of the matched-filter SNR of a CW is in the
  pulsar terms, whose phases the injection knows but the analysis samples.
  :func:`phase_marginalized_statistic` evaluates the marginalized
  log-likelihood gain with those phases integrated out.
* **absorption by the noise model.**  The per-pulsar red-noise and common
  power laws are free in the analysis, and they can soak up part of a
  single-frequency excess.  :func:`absorption` measures how much of the
  signal's log-likelihood gain the noise hyper-parameters can claim.

The remainder, ``phase_marginalized - absorption``, is roughly what has to
beat the Occam penalty of the CW parameters themselves for the signal mode to
carry weight in the posterior.

Everything here works on :class:`~pta_sims.inner_products.InnerProducts` in
natural units (seconds, float64) and treats the stochastic prior as
uncorrelated between pulsars, which is what the marginalized statistics in
:mod:`pta_sims.simulate` do as well.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import units
from .inner_products import InnerProducts

__all__ = ['power_law_phi', 'hyper_log_likelihood', 'strip_deterministic',
           'fit_power_laws', 'absorption', 'cw_terms',
           'phase_marginalized_statistic']

DEFAULT_RN_BOUNDS = ((-20.0, -11.0), (0.0, 7.0))
DEFAULT_GWB_BOUNDS = ((-18.0, -11.0), (0.0, 7.0))


def power_law_phi(freqs, log10_A, gamma):
    """
    Power-law prior variance of the Fourier amplitudes, in s^2.

    Same convention as ``prometheus.spectra.power_law`` (one value per
    sine/cosine pair, repeated), without the ns^2 renormalization.
    """
    from prometheus import utilities as putils
    freqs = np.asarray(freqs, dtype=np.float64)
    df = np.diff(np.concatenate([[0.0], freqs]))
    v = (10.0 ** (2 * log10_A) / (12 * np.pi ** 2) * putils.fyr ** (gamma - 3)
         * freqs ** (-gamma) * df)
    return np.repeat(v, 2)


def hyper_log_likelihood(FNF, FNr, rNr, phi):
    """
    Log marginal likelihood of one pulsar's compressed data given the prior
    variances ``phi`` (s^2) of its stochastic Fourier amplitudes, up to a
    constant.  The amplitudes are integrated out analytically.
    """
    K = FNF + np.diag(1.0 / phi)
    L = np.linalg.cholesky(K)
    z = np.linalg.solve(L, FNr)
    return -0.5 * (np.sum(np.log(phi)) + 2 * np.sum(np.log(np.diag(L))) + rNr - z @ z)


def strip_deterministic(ip, a_det):
    """
    The same compressed data with a deterministic signal removed.

    Parameters
    ----------
    ip : InnerProducts
    a_det : ndarray, shape (Np, Nd)
        Amplitudes in nanoseconds.
    """
    a = units.ns_to_seconds(a_det)
    FNr = ip.FNr - np.einsum('pij,pj->pi', ip.FNFD, a)
    FDNr = ip.FDNr - np.einsum('pij,pj->pi', ip.FDNFD, a)
    rNr = (ip.rNr - 2 * np.einsum('pi,pi->p', a, ip.FDNr)
           + np.einsum('pi,pij,pj->p', a, ip.FDNFD, a))
    return InnerProducts(FNF=ip.FNF, FNr=FNr, FNFD=ip.FNFD, FDNFD=ip.FDNFD,
                         FDNr=FDNr, rNr=rNr, ntoas=ip.ntoas, ntmpars=ip.ntmpars,
                         psr_names=ip.psr_names)


def fit_power_laws(ip, freqs, rn0, gwb0, rn_bounds=DEFAULT_RN_BOUNDS,
                   gwb_bounds=DEFAULT_GWB_BOUNDS, rounds=3):
    """
    Maximize the noise-only marginal likelihood over per-pulsar and common
    power-law hyper-parameters, by alternating Nelder-Mead fits.

    Returns
    -------
    (ln_max, rn_fit, gwb_fit)
    """
    Np = ip.FNF.shape[0]
    rn_fit = np.array(rn0, dtype=np.float64).copy()
    g = np.array(gwb0, dtype=np.float64).copy()

    def inside(x, b):
        return b[0][0] < x[0] < b[0][1] and b[1][0] < x[1] < b[1][1]

    total = None
    for _ in range(rounds):
        gphi = power_law_phi(freqs, *g)
        for p in range(Np):
            def f(x, p=p):
                if not inside(x, rn_bounds):
                    return 1e30
                return -hyper_log_likelihood(ip.FNF[p], ip.FNr[p], ip.rNr[p],
                                             power_law_phi(freqs, *x) + gphi)
            r = minimize(f, rn_fit[p], method='Nelder-Mead',
                         options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 400})
            rn_fit[p] = r.x
        rn_phi = [power_law_phi(freqs, *rn_fit[p]) for p in range(Np)]

        def fg(x):
            if not inside(x, gwb_bounds):
                return 1e30
            gp = power_law_phi(freqs, *x)
            return -sum(hyper_log_likelihood(ip.FNF[p], ip.FNr[p], ip.rNr[p], rn_phi[p] + gp)
                        for p in range(Np))
        r = minimize(fg, g, method='Nelder-Mead',
                     options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 200})
        g, total = r.x, -r.fun
    return total, rn_fit, g


def absorption(ip, a_det, freqs, rn_true, gwb_true, **kwargs):
    """
    How much of a deterministic signal the noise hyper-parameters can absorb.

    Fits the per-pulsar and common power laws to the data with and without
    the signal, and returns the extra log-likelihood the noise-only model gains
    *because* the signal is there, relative to the injected hyper-parameters:

        [max lnL_noise(data) - lnL_noise(data | truth)]
      - [max lnL_noise(data - signal) - lnL_noise(data - signal | truth)]

    That is the part of the signal's log-likelihood gain the no-signal mode
    of the analysis can claim for itself.

    Returns
    -------
    (nats, gwb_fit_with_signal, gwb_fit_without_signal)
    """
    rn_true = np.asarray(rn_true, dtype=np.float64)
    gphi = power_law_phi(freqs, *gwb_true)

    def base(ipx):
        return sum(hyper_log_likelihood(ipx.FNF[p], ipx.FNr[p], ipx.rNr[p],
                                        power_law_phi(freqs, *rn_true[p]) + gphi)
                   for p in range(ipx.FNF.shape[0]))

    with_max, _, g_with = fit_power_laws(ip, freqs, rn_true, gwb_true, **kwargs)
    without = strip_deterministic(ip, a_det)
    without_max, _, g_without = fit_power_laws(without, freqs, rn_true, gwb_true, **kwargs)
    return (with_max - base(ip)) - (without_max - base(without)), g_with, g_without


def cw_terms(det_model, params, psr_phases, psr_dists):
    """
    Split a CW template into its Earth term and pulsar-term quadratures.

    Uses the waveform itself: the pulsar term flips sign under a pi/2 shift
    of the (orbital) pulsar phase while the Earth term does not.

    Returns
    -------
    (E, Pc, Ps) : ndarrays, shape (Np, Nd), in seconds
        The full template at pulsar-phase offsets ``dphi`` from ``psr_phases``
        is ``E + Pc cos(2 dphi) + Ps sin(2 dphi)``.
    """
    from .coefficients import deterministic_coefficients
    psr_phases = np.asarray(psr_phases, dtype=np.float64)

    def coef(ph):
        return units.ns_to_seconds(deterministic_coefficients(det_model, params, ph, psr_dists))

    a0, a1 = coef(psr_phases), coef(psr_phases + np.pi / 2)
    E, Pc = (a0 + a1) / 2, (a0 - a1) / 2
    Ps = (coef(psr_phases + np.pi / 4) - coef(psr_phases + 3 * np.pi / 4)) / 2
    return E, Pc, Ps


def phase_marginalized_statistic(statistic, E, Pc, Ps, metric, scale=1.0, nphase=64):
    """
    Marginalized log-likelihood gain of a CW template with every pulsar-term
    phase integrated over its period, summed over pulsars:

        sum_p  ln < exp( a_p . y_p - a_p^T M_p a_p / 2 ) >_phase

    with ``a = scale * (E + P(phase))``.  ``statistic`` and ``metric`` are
    the ``y`` and ``M`` of :func:`pta_sims.simulate.deterministic_statistic`
    and :func:`pta_sims.simulate.deterministic_metric`.
    """
    grid = np.linspace(0.0, np.pi, nphase, endpoint=False)
    vals = []
    for th in grid:
        a = scale * (E + Pc * np.cos(2 * th) + Ps * np.sin(2 * th))
        vals.append(np.einsum('pi,pi->p', a, statistic)
                    - 0.5 * np.einsum('pi,pij,pj->p', a, metric, a))
    return float((logsumexp(np.array(vals), axis=0) - np.log(nphase)).sum())
