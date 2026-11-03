"""Tests for pta_sims.detectability on the toy array."""
import numpy as np
import pytest

from pta_sims import detectability, simulate_inner_products
from pta_sims.simulate import CompressedNoise, deterministic_metric, deterministic_statistic


def test_strip_deterministic_recovers_the_signal_free_data(toy_ip):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(toy_ip.npsrs, toy_ip.ncomponents)) * 50.0
    a_det = rng.normal(size=(toy_ip.npsrs, toy_ip.FDNFD.shape[-1])) * 200.0
    noise = CompressedNoise.from_inner_products(toy_ip)
    with_cw = simulate_inner_products(toy_ip, a, a_det, np.random.default_rng(7), noise=noise)
    without = simulate_inner_products(toy_ip, a, np.zeros_like(a_det), np.random.default_rng(7), noise=noise)
    stripped = detectability.strip_deterministic(with_cw.inner_products, a_det)
    ref = without.inner_products
    assert np.allclose(stripped.FNr, ref.FNr, rtol=1e-9, atol=1e-12 * np.abs(ref.FNr).max())
    assert np.allclose(stripped.FDNr, ref.FDNr, rtol=1e-9, atol=1e-12 * np.abs(ref.FDNr).max())
    assert np.allclose(stripped.rNr, ref.rNr, rtol=1e-9)


def test_hyper_log_likelihood_prefers_the_true_spectrum_on_average(toy_ip):
    """The marginal likelihood, averaged over draws, peaks at the generating phi."""
    nfreqs = toy_ip.ncomponents // 2
    freqs = np.arange(1, nfreqs + 1) / (10.0 * 365.25 * 86400.0)
    phi_true = detectability.power_law_phi(freqs, -13.0, 4.0)
    phi_cube = np.zeros((toy_ip.ncomponents, toy_ip.npsrs, toy_ip.npsrs))
    ii = np.arange(toy_ip.npsrs)
    phi_cube[:, ii, ii] = phi_true[:, None] * 1e18          # ns^2
    noise = CompressedNoise.from_inner_products(toy_ip)
    rng = np.random.default_rng(11)
    gains = []
    for _ in range(40):
        a = np.sqrt(phi_true * 1e18)[None, :] * rng.standard_normal((toy_ip.npsrs, toy_ip.ncomponents))
        ip = simulate_inner_products(toy_ip, a, None, rng, noise=noise).inner_products
        def total(log10_A, gamma):
            phi = detectability.power_law_phi(freqs, log10_A, gamma)
            return sum(detectability.hyper_log_likelihood(ip.FNF[p], ip.FNr[p], ip.rNr[p], phi)
                       for p in range(toy_ip.npsrs))
        gains.append([total(-13.0, 4.0) - total(-12.5, 4.0), total(-13.0, 4.0) - total(-13.0, 3.0)])
    gains = np.mean(gains, axis=0)
    assert np.all(gains > 0.0)


def test_phase_marginalized_statistic_is_below_the_true_phase_value_without_noise():
    """With the right phases known, marginalizing them can only lose information."""
    rng = np.random.default_rng(5)
    Np, Nd = 3, 8
    metric = np.stack([np.eye(Nd) for _ in range(Np)])
    E = rng.normal(size=(Np, Nd)); Pc = rng.normal(size=(Np, Nd)); Ps = rng.normal(size=(Np, Nd))
    true = E + Pc                                     # phase offset 0
    y = np.einsum('pij,pj->pi', metric, true)
    at_truth = float(np.einsum('pi,pi->', true, y) - 0.5 * np.einsum('pi,pij,pj->', true, metric, true))
    marg = detectability.phase_marginalized_statistic(y, E, Pc, Ps, metric)
    assert marg < at_truth
    assert marg > 0.0
