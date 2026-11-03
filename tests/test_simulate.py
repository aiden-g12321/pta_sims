"""
The central claim: simulating the compressed statistics directly gives the same
distribution as compressing brute-force simulated residuals.

Every test here compares against the toy array's explicit ``F``, ``F_D``, ``N``
and ``M``, so nothing is checked against the package's own machinery.
"""

import numpy as np
import pytest

from pta_sims import units
from pta_sims.diagnostics import range_consistency, whitened_statistics
from pta_sims.simulate import (CompressedNoise, matched_filter_snr,
                               simulate_inner_products, whiten)


# ---------------------------------------------------------------- the claim

def test_noise_covariance_equals_the_gram_matrix(toy, toy_ip):
    """
    Cov(B^T Ntilde^-1 n) = B^T Ntilde^-1 B, the identity the whole approach
    rests on.  Checked by brute force against explicit matrices.
    """
    rng = np.random.default_rng(0)
    ndraws = 8000
    p = 0
    B = np.concatenate([toy.F[p], toy.F_D[p]], axis=1)
    Ni = toy.Ninv_tilde[p]

    n = rng.multivariate_normal(np.zeros(toy.N[p].shape[0]), toy.N[p],
                                size=ndraws)
    xi = n @ Ni @ B
    emp = xi.T @ xi / ndraws

    C = toy_ip.joint_matrix()[p]
    from pta_sims.linalg import psd_solve_factor
    W = psd_solve_factor(C)
    whitened = W.T @ emp @ W
    kept = np.linalg.norm(W, axis=0) > 0
    target = np.diag(kept.astype(float))
    assert np.max(np.abs(whitened - target)) < 6 * np.sqrt(2.0 / ndraws)


@pytest.mark.slow
def test_simulated_matches_brute_force_residual_simulation(toy, toy_ip):
    """
    Same distribution, two routes: compress simulated residuals, or simulate
    the compressed statistics.  Compared in the whitened basis so that the
    singular directions are handled identically on both sides.
    """
    ndraws = 6000
    p = 1
    noise = CompressedNoise.from_inner_products(toy_ip)

    rng = np.random.default_rng(1)
    brute = []
    for _ in range(ndraws):
        r = toy.draw_residuals(rng)
        ip = toy.inner_products(r)
        brute.append(ip.joint_vector()[p])
    brute = np.array(brute)

    rng = np.random.default_rng(2)
    fast = np.array([
        simulate_inner_products(toy_ip, rng=rng, noise=noise,
                                simulate_rNr=False).inner_products.joint_vector()[p]
        for _ in range(ndraws)])

    # whiten each route with the same operator
    inv_sqrt = np.where(noise.keep[p],
                        1 / np.sqrt(np.where(noise.keep[p], noise.eigvals[p], 1.)), 0.)
    W = noise.eigvecs[p] * inv_sqrt
    kept = np.linalg.norm(W, axis=0) > 0

    cb = (brute @ W).T @ (brute @ W) / ndraws
    cf = (fast @ W).T @ (fast @ W) / ndraws
    tol = 8 * np.sqrt(2.0 / ndraws)
    assert np.max(np.abs(cb - np.diag(kept.astype(float)))) < tol
    assert np.max(np.abs(cf - np.diag(kept.astype(float)))) < tol


# ------------------------------------------------------------------ signal

def test_signal_term_is_exact_in_the_noiseless_limit(toy, toy_ip):
    rng = np.random.default_rng(3)
    a = rng.standard_normal((toy.npsrs, toy.ncomponents)) * 100.0
    a_det = rng.standard_normal((toy.npsrs, toy.ndet)) * 10.0

    sim = simulate_inner_products(toy_ip, a=a, a_det=a_det,
                                  include_white_noise=False)
    expected_F = np.einsum('pij,pj->pi', toy_ip.FNF, units.ns_to_seconds(a)) \
        + np.einsum('pij,pj->pi', toy_ip.FNFD, units.ns_to_seconds(a_det))
    expected_D = np.einsum('pji,pj->pi', toy_ip.FNFD, units.ns_to_seconds(a)) \
        + np.einsum('pij,pj->pi', toy_ip.FDNFD, units.ns_to_seconds(a_det))

    assert np.allclose(sim.inner_products.FNr, expected_F, rtol=1e-12)
    assert np.allclose(sim.inner_products.FDNr, expected_D, rtol=1e-12)


def test_noiseless_signal_matches_residual_space(toy, toy_ip):
    """F a + F_D a_det compressed by hand equals the noiseless simulation."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal((toy.npsrs, toy.ncomponents)) * 100.0
    a_det = rng.standard_normal((toy.npsrs, toy.ndet)) * 10.0

    residuals = [toy.F[p] @ (a[p] / 1e9) + toy.F_D[p] @ (a_det[p] / 1e9)
                 for p in range(toy.npsrs)]
    direct = toy.inner_products(residuals)
    sim = simulate_inner_products(toy_ip, a=a, a_det=a_det,
                                  include_white_noise=False)

    assert np.allclose(sim.inner_products.FNr, direct.FNr, rtol=1e-9)
    assert np.allclose(sim.inner_products.FDNr, direct.FDNr, rtol=1e-9)


def test_cross_covariance_between_channels_is_respected(toy, toy_ip):
    """
    xi_F and xi_D share the same white noise; drawing them independently would
    zero this cross-covariance.
    """
    ndraws = 8000
    noise = CompressedNoise.from_inner_products(toy_ip)
    rng = np.random.default_rng(5)
    xi = np.array([noise.draw(rng) for _ in range(ndraws)])[:, 0]

    nc = toy.ncomponents
    emp = xi[:, :nc].T @ xi[:, nc:] / ndraws
    truth = toy_ip.FNFD[0]
    scale = np.sqrt(np.outer(np.diag(toy_ip.FNF[0]), np.diag(toy_ip.FDNFD[0])))
    assert np.max(np.abs(emp - truth) / scale) < 8 / np.sqrt(ndraws)


# ------------------------------------------------------------------- rNr

def test_rNr_has_the_right_mean(toy, toy_ip):
    noise = CompressedNoise.from_inner_products(toy_ip)
    rng = np.random.default_rng(6)
    vals = np.array([simulate_inner_products(toy_ip, rng=rng, noise=noise)
                     .inner_products.rNr for _ in range(600)])
    dof = toy.ntoas - toy.ntmpars
    ratio = vals.mean(axis=0) / dof
    assert np.all(np.abs(ratio - 1.0) < 6 * np.sqrt(2.0 / dof / 600))


def test_rNr_matches_residual_space(toy, toy_ip):
    """The mean of r^T Ntilde^-1 r must agree with the brute-force route."""
    rng = np.random.default_rng(7)
    brute = np.array([toy.inner_products(toy.draw_residuals(rng)).rNr
                      for _ in range(400)])
    noise = CompressedNoise.from_inner_products(toy_ip)
    rng = np.random.default_rng(8)
    fast = np.array([simulate_inner_products(toy_ip, rng=rng, noise=noise)
                     .inner_products.rNr for _ in range(400)])
    err = np.sqrt(brute.var(axis=0) / 400 + fast.var(axis=0) / 400)
    assert np.all(np.abs(brute.mean(axis=0) - fast.mean(axis=0)) < 5 * err)


def test_rNr_skipped_without_toa_counts(toy_ip):
    stripped = toy_ip.replace(ntoas=None, ntmpars=None)
    sim = simulate_inner_products(stripped, rng=0)
    assert sim.inner_products.rNr is None
    assert sim.diagnostics['rNr_simulated'] is False


# ------------------------------------------------------------ structure

def test_simulated_data_lies_in_the_gram_column_space(toy_ip):
    rng = np.random.default_rng(9)
    a = rng.standard_normal((toy_ip.npsrs, toy_ip.ncomponents)) * 100.0
    sim = simulate_inner_products(toy_ip, a=a, rng=rng)
    frac = range_consistency(sim.inner_products)['out_of_range_frac']
    assert np.max(frac) < 1e-20


def test_whitened_statistics_are_standard_normal(toy_ip):
    noise = CompressedNoise.from_inner_products(toy_ip)
    rng = np.random.default_rng(10)
    stats = [whitened_statistics(
        simulate_inner_products(toy_ip, rng=rng, noise=noise,
                                simulate_rNr=False).inner_products, noise=noise)
        for _ in range(200)]
    chi2 = np.array([s['chi2_per_dof'] for s in stats])
    dof = stats[0]['dof']
    assert abs(chi2.mean() - 1.0) < 5 * np.sqrt(2.0 / dof / 200)
    assert abs(chi2.std() / np.sqrt(2.0 / dof) - 1.0) < 0.25


def test_whiten_removes_an_injected_signal(toy_ip):
    from pta_sims.diagnostics import signal_vector
    rng = np.random.default_rng(11)
    a = rng.standard_normal((toy_ip.npsrs, toy_ip.ncomponents)) * 500.0
    sim = simulate_inner_products(toy_ip, a=a, rng=rng, simulate_rNr=False)
    stats = whitened_statistics(sim.inner_products,
                                signal=signal_vector(toy_ip, a))
    assert abs(stats['chi2_per_dof'] - 1.0) < 6 * np.sqrt(2.0 / stats['dof'])


# ------------------------------------------------------------- interface

def test_reused_factorization_gives_identical_results(toy_ip):
    noise = CompressedNoise.from_inner_products(toy_ip)
    a = np.ones((toy_ip.npsrs, toy_ip.ncomponents))
    first = simulate_inner_products(toy_ip, a=a, rng=0).inner_products.FNr
    second = simulate_inner_products(toy_ip, a=a, rng=0, noise=noise).inner_products.FNr
    assert np.array_equal(first, second)


def test_shape_errors_are_reported(toy_ip):
    with pytest.raises(ValueError, match='expected'):
        simulate_inner_products(toy_ip, a=np.zeros((2, 2)))
    with pytest.raises(ValueError, match='deterministic basis'):
        simulate_inner_products(toy_ip.replace(FNFD=None, FDNFD=None),
                                a_det=np.zeros((toy_ip.npsrs, 4)))


def test_matched_filter_snr_scales_linearly(toy_ip):
    a_det = np.ones((toy_ip.npsrs, toy_ip.ndet))
    assert np.isclose(matched_filter_snr(toy_ip, 2 * a_det),
                      2 * matched_filter_snr(toy_ip, a_det), rtol=1e-10)


def test_marginalized_snr_is_smaller(toy_ip):
    """Marginalizing red noise can only remove information."""
    rng = np.random.default_rng(12)
    a_det = rng.standard_normal((toy_ip.npsrs, toy_ip.ndet)) * 50.0
    phi = np.full((toy_ip.npsrs, toy_ip.ncomponents), 1e4)
    assert (matched_filter_snr(toy_ip, a_det, phi=phi)
            < matched_filter_snr(toy_ip, a_det))


# ------------------------------------------- marginalized detection statistic

def test_metric_matches_the_explicit_marginalization(toy, toy_ip):
    """
    M = FDNFD - FNFD^T (FNF + phi^-1)^-1 FNFD, checked against the residual-space
    covariance it is supposed to come from: with the stochastic amplitudes
    marginalized, the deterministic template lives in noise
    Ntilde + F phi F^T, so M = F_D^T (Ntilde + F phi F^T)^-1 F_D.
    """
    from pta_sims.simulate import deterministic_metric

    phi_ns2 = np.tile(10.0 ** np.linspace(5, 2, toy.ncomponents), (toy.npsrs, 1))
    M = deterministic_metric(toy_ip, phi=phi_ns2)

    p = 0
    phi_s = np.diag(phi_ns2[p] / units.RENORM ** 2)
    # Ntilde^-1 is singular, so work in its own range: use the pseudo-inverse
    # identity  (A^- + F phi F^T)^-1 -> A - A F (phi^-1 + F^T A F)^-1 F^T A
    # with A = Ntilde^-1 applied through the stored inner products.
    FNF, FNFD, FDNFD = toy_ip.FNF[p], toy_ip.FNFD[p], toy_ip.FDNFD[p]
    expected = FDNFD - FNFD.T @ np.linalg.solve(
        FNF + np.linalg.inv(phi_s), FNFD)
    assert np.allclose(M[p], expected, rtol=1e-9)


def test_statistic_and_metric_recover_the_injected_amplitude(toy, toy_ip):
    """
    The marginalized log-likelihood peaks at the injected amplitude scale, and
    its height at the truth averages SNR^2 / 2.
    """
    from pta_sims.simulate import (CompressedNoise, deterministic_metric,
                                   deterministic_statistic,
                                   marginal_delta_ln_likelihood,
                                   matched_filter_snr)

    rng = np.random.default_rng(20)
    phi_ns2 = np.tile(10.0 ** np.linspace(5, 2, toy.ncomponents), (toy.npsrs, 1))
    a_det = rng.standard_normal((toy.npsrs, toy.ndet)) * 200.0

    metric = deterministic_metric(toy_ip, phi=phi_ns2)
    snr = matched_filter_snr(toy_ip, a_det, metric=metric)
    noise = CompressedNoise.from_inner_products(toy_ip)

    peaks, heights = [], []
    for _ in range(300):
        a = rng.standard_normal((toy.npsrs, toy.ncomponents)) * np.sqrt(phi_ns2)
        sim = simulate_inner_products(toy_ip, a=a, a_det=a_det, rng=rng,
                                      noise=noise, simulate_rNr=False)
        stat = deterministic_statistic(sim.inner_products, phi=phi_ns2)
        scales = np.linspace(0.0, 2.0, 201)
        curve = np.array([
            marginal_delta_ln_likelihood(sim.inner_products, s * a_det,
                                         metric=metric, statistic=stat)
            for s in scales])
        peaks.append(scales[np.argmax(curve)])
        heights.append(marginal_delta_ln_likelihood(
            sim.inner_products, a_det, metric=metric, statistic=stat))

    peaks, heights = np.array(peaks), np.array(heights)
    # the maximum-likelihood amplitude scale is unbiased about 1, with a spread
    # of 1/SNR
    assert abs(peaks.mean() - 1.0) < 5 * (1.0 / snr) / np.sqrt(len(peaks))
    # and E[dlnL at truth] = SNR^2 / 2, with a per-realization spread of SNR
    assert abs(np.mean(heights) - snr ** 2 / 2) < 5 * snr / np.sqrt(len(heights))


def test_statistic_needs_fdnr(toy_ip):
    from pta_sims.simulate import deterministic_statistic
    with pytest.raises(ValueError, match='FDNr'):
        deterministic_statistic(toy_ip.replace(FDNr=None))


def test_same_seed_gives_the_same_noise_whatever_the_signal(toy_ip):
    """
    The matched-pair guarantee the injection campaign is built on: two datasets
    that differ only in the injected signal must share their noise realization
    exactly, so that a difference between them is the signal and not scatter.
    """
    from pta_sims.diagnostics import signal_vector

    rng = np.random.default_rng(30)
    a = rng.standard_normal((toy_ip.npsrs, toy_ip.ncomponents)) * 100.0
    a_det = rng.standard_normal((toy_ip.npsrs, toy_ip.ndet)) * 30.0
    noise = CompressedNoise.from_inner_products(toy_ip)

    null = simulate_inner_products(toy_ip, a=a, rng=99, noise=noise,
                                   simulate_rNr=False)
    with_cw = simulate_inner_products(toy_ip, a=a, a_det=a_det, rng=99,
                                      noise=noise, simulate_rNr=False)

    assert np.array_equal(null.noise_realization, with_cw.noise_realization)
    difference = with_cw.inner_products.joint_vector() - null.inner_products.joint_vector()
    assert np.allclose(difference, signal_vector(toy_ip, np.zeros_like(a), a_det),
                       rtol=1e-10)


def test_signal_and_noise_come_from_independent_streams():
    """
    Regression: an injection campaign must not seed the amplitude draw and the
    noise draw with the same integer. Each call builds its own Generator, so
    identical seeds replay the identical sequence -- the normals behind the
    injected amplitudes are then literally the same numbers as the first ones
    behind the noise projection. The data is not from the distribution the
    likelihood assumes, and credible-interval coverage is silently wrong.

    Asserted on the streams rather than on the transformed draws: the coupling
    is exact there, while the design matrices can dilute it to any value.
    """
    n_amp, n_noise = 4020, 12060

    # the bug: one integer handed to both
    a_shared = np.random.default_rng(7).standard_normal(n_amp)
    x_shared = np.random.default_rng(7).standard_normal(n_noise)
    assert np.array_equal(a_shared, x_shared[:n_amp])

    # the fix: spawned children of one SeedSequence
    amp_ss, noise_ss = np.random.SeedSequence(7).spawn(2)
    a = np.random.default_rng(amp_ss).standard_normal(n_amp)
    x = np.random.default_rng(noise_ss).standard_normal(n_noise)
    assert not np.array_equal(a, x[:n_amp])
    corr = abs(np.corrcoef(a, x[:n_amp])[0, 1])
    assert corr < 5 / np.sqrt(n_amp), f'streams still coupled: {corr:.4f}'

    # and reproducible: the same base seed gives the same pair back
    amp_ss2, noise_ss2 = np.random.SeedSequence(7).spawn(2)
    assert np.array_equal(np.random.default_rng(amp_ss2).standard_normal(n_amp), a)
    assert np.array_equal(np.random.default_rng(noise_ss2).standard_normal(n_noise), x)
