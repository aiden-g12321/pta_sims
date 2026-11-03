"""Posterior summary and sky localization helpers."""

import numpy as np

from pta_sims.diagnostics import posterior_summary, sky_summary


def test_summary_reports_quantiles_and_spread():
    rng = np.random.default_rng(0)
    chain = rng.normal(2.0, 0.5, 20000)
    out = posterior_summary(chain)

    assert np.isclose(out['mean'], 2.0, atol=0.02)
    assert np.isclose(out['std'], 0.5, rtol=0.03)
    assert out['quantiles']['q5'] < out['quantiles']['q50'] < out['quantiles']['q95']
    assert 'truth' not in out


def test_summary_measures_accuracy_against_a_truth():
    rng = np.random.default_rng(1)
    chain = rng.normal(2.0, 0.5, 20000)

    centred = posterior_summary(chain, truth=2.0)
    assert abs(centred['z_score']) < 0.1
    assert centred['covered_90']
    assert np.isclose(centred['truth_quantile'], 0.5, atol=0.02)

    offset = posterior_summary(chain, truth=3.5)     # 3 sigma out
    assert offset['z_score'] < -2.5
    assert not offset['covered_90']
    assert offset['truth_quantile'] > 0.99


def test_truth_quantile_is_uniform_over_realizations():
    """
    The calibration check the campaign relies on: for a correctly centred
    posterior the truth's quantile is uniform, so its mean is 0.5.
    """
    rng = np.random.default_rng(2)
    quantiles = [posterior_summary(rng.normal(truth, 1.0, 2000),
                                   truth=truth)['truth_quantile']
                 for truth in rng.normal(0.0, 3.0, 300)]
    assert abs(np.mean(quantiles) - 0.5) < 0.05


def test_sky_summary_localizes_a_concentrated_posterior():
    rng = np.random.default_rng(3)
    true_cos, true_phi = 0.4, 2.0
    cos_theta = np.clip(rng.normal(true_cos, 0.02, 40000), -1, 1)
    phi = np.mod(rng.normal(true_phi, 0.02, 40000), 2 * np.pi)

    out = sky_summary(cos_theta, phi, truth=(true_cos, true_phi))
    assert out['offset_deg'] < 2.0
    assert out['area90_deg2'] < out['area50_deg2'] * 4
    assert out['area90_deg2'] < 500
    assert out['concentration'] > 0.99
    assert out['truth_credible_level'] < 0.5


def test_sky_summary_on_an_isotropic_posterior():
    rng = np.random.default_rng(4)
    cos_theta = rng.uniform(-1, 1, 60000)
    phi = rng.uniform(0, 2 * np.pi, 60000)

    out = sky_summary(cos_theta, phi, truth=(0.3, 1.0))
    whole_sky = 4 * np.pi * (180 / np.pi) ** 2
    assert out['area90_deg2'] > 0.7 * whole_sky
    assert out['concentration'] < 0.05


def test_sky_summary_credible_area_grows_with_the_level():
    rng = np.random.default_rng(5)
    cos_theta = np.clip(rng.normal(0.0, 0.15, 40000), -1, 1)
    phi = np.mod(rng.normal(3.0, 0.15, 40000), 2 * np.pi)
    out = sky_summary(cos_theta, phi)
    assert out['area50_deg2'] < out['area90_deg2']
