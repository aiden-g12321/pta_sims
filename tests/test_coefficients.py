"""Drawing Fourier amplitudes from the spectral prior."""

import numpy as np

from pta_sims.coefficients import draw_from_blocks, draw_from_phi_cube


def _phi_cube(nc=8, npsr=5, seed=0):
    """Per-pulsar variance plus a correlated common process."""
    rng = np.random.default_rng(seed)
    corr = rng.standard_normal((npsr, npsr))
    corr = corr @ corr.T / npsr + np.eye(npsr)
    power = 10.0 ** np.linspace(4, 1, nc)
    common = 0.4 * power
    diag = np.outer(power, 0.5 + rng.random(npsr))
    cube = common[:, None, None] * corr[None] \
        + diag[:, :, None] * np.eye(npsr)[None]
    return cube


def test_draw_reproduces_the_prior_covariance():
    cube = _phi_cube()
    rng = np.random.default_rng(1)
    ndraws = 20000
    draws = np.array([draw_from_phi_cube(cube, rng=rng)[0] for _ in range(ndraws)])

    for c in (0, 3, 7):
        emp = draws[:, :, c].T @ draws[:, :, c] / ndraws
        scale = np.sqrt(np.outer(np.diag(cube[c]), np.diag(cube[c])))
        assert np.max(np.abs(emp - cube[c]) / scale) < 8 / np.sqrt(ndraws)


def test_components_are_independent():
    cube = _phi_cube()
    rng = np.random.default_rng(2)
    draws = np.array([draw_from_phi_cube(cube, rng=rng)[0] for _ in range(20000)])
    p = 0
    corr = np.corrcoef(draws[:, p, :].T)
    off = corr - np.eye(corr.shape[0])
    assert np.max(np.abs(off)) < 8 / np.sqrt(20000)


def test_zero_variance_components_give_zero_amplitudes():
    """A GWB covering fewer frequencies than the data leaves exact zeros."""
    cube = _phi_cube()
    cube[5:] = 0.0
    a, info = draw_from_phi_cube(cube, rng=0)
    assert np.all(a[:, 5:] == 0.0)
    assert np.all(info['rank'][5:] == 0)


def test_blocks_route_matches_the_cube_route_in_distribution():
    """
    With a component-diagonal prior the Kronecker route must reproduce the
    per-component route.
    """
    npsr, nc = 4, 6
    rng = np.random.default_rng(3)
    corr = rng.standard_normal((npsr, npsr))
    corr = corr @ corr.T / npsr + np.eye(npsr)
    common = np.diag(10.0 ** np.linspace(3, 1, nc))
    per = np.stack([np.diag(10.0 ** np.linspace(2.5, 0.5, nc))] * npsr)

    cube = np.einsum('cc,ij->cij', common, corr) \
        + np.einsum('pcc,pq->cpq', per, np.eye(npsr))

    ndraws = 20000
    rng = np.random.default_rng(4)
    d1 = np.array([draw_from_phi_cube(cube, rng=rng)[0] for _ in range(ndraws)])
    rng = np.random.default_rng(5)
    d2 = np.array([draw_from_blocks(per, corr, common, rng=rng)[0]
                   for _ in range(ndraws)])

    c1 = np.cov(d1.reshape(ndraws, -1).T)
    c2 = np.cov(d2.reshape(ndraws, -1).T)
    scale = np.sqrt(np.outer(np.diag(c1), np.diag(c1)))
    assert np.max(np.abs(c1 - c2) / scale) < 12 / np.sqrt(ndraws)


def test_blocks_route_carries_inter_frequency_correlation():
    npsr, nc = 3, 4
    per = np.stack([np.full((nc, nc), 50.0) + np.eye(nc) * 50.0] * npsr)
    rng = np.random.default_rng(6)
    draws = np.array([draw_from_blocks(per, rng=rng)[0] for _ in range(20000)])
    emp = np.cov(draws[:, 0, :].T)
    assert np.max(np.abs(emp - per[0]) / 100.0) < 10 / np.sqrt(20000)
