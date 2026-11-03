"""
DM Gaussian processes: the DM basis ``F_dm = F * (fref / nu)^2`` is folded into
the stochastic basis, so the compressed simulation must agree with compressing
brute-force residuals ``r = F a + F_dm a_dm + F_D a_det + n``.

As in the other tests, the reference side uses only the toy array's explicit
matrices.
"""

import numpy as np
import pytest

from pta_sims import dm
from pta_sims.inner_products import InnerProducts
from pta_sims.simulate import CompressedNoise, simulate_inner_products


@pytest.fixture(scope='module')
def toy_dm(toy):
    """Radio frequencies and the explicit DM basis for every toy pulsar."""
    rng = np.random.default_rng(11)
    F_dm = []
    for p in range(toy.npsrs):
        # two observing bands, as for most PTA pulsars
        nu = np.where(rng.random(toy.F[p].shape[0]) < 0.5,
                      rng.uniform(700., 900., toy.F[p].shape[0]),
                      rng.uniform(1300., 1700., toy.F[p].shape[0]))
        F_dm.append(toy.F[p] * ((1400. / nu) ** 2)[:, None])
    return F_dm


def _dm_inner_products(toy, F_dm, residuals=None):
    out = {'FNFdm': [], 'FdmNFdm': [], 'FdmNFD': [], 'FdmNr': []}
    for p in range(toy.npsrs):
        Ni = toy.Ninv_tilde[p]
        r = np.zeros(Ni.shape[0]) if residuals is None else residuals[p]
        out['FNFdm'].append(toy.F[p].T @ Ni @ F_dm[p])
        out['FdmNFdm'].append(F_dm[p].T @ Ni @ F_dm[p])
        out['FdmNFD'].append(F_dm[p].T @ Ni @ toy.F_D[p])
        out['FdmNr'].append(F_dm[p].T @ Ni @ r)
    return dm.DMInnerProducts(**{k: np.stack(v) for k, v in out.items()})


def test_augment_split_round_trip(toy, toy_ip, toy_dm):
    dm_ip = _dm_inner_products(toy, toy_dm)
    aug = dm.augment(toy_ip, dm_ip)
    assert aug.ncomponents == toy_ip.ncomponents + dm_ip.ncomponents

    # the augmented joint matrix is the Gram matrix of [F | F_dm | F_D]
    p = 2
    B = np.concatenate([toy.F[p], toy_dm[p], toy.F_D[p]], axis=1)
    expected = B.T @ toy.Ninv_tilde[p] @ B
    assert np.max(np.abs(aug.joint_matrix()[p] - expected)) < 1e-12 * np.max(np.abs(expected))

    ip_back, dm_back = dm.split(aug, toy_ip.ncomponents)
    np.testing.assert_array_equal(ip_back.FNF, toy_ip.FNF)
    np.testing.assert_array_equal(ip_back.FNFD, toy_ip.FNFD)
    np.testing.assert_array_equal(dm_back.FdmNFdm, dm_ip.FdmNFdm)
    np.testing.assert_array_equal(dm_back.FNFdm, dm_ip.FNFdm)
    np.testing.assert_array_equal(dm_back.FdmNFD, dm_ip.FdmNFD)


def test_noiseless_dm_injection_is_exact(toy, toy_ip, toy_dm):
    """With the white noise off, FdmNr = F_dm^T Ntilde^-1 (F a + F_dm a_dm + F_D a_det)."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 50., (toy.npsrs, toy.ncomponents))
    a_dm = rng.normal(0, 200., (toy.npsrs, toy.ncomponents))
    a_det = rng.normal(0, 20., (toy.npsrs, toy.ndet))

    aug = dm.augment(toy_ip, _dm_inner_products(toy, toy_dm))
    sim = simulate_inner_products(aug, a=np.concatenate([a, a_dm], 1), a_det=a_det,
                                  include_white_noise=False, simulate_rNr=False)
    ip_sim, dm_sim = dm.split(sim.inner_products, toy.ncomponents)

    r = [toy.F[p] @ a[p] / 1e9 + toy_dm[p] @ a_dm[p] / 1e9 + toy.F_D[p] @ a_det[p] / 1e9
         for p in range(toy.npsrs)]
    exact = toy.inner_products(r)
    exact_dm = _dm_inner_products(toy, toy_dm, r)
    scale = np.max(np.abs(exact_dm.FdmNr))
    assert np.max(np.abs(dm_sim.FdmNr - exact_dm.FdmNr)) < 1e-10 * scale
    assert np.max(np.abs(ip_sim.FNr - exact.FNr)) < 1e-10 * np.max(np.abs(exact.FNr))
    assert np.max(np.abs(ip_sim.FDNr - exact.FDNr)) < 1e-10 * np.max(np.abs(exact.FDNr))


def test_dm_noise_matches_brute_force(toy, toy_ip, toy_dm):
    """
    The compressed white noise on [F | F_dm | F_D] has the same distribution
    as compressing brute-force white noise, compared in the whitened basis.
    """
    ndraws = 6000
    p = 1
    aug = dm.augment(toy_ip, _dm_inner_products(toy, toy_dm))
    noise = CompressedNoise.from_inner_products(aug)
    B = np.concatenate([toy.F[p], toy_dm[p], toy.F_D[p]], axis=1)

    rng = np.random.default_rng(4)
    n = rng.multivariate_normal(np.zeros(toy.N[p].shape[0]), toy.N[p], size=ndraws)
    brute = n @ toy.Ninv_tilde[p] @ B

    rng = np.random.default_rng(5)
    fast = np.array([
        simulate_inner_products(aug, rng=rng, noise=noise,
                                simulate_rNr=False).inner_products.joint_vector()[p]
        for _ in range(ndraws)])

    inv_sqrt = np.where(noise.keep[p],
                        1 / np.sqrt(np.where(noise.keep[p], noise.eigvals[p], 1.)), 0.)
    W = noise.eigvecs[p] * inv_sqrt
    target = np.diag((np.linalg.norm(W, axis=0) > 0).astype(float))
    tol = 8 * np.sqrt(2.0 / ndraws)
    for x in (brute, fast):
        c = (x @ W).T @ (x @ W) / ndraws
        assert np.max(np.abs(c - target)) < tol
