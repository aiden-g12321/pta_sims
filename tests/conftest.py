"""
A small synthetic pulsar array built from explicit design matrices.

The point of building ``F``, ``F_D``, ``N`` and ``M`` by hand is that the tests
can then compare the compressed simulation against a brute-force simulation in
*residual* space -- draw white noise, project it, and take the inner products --
which is the thing ``pta_sims`` claims to be able to skip.  Nothing in the
fixture uses the package under test.

The deterministic frequency grid deliberately overlaps the stochastic one, so
the stacked basis is exactly rank deficient and the tests exercise the
singular-covariance path rather than a well-conditioned special case.
"""

import numpy as np
import pytest


class ToyArray:
    """Explicit design matrices and noise for a few pulsars."""

    def __init__(self, npsrs=4, ntoas=180, nfreqs=5, nfreqs_det=8, ntmpars=3,
                 seed=0):
        rng = np.random.default_rng(seed)
        self.npsrs = npsrs
        self.nfreqs = nfreqs
        self.ncomponents = 2 * nfreqs
        self.ndet = 2 * nfreqs_det
        self.ntoas = np.full(npsrs, ntoas)
        self.ntmpars = np.full(npsrs, ntmpars)

        Tspan = 10.0 * 365.25 * 86400.0
        self.F, self.F_D, self.M, self.N, self.Ninv_tilde = [], [], [], [], []

        for p in range(npsrs):
            toas = np.sort(rng.uniform(0.0, Tspan, ntoas))

            F = np.empty((ntoas, self.ncomponents))
            for k in range(nfreqs):
                f = (k + 1) / Tspan
                F[:, 2 * k] = np.sin(2 * np.pi * f * toas)
                F[:, 2 * k + 1] = np.cos(2 * np.pi * f * toas)

            # extended window, as Prometheus does for deterministic signals:
            # every 5th frequency coincides with a stochastic one, which makes
            # the stacked basis exactly degenerate
            F_D = np.empty((ntoas, self.ndet))
            for k in range(nfreqs_det):
                f = (k + 1) / (5.0 * Tspan)
                F_D[:, 2 * k] = np.sin(2 * np.pi * f * toas)
                F_D[:, 2 * k + 1] = np.cos(2 * np.pi * f * toas)

            M = np.stack([np.ones(ntoas), toas / Tspan, (toas / Tspan) ** 2],
                         axis=1)[:, :ntmpars]

            sigma = 1e-6 * (0.5 + rng.random(ntoas))
            N = np.diag(sigma ** 2)
            Ninv = np.diag(sigma ** -2)
            MNM = M.T @ Ninv @ M
            Ninv_tilde = Ninv - Ninv @ M @ np.linalg.solve(MNM, M.T @ Ninv)

            self.F.append(F)
            self.F_D.append(F_D)
            self.M.append(M)
            self.N.append(N)
            self.Ninv_tilde.append(Ninv_tilde)

    def inner_products(self, residuals=None):
        """Build an :class:`InnerProducts` from the explicit matrices."""
        from pta_sims.inner_products import InnerProducts

        FNF, FNr, FNFD, FDNFD, FDNr, rNr = [], [], [], [], [], []
        for p in range(self.npsrs):
            F, F_D, Ni = self.F[p], self.F_D[p], self.Ninv_tilde[p]
            r = np.zeros(F.shape[0]) if residuals is None else residuals[p]
            FNF.append(F.T @ Ni @ F)
            FNFD.append(F.T @ Ni @ F_D)
            FDNFD.append(F_D.T @ Ni @ F_D)
            FNr.append(F.T @ Ni @ r)
            FDNr.append(F_D.T @ Ni @ r)
            rNr.append(r @ Ni @ r)

        return InnerProducts(
            psr_names=[f'PSR{p}' for p in range(self.npsrs)],
            FNF=np.stack(FNF), FNr=np.stack(FNr), FNFD=np.stack(FNFD),
            FDNFD=np.stack(FDNFD), FDNr=np.stack(FDNr), rNr=np.array(rNr),
            ntoas=self.ntoas, ntmpars=self.ntmpars)

    def draw_residuals(self, rng, a_ns=None, a_det_ns=None):
        """Brute-force residuals ``r = F a + F_D a_det + n`` in seconds."""
        out = []
        for p in range(self.npsrs):
            n = rng.multivariate_normal(np.zeros(self.N[p].shape[0]), self.N[p])
            r = n.copy()
            if a_ns is not None:
                r += self.F[p] @ (a_ns[p] / 1e9)
            if a_det_ns is not None:
                r += self.F_D[p] @ (a_det_ns[p] / 1e9)
            out.append(r)
        return out


@pytest.fixture(scope='session')
def toy():
    return ToyArray()


@pytest.fixture(scope='session')
def toy_ip(toy):
    return toy.inner_products()
