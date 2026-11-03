"""Factorization of singular positive semi-definite matrices."""

import numpy as np
import pytest

from pta_sims.linalg import (block_matrix, condition_report, psd_factor,
                             psd_pinv_quadform, psd_solve_factor, symmetrize)


def _singular_psd(n=12, rank=5, scale=1e12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rank, n)) * np.sqrt(scale)
    return X.T @ X


def test_factor_reproduces_matrix_on_its_range():
    C = _singular_psd()
    L, info = psd_factor(C)
    assert info['rank'] == 5
    assert np.allclose(L @ L.T, C, rtol=0, atol=1e-8 * np.max(np.abs(C)))


def test_factor_kills_the_null_space_exactly():
    """A draw must have *zero* variance outside the column space."""
    C = _singular_psd()
    L, _ = psd_factor(C)
    eigvals, eigvecs = np.linalg.eigh(symmetrize(C))
    null = eigvecs[:, eigvals <= 1e-12 * eigvals[-1]]

    rng = np.random.default_rng(1)
    draws = (L @ rng.standard_normal((C.shape[0], 4000))).T
    assert np.max(np.abs(draws @ null)) < 1e-9 * np.max(np.abs(draws))


def test_factor_batches_over_leading_axis():
    C = np.stack([_singular_psd(seed=s) for s in range(3)])
    L, info = psd_factor(C)
    assert L.shape == C.shape
    assert info['rank'].shape == (3,)
    for p in range(3):
        assert np.allclose(L[p] @ L[p].T, C[p], rtol=0,
                           atol=1e-8 * np.max(np.abs(C[p])))


def test_factor_tolerates_negative_rounding_eigenvalues():
    """Real Gram matrices come back from eigh with small negative eigenvalues."""
    C = _singular_psd()
    C = C - 1e-4 * np.eye(C.shape[0])          # push the null space negative
    L, info = psd_factor(C)
    assert np.all(np.isfinite(L))
    assert info['rank'] == 5
    assert info['neg_frac'] > 0


def test_pinv_quadform_matches_numpy_pinv():
    C = _singular_psd()
    rng = np.random.default_rng(2)
    x = C @ rng.standard_normal(C.shape[0])     # in the column space
    assert np.isclose(psd_pinv_quadform(C, x), x @ np.linalg.pinv(C) @ x,
                      rtol=1e-8)


def test_solve_factor_whitens():
    C = _singular_psd()
    L, info = psd_factor(C)
    W = psd_solve_factor(C)
    rng = np.random.default_rng(3)
    draws = (L @ rng.standard_normal((C.shape[0], 40000))).T
    whitened = draws @ W
    emp = whitened.T @ whitened / draws.shape[0]
    kept = np.linalg.norm(W, axis=0) > 0
    assert np.allclose(np.diag(emp)[kept], 1.0, atol=0.05)
    assert np.allclose(np.diag(emp)[~kept], 0.0)


def test_block_matrix_layout():
    A = np.arange(4.).reshape(1, 2, 2)
    B = np.arange(6.).reshape(1, 2, 3)
    D = np.arange(9.).reshape(1, 3, 3)
    C = block_matrix(A, B, D)
    assert C.shape == (1, 5, 5)
    assert np.array_equal(C[0, :2, :2], A[0])
    assert np.array_equal(C[0, :2, 2:], B[0])
    assert np.array_equal(C[0, 2:, :2], B[0].T)
    assert np.array_equal(C[0, 2:, 2:], D[0])


def test_condition_report_finds_the_rank():
    C = np.stack([_singular_psd(seed=s) for s in range(2)])
    rep = condition_report(C)
    assert np.all(rep['numerical_rank'] == 5)
    assert np.all(np.isfinite(rep['cond']))
