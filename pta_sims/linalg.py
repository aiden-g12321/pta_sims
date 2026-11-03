"""
Numerically careful linear algebra for rank-deficient PSD matrices.

Why not Cholesky
----------------
The object we have to sample from is the Gram matrix of the *stacked* design
matrix ``B = [F | F_D]`` in the noise metric,

    C = B^T Ntilde^-1 B = [[ F^T Ntilde^-1 F ,  F^T Ntilde^-1 F_D  ],
                           [ F_D^T Ntilde^-1 F, F_D^T Ntilde^-1 F_D]],

and on real data this matrix is *badly* rank deficient.  Two independent
mechanisms are responsible.

1.  ``Ntilde^-1`` is the white noise inverse projected orthogonal to the timing
    model, ``Ntilde^-1 = N^-1 - N^-1 M (M^T N^-1 M)^-1 M^T N^-1``.  It
    annihilates every direction the timing fit absorbs.  For a pulsar whose
    own baseline is much shorter than the PTA span, the lowest Fourier modes
    are nearly a quadratic in time and are therefore very nearly annihilated,
    so ``F^T Ntilde^-1 F`` acquires eigenvalues that fall to the float64
    rounding floor.  On the NG15 array ``J1312+0051`` has a numerical rank of
    24 out of 60.

2.  ``F`` and ``F_D`` are not linearly independent.  ``F`` samples frequencies
    ``k / Tspan``; ``F_D`` samples ``j / Tspan_ext`` on a window extended by
    ``det_window_ext_factor``, and both are evaluated at the *same* TOAs.
    Whenever the two grids come close the stacked basis is near-degenerate, so
    the joint ``C`` is far more rank deficient than either block alone (rank
    24-61 out of 180 across NG15).

Adding diagonal jitter until a Cholesky succeeds -- the obvious fix -- is
statistically wrong here, not merely inelegant.  The exact data satisfies

    B^T Ntilde^-1 r  in  col(C)

identically (write ``Ntilde^-1 = N^-1/2 P N^-1/2`` with ``P`` an orthogonal
projector and ``X = P N^-1/2 B``; then ``C = X^T X`` and
``B^T Ntilde^-1 r = X^T (N^-1/2 r)``, which lies in ``col(X^T) = col(C)``).
Jitter injects variance into directions that must carry *exactly* zero, so the
simulated data violates linear identities that the real data obeys exactly.

The eigendecomposition-based factor below is exact for singular PSD matrices,
respects those identities, and costs nothing at these sizes (a 180x180 ``eigh``
per pulsar).
"""

import numpy as np

DEFAULT_RTOL = 1e-12


def symmetrize(A):
    """Return the symmetric part of ``A``, killing accumulated asymmetry."""
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def psd_factor(C, rtol=DEFAULT_RTOL):
    """
    Square-root factor of a (possibly singular) positive semi-definite matrix.

    Returns ``L`` with ``L L^T = C_+``, where ``C_+`` is ``C`` with its
    numerically-null eigen-directions removed.  ``L`` keeps the full ``(n, n)``
    shape with zero columns where directions were dropped, so that ``L @ z``
    with ``z ~ N(0, I_n)`` is an exact draw from ``N(0, C_+)`` and the whole
    thing batches over a leading pulsar axis.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        Symmetric PSD matrix (or a stack of them), float64.
    rtol : float
        Eigenvalues below ``rtol * lambda_max`` are treated as zero.  The
        default 1e-12 sits a few orders of magnitude above the float64 rounding
        floor of an ``eigh`` on these matrices (~1e-15 relative), which is where
        the spurious near-zero eigenvalues of a rank-deficient Gram matrix pile
        up.

    Returns
    -------
    L : ndarray, shape (..., n, n)
    info : dict
        ``rank``           -- number of retained directions (per matrix),
        ``max_eig``        -- largest eigenvalue,
        ``min_kept_eig``   -- smallest retained eigenvalue,
        ``min_eig``        -- smallest eigenvalue (may be negative),
        ``neg_frac``       -- ``-min_eig / max_eig``, a float64 health check;
                              values above ~1e-13 mean the input matrix itself
                              is no longer numerically PSD,
        ``dropped_trace_frac`` -- fraction of the trace discarded.
    """
    C = symmetrize(C)
    eigvals, eigvecs = np.linalg.eigh(C)                    # ascending

    max_eig = eigvals[..., -1]
    threshold = rtol * np.maximum(max_eig, 0.0)
    keep = eigvals > threshold[..., None]

    kept_vals = np.where(keep, eigvals, 0.0)
    L = eigvecs * np.sqrt(kept_vals)[..., None, :]

    total_trace = np.sum(np.abs(eigvals), axis=-1)
    info = {
        'rank': np.sum(keep, axis=-1),
        'max_eig': max_eig,
        'min_eig': eigvals[..., 0],
        'min_kept_eig': np.min(np.where(keep, eigvals, np.inf), axis=-1),
        'neg_frac': np.where(max_eig > 0,
                             -np.minimum(eigvals[..., 0], 0.0) / np.maximum(max_eig, 1e-300),
                             0.0),
        'dropped_trace_frac': np.where(
            total_trace > 0,
            np.sum(np.where(keep, 0.0, np.abs(eigvals)), axis=-1) / np.maximum(total_trace, 1e-300),
            0.0),
    }
    return L, info


def psd_pinv_quadform(C, x, rtol=DEFAULT_RTOL):
    """
    Evaluate ``x^T C^+ x`` with ``C^+`` the Moore-Penrose pseudo-inverse.

    Used for the chi-squared decomposition of ``r^T Ntilde^-1 r``; see
    :func:`pta_sims.simulate.simulate_inner_products`.

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
    x : ndarray, shape (..., n)
    rtol : float
        Same truncation rule as :func:`psd_factor`.

    Returns
    -------
    ndarray, shape (...)
    """
    C = symmetrize(C)
    x = np.asarray(x, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(C)
    threshold = rtol * np.maximum(eigvals[..., -1], 0.0)
    keep = eigvals > threshold[..., None]
    proj = np.einsum('...ij,...i->...j', eigvecs, x)
    inv = np.where(keep, 1.0 / np.where(keep, eigvals, 1.0), 0.0)
    return np.sum(proj ** 2 * inv, axis=-1)


def psd_solve_factor(C, rtol=DEFAULT_RTOL):
    """
    Factor of the pseudo-inverse: ``W`` with ``W W^T = C^+``.

    Useful for whitening a simulated data vector, ``W^T x ~ N(0, I_rank)`` when
    ``x ~ N(0, C)``.  Zero columns mark the dropped directions.
    """
    C = symmetrize(C)
    eigvals, eigvecs = np.linalg.eigh(C)
    threshold = rtol * np.maximum(eigvals[..., -1], 0.0)
    keep = eigvals > threshold[..., None]
    inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, eigvals, 1.0)), 0.0)
    return eigvecs * inv_sqrt[..., None, :]


def block_matrix(A, B, D):
    """
    Assemble ``[[A, B], [B^T, D]]`` for a batch of blocks.

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
    B : ndarray, shape (..., n, m)
    D : ndarray, shape (..., m, m)

    Returns
    -------
    ndarray, shape (..., n + m, n + m)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    top = np.concatenate([A, B], axis=-1)
    bottom = np.concatenate([np.swapaxes(B, -1, -2), D], axis=-1)
    return np.concatenate([top, bottom], axis=-2)


def condition_report(C):
    """
    Summarize the eigen-spectrum of a stack of symmetric matrices.

    Returns a dict of arrays over the leading axis: ``max_eig``, ``min_eig``,
    ``cond`` (ratio of largest to smallest strictly positive eigenvalue), and
    ``numerical_rank`` at the default tolerance.
    """
    eigvals = np.linalg.eigvalsh(symmetrize(C))
    max_eig = eigvals[..., -1]
    keep = eigvals > DEFAULT_RTOL * np.maximum(max_eig, 0.0)[..., None]
    min_kept = np.min(np.where(keep, eigvals, np.inf), axis=-1)
    return {
        'max_eig': max_eig,
        'min_eig': eigvals[..., 0],
        'min_kept_eig': min_kept,
        'cond': max_eig / min_kept,
        'numerical_rank': np.sum(keep, axis=-1),
    }
