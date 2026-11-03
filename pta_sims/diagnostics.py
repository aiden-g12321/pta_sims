"""
Checks that a simulated dataset is what it claims to be.

Three independent things can go wrong, and each has its own check here.

*The derivation.*  :func:`range_consistency` tests the structural identity that
motivates the whole approach -- that ``[F | F_D]^T Ntilde^-1 r`` lies in the
column space of the joint Gram matrix.  Run it on the *real* NG15 data and it
either holds to rounding error, in which case the compressed noise covariance
really is the Gram matrix, or it does not, in which case the model of the
compression is wrong.  Jittered-Cholesky simulations fail this test; the
eigen-truncated draw passes it by construction.

*The sampler.*  :func:`noise_covariance_check` draws many noise realizations
and compares their empirical covariance against the Gram matrix, restricted to
the retained subspace.  This catches factorization bugs.

*The precision.*  :func:`whitened_statistics` reduces a dataset to the
standard-normal variables it should consist of, and the ``downcast`` entry of a
simulation's diagnostics reports what the final float32 cast cost.
"""

import numpy as np

from . import units
from .linalg import DEFAULT_RTOL
from .sampling import effective_sample_size


def range_consistency(inner_products, rtol=DEFAULT_RTOL, noise=None):
    """
    Check that the data vector lies in the column space of the Gram matrix.

    With ``Ntilde^-1 = N^-1/2 P N^-1/2`` and ``X = P N^-1/2 [F | F_D]``, the
    Gram matrix is ``C = X^T X`` and the data vector is ``X^T (N^-1/2 r)``,
    which lies in ``col(X^T) = col(C)`` identically.  The fraction of the data
    vector's norm outside that column space is therefore a pure numerical
    quantity: it should sit at the level of the truncation tolerance, for real
    and simulated data alike.

    Parameters
    ----------
    inner_products : InnerProducts
    rtol : float
    noise : CompressedNoise, optional
        Pre-built factorization to reuse.

    Returns
    -------
    dict
        ``out_of_range_frac`` -- per pulsar, the fraction of ``|x|^2`` lying
        outside ``col(C)``; ``rank`` and ``size`` of the Gram matrix.
    """
    from .simulate import CompressedNoise

    if noise is None:
        noise = CompressedNoise.from_inner_products(inner_products, rtol=rtol)
    x = inner_products.joint_vector()

    proj = np.einsum('pij,pi->pj', noise.eigvecs, x)
    inside = np.sum(np.where(noise.keep, proj ** 2, 0.0), axis=-1)
    total = np.sum(proj ** 2, axis=-1)

    return {
        'out_of_range_frac': np.where(total > 0, 1.0 - inside / total, 0.0),
        'rank': noise.rank,
        'size': noise.size,
    }


def noise_covariance_check(inner_products, psr_index=0, ndraws=20000, seed=0,
                           rtol=DEFAULT_RTOL):
    """
    Compare the empirical covariance of the noise draw against the Gram matrix.

    Both are projected onto the retained subspace before comparison, since the
    null directions carry exactly zero variance in both.

    Parameters
    ----------
    inner_products : InnerProducts
    psr_index : int
    ndraws : int
    seed : int
    rtol : float

    Returns
    -------
    dict
        ``max_rel_err`` -- largest deviation relative to the matrix norm;
        ``expected_mc_err`` -- the Monte Carlo floor, ``sqrt(2 / ndraws)``.
        The check passes when the first is comparable to the second.
    """
    from .linalg import psd_factor, psd_solve_factor

    C = inner_products.joint_matrix()[psr_index]
    L, info = psd_factor(C, rtol=rtol)
    rank = int(info['rank'])

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((ndraws, C.shape[-1]))
    xi = z @ L.T
    emp = xi.T @ xi / ndraws

    W = psd_solve_factor(C, rtol=rtol)                  # C^{+/2}
    whitened = W.T @ emp @ W                            # should be I on the rank
    target = np.zeros_like(whitened)
    idx = np.arange(whitened.shape[0])
    kept = np.linalg.norm(W, axis=0) > 0
    target[idx[kept], idx[kept]] = 1.0

    return {
        'rank': rank,
        'max_rel_err': float(np.max(np.abs(whitened - target))),
        'expected_mc_err': float(np.sqrt(2.0 / ndraws) * 4.0),
        'ndraws': ndraws,
    }


def whitened_statistics(inner_products, signal=None, rtol=DEFAULT_RTOL,
                        noise=None):
    """
    Reduce a dataset to the standard normal variables it should consist of.

    Parameters
    ----------
    inner_products : InnerProducts
    signal : array_like, shape (Np, Nc + Nd), optional
        Injected signal ``C v`` in natural units, removed before whitening.
        Omit to measure how far the data departs from pure white noise, which
        is the natural sanity check on real data.
    rtol : float
    noise : CompressedNoise, optional
        Pre-built factorization to reuse.

    Returns
    -------
    dict
        ``mean``, ``std`` and ``chi2_per_dof`` of the whitened components over
        the retained directions.  For a correctly simulated white-noise-only
        dataset all three sit at 0, 1 and 1.
    """
    from .simulate import CompressedNoise

    if noise is None:
        noise = CompressedNoise.from_inner_products(inner_products, rtol=rtol)
    x = inner_products.joint_vector()
    if signal is not None:
        x = x - np.asarray(signal, dtype=np.float64)
    w = noise.whiten(x)
    dof = int(np.sum(noise.rank))

    vals = w[np.abs(w) > 0] if dof else np.array([0.0])
    return {
        'dof': dof,
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'chi2_per_dof': float(np.sum(w ** 2) / dof) if dof else np.nan,
    }


def signal_vector(inner_products, a, a_det=None, noise=None):
    """
    The noiseless data vector ``C v`` for injected amplitudes, natural units.

    Parameters
    ----------
    inner_products : InnerProducts
    a : array_like, shape (Np, Nc)
        Stochastic amplitudes in nanoseconds.
    a_det : array_like, shape (Np, Nd), optional
        Deterministic amplitudes in nanoseconds.
    noise : CompressedNoise, optional
        Pre-built factorization, whose cached Gram matrix is reused.

    Returns
    -------
    ndarray, shape (Np, Nc + Nd)
    """
    v = np.asarray(a, dtype=np.float64)
    if inner_products.has_deterministic:
        if a_det is None:
            a_det = np.zeros((inner_products.npsrs, inner_products.ndet))
        v = np.concatenate([v, np.asarray(a_det, dtype=np.float64)], axis=-1)
    C = inner_products.joint_matrix() if noise is None else noise.C
    return np.einsum('pij,pj->pi', C, units.ns_to_seconds(v))


def format_report(sim):
    """
    Render a :class:`~pta_sims.api.SimulatedDataset` as a short text report.

    Parameters
    ----------
    sim : SimulatedDataset

    Returns
    -------
    str
    """
    d = sim.diagnostics
    ip = sim.inner_products
    lines = [
        f'Simulated dataset "{sim.data.name}"',
        f'  pulsars {ip.npsrs}, stochastic components {ip.ncomponents}, '
        f'deterministic components {ip.ndet}',
        f'  white noise included: {d.get("include_white_noise")}',
    ]

    rank = np.asarray(d.get('joint_rank', []))
    if rank.size:
        lines.append(
            f'  joint Gram rank {rank.min()}-{rank.max()} of {d["joint_size"]} '
            f'(truncation rtol {d["rtol"]:g})')
        lines.append(
            f'  largest negative-eigenvalue fraction '
            f'{np.max(d["joint_neg_frac"]):.2e}  '
            f'(float64 rounding floor ~1e-15; >1e-13 means the stored Gram '
            f'matrix is itself no longer numerically PSD)')
        lines.append(
            f'  trace discarded by truncation '
            f'{np.max(d["joint_dropped_trace_frac"]):.2e}')

    if sim.truth.get('snr'):
        lines.append(f'  deterministic SNR {sim.truth["snr"]:.2f} '
                     f'(white noise only), '
                     f'{sim.truth["snr_marginalized"]:.2f} '
                     f'(stochastic processes marginalized)')

    lines.append('  float32 downcast into Data:')
    for key, rep in d.get('downcast', {}).items():
        lines.append(
            f'    {key:<8s} max rel err {rep["max_rel_err"]:.2e}, '
            f'{rep["n_nonfinite"]} non-finite, '
            f'{rep["n_underflow"]} underflowed to zero '
            f'(range {rep["abs_min_nonzero"]:.2e} .. {rep["abs_max"]:.2e})')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# posterior summaries
#
# Generic enough to be worth keeping out of any one study: quantiles and a
# truth-relative accuracy for a scalar parameter, and a solid-angle
# localization for a sky position.
# ---------------------------------------------------------------------------

QUANTILES = (5, 16, 50, 84, 95)


def posterior_summary(chain, truth=None):
    """Quantiles, spread, effective sample size and truth-relative accuracy."""
    chain = np.asarray(chain, dtype=np.float64)
    quantiles = np.percentile(chain, QUANTILES)
    out = {
        'mean': float(chain.mean()),
        'std': float(chain.std(ddof=1)),
        'quantiles': {f'q{q}': float(v) for q, v in zip(QUANTILES, quantiles)},
        'ess': float(effective_sample_size(chain)),
    }
    if truth is not None:
        truth = float(truth)
        out['truth'] = truth
        out['bias'] = out['mean'] - truth
        out['z_score'] = ((out['mean'] - truth) / out['std']
                          if out['std'] > 0 else np.nan)
        # where the truth falls in the posterior; 0.5 is centred, 0 or 1 is a
        # complete miss. This is the quantity that should be uniform over
        # realizations if the posterior is calibrated.
        out['truth_quantile'] = float(np.mean(chain < truth))
        out['covered_90'] = bool(quantiles[0] <= truth <= quantiles[-1])
    return out


def sky_summary(cos_theta, phi, truth=None, nside_theta=60, nside_phi=120):
    """
    Sky localization from the posterior samples.

    ``(cos_theta, phi)`` is an equal-area coordinate pair, so a uniform grid in
    it has cells of equal solid angle and the credible area is just a cell
    count.
    """
    counts, theta_edges, phi_edges = np.histogram2d(
        cos_theta, phi, bins=[nside_theta, nside_phi],
        range=[[-1.0, 1.0], [0.0, 2 * np.pi]])
    total = counts.sum()
    if total == 0:
        return {}

    flat = np.sort(counts.ravel())[::-1]
    cumulative = np.cumsum(flat) / total
    cell_area = (2.0 / nside_theta) * (2 * np.pi / nside_phi)
    to_deg2 = (180.0 / np.pi) ** 2

    out = {}
    for level in (0.5, 0.9):
        n_cells = int(np.searchsorted(cumulative, level) + 1)
        out[f'area{int(level * 100)}_deg2'] = float(n_cells * cell_area * to_deg2)

    sin_theta = np.sqrt(np.clip(1 - cos_theta ** 2, 0, 1))
    vectors = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi),
                        cos_theta], axis=-1)
    mean_vector = vectors.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    out['concentration'] = float(norm)          # 0 = isotropic, 1 = a point

    if truth is not None and norm > 0:
        true_cos, true_phi = truth
        true_sin = np.sqrt(max(1 - true_cos ** 2, 0))
        true_vector = np.array([true_sin * np.cos(true_phi),
                                true_sin * np.sin(true_phi), true_cos])
        out['offset_deg'] = float(np.degrees(np.arccos(np.clip(
            mean_vector @ true_vector / norm, -1, 1))))
        # does the credible region actually contain the source?
        i = min(np.searchsorted(theta_edges, true_cos) - 1, nside_theta - 1)
        j = min(np.searchsorted(phi_edges, true_phi) - 1, nside_phi - 1)
        rank = int(np.sum(counts.ravel() > counts[i, j]))
        out['truth_credible_level'] = float(
            np.sum(np.sort(counts.ravel())[::-1][:rank + 1]) / total)
    return out
