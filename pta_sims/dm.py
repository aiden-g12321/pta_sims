"""
Dispersion measure (DM) Gaussian processes.

A DM Gaussian process lives on its own Fourier basis,

    F_dm = F * (fref / nu)^2,

the usual Fourier basis scaled per TOA by the cold-plasma dispersion law, with
``nu`` the radio frequency of the TOA.  Prometheus' generative model becomes

    r = F a + F_dm a_dm + F_D a_det + n,

so the DM process is nothing but extra stochastic basis columns.  Everything in
:mod:`pta_sims.simulate` therefore carries over unchanged once the stochastic
basis is *augmented* to ``[F | F_dm]``: the compressed noise covariance is the
Gram matrix of the stacked basis ``[F | F_dm | F_D]``, exactly as before, and
the three channels share their white noise through the off-diagonal blocks.

The extra inner products, built by ``prometheus.data.Data(..., nfreqs_dm=...)``:

    FNFdm   = F^T    Ntilde^-1 F_dm     (Np, Nc, Ndm)   Data.FNFdms
    FdmNFdm = F_dm^T Ntilde^-1 F_dm     (Np, Ndm, Ndm)  Data.FdmNFdms
    FdmNFD  = F_dm^T Ntilde^-1 F_D      (Np, Ndm, Nd)   Data.FdmNTDs
    FdmNr   = F_dm^T Ntilde^-1 r        (Np, Ndm)       Data.FdmNrs

Only ``FdmNr`` depends on the data realization.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import units
from .coefficients import _as_rng
from .inner_products import InnerProducts
from .linalg import DEFAULT_RTOL, psd_factor


@dataclass
class DMInnerProducts:
    """
    DM-basis inner products in natural units and float64.

    Attributes
    ----------
    FNFdm : ndarray, shape (Np, Nc, Ndm)
    FdmNFdm : ndarray, shape (Np, Ndm, Ndm)
    FdmNr : ndarray, shape (Np, Ndm)
    FdmNFD : ndarray, shape (Np, Ndm, Nd) or None
    """

    FNFdm: np.ndarray
    FdmNFdm: np.ndarray
    FdmNr: np.ndarray
    FdmNFD: Optional[np.ndarray] = None

    @property
    def ncomponents(self):
        """Number of DM Fourier basis columns (``2 * nfreqs_dm``)."""
        return self.FdmNFdm.shape[-1]

    @classmethod
    def from_data(cls, data):
        """
        Extract the float64 DM inner products from a Prometheus ``Data`` object
        built with ``nfreqs_dm > 0``.  As in
        :meth:`InnerProducts.from_data`, the float64 originals in
        ``data.per_psr_data_dict`` are preferred over the float32 copies.
        """
        if not getattr(data, 'has_dm', False):
            raise ValueError('This Data object has no DM basis; build it with '
                             'prometheus.data.Data(..., nfreqs_dm=...).')
        names = list(data.psr_names)
        psd = getattr(data, 'per_psr_data_dict', None)
        if psd is not None:
            def stack(key):
                return np.stack([np.asarray(psd[n][key], dtype=np.float64) for n in names])
            return cls(FNFdm=stack('TNFdm'), FdmNFdm=stack('FdmNFdm'),
                       FdmNr=stack('FdmNr'), FdmNFD=stack('FdmNTD'))
        return cls(FNFdm=units.from_renorm(np.asarray(data.FNFdms), 2),
                   FdmNFdm=units.from_renorm(np.asarray(data.FdmNFdms), 2),
                   FdmNr=units.from_renorm(np.asarray(data.FdmNrs), 1),
                   FdmNFD=units.from_renorm(np.asarray(data.FdmNTDs), 2))


def augment(inner_products, dm_inner_products):
    """
    Fold the DM basis into the stochastic basis: ``F -> [F | F_dm]``.

    Parameters
    ----------
    inner_products : InnerProducts
    dm_inner_products : DMInnerProducts

    Returns
    -------
    InnerProducts
        With ``ncomponents = Nc + Ndm``.  Hand it to
        :func:`pta_sims.simulate.simulate_inner_products` or
        :class:`pta_sims.simulate.CompressedNoise` exactly like the original;
        undo with :func:`split`.
    """
    ip, dm = inner_products, dm_inner_products
    FNF = np.asarray(ip.FNF, dtype=np.float64)
    FNFdm = np.asarray(dm.FNFdm, dtype=np.float64)
    FNF_aug = np.concatenate([
        np.concatenate([FNF, FNFdm], axis=2),
        np.concatenate([np.swapaxes(FNFdm, 1, 2), dm.FdmNFdm], axis=2)], axis=1)
    FNr_aug = np.concatenate([ip.FNr, dm.FdmNr], axis=1)
    FNFD_aug = None
    if ip.has_deterministic:
        if dm.FdmNFD is None:
            raise ValueError('The inner products carry a deterministic basis but '
                             'the DM inner products have no FdmNFD block.')
        FNFD_aug = np.concatenate([ip.FNFD, dm.FdmNFD], axis=1)
    return ip.replace(FNF=FNF_aug, FNr=FNr_aug, FNFD=FNFD_aug)


def split(augmented, ncomponents):
    """
    Inverse of :func:`augment`.

    Parameters
    ----------
    augmented : InnerProducts
    ncomponents : int
        Number of columns of the original stochastic basis ``F``.

    Returns
    -------
    (InnerProducts, DMInnerProducts)
    """
    nc = ncomponents
    ip = augmented.replace(
        FNF=augmented.FNF[:, :nc, :nc],
        FNr=augmented.FNr[:, :nc],
        FNFD=None if augmented.FNFD is None else augmented.FNFD[:, :nc])
    dm = DMInnerProducts(
        FNFdm=augmented.FNF[:, :nc, nc:],
        FdmNFdm=augmented.FNF[:, nc:, nc:],
        FdmNr=augmented.FNr[:, nc:],
        FdmNFD=None if augmented.FNFD is None else augmented.FNFD[:, nc:])
    return ip, dm


def augmented_inner_products(data):
    """
    ``InnerProducts.from_data(data)`` with the DM basis folded in when the
    ``Data`` object carries one.  Build a reusable
    :class:`~pta_sims.simulate.CompressedNoise` for a model with a DM Gaussian
    process from this.
    """
    ip = InnerProducts.from_data(data)
    if getattr(data, 'has_dm', False):
        ip = augment(ip, DMInnerProducts.from_data(data))
    return ip


def draw_dm_amplitudes(pta_model, params, rng=None, rtol=DEFAULT_RTOL):
    """
    Draw DM Fourier amplitudes from the model's DM Gaussian process prior.

    Uses the DM model's own ``get_phi_func`` (dense, so inter-frequency
    correlations are included when the model has them), i.e. exactly the
    prior covariance the sampler evaluates.

    Parameters
    ----------
    pta_model : prometheus.pta_model.PTAModel
        Must carry a ``dm_model``.
    params : dict
        Must contain ``params[pta_model.dm_model.name]``, shape (Np, nparams).
    rng : numpy Generator, int, or None
    rtol : float

    Returns
    -------
    a_dm : ndarray, shape (Np, Ndm), float64
        Amplitudes in nanoseconds (delays at the reference radio frequency).
    info : dict
        Factorization diagnostics.
    """
    from jax import vmap
    import jax.numpy as jnp

    rng = _as_rng(rng)
    dm_model = pta_model.dm_model
    if dm_model.name not in params:
        raise KeyError(f"Missing DM Gaussian process parameters under key "
                       f"'{dm_model.name}'. Supplied keys: {sorted(params)}.")
    dm_params = jnp.asarray(np.asarray(params[dm_model.name], dtype=np.float64))
    # the model's own basis frequencies, which is what the sampler evaluates
    # the spectrum on; data.freqs_dm is the same grid rounded differently
    phi = np.asarray(vmap(dm_model.get_phi_func, in_axes=(0, None))(
        dm_params, dm_model.freqs), dtype=np.float64)            # (Np, Ndm, Ndm)

    L, info = psd_factor(phi, rtol=rtol)
    z = rng.standard_normal(phi.shape[:-1])
    return np.einsum('pij,pj->pi', L, z), info


def install(data, dm_inner_products):
    """
    Write simulated DM residual inner products into a ``Data`` object.

    Modifies ``data`` in place: meant for the fresh copy returned by
    :func:`pta_sims.datasets.rebuild_data`, whose per-pulsar dictionary entries
    are already copies of the template's.
    """
    from .datasets import _to_array

    dm = dm_inner_products
    data.FdmNrs = _to_array(units.to_renorm(dm.FdmNr, 1).astype(units.PROMETHEUS_DTYPE),
                            getattr(data, 'FdmNrs', None))
    psd = getattr(data, 'per_psr_data_dict', None)
    if psd is not None:
        for i, n in enumerate(data.psr_names):
            psd[n]['FdmNr'] = np.asarray(dm.FdmNr[i], dtype=np.float64)
    return data
