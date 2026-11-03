"""
The compressed representation of a PTA dataset: noise-weighted inner products.

Prometheus never touches TOAs during sampling.  Everything it needs is a small
set of inner products taken in the metric ``Ntilde^-1``, the white-noise
inverse projected orthogonal to the linearized timing model:

    FNF   = F^T   Ntilde^-1 F       (Np, Nc, Nc)   stochastic basis Gram matrix
    FNr   = F^T   Ntilde^-1 r       (Np, Nc)       stochastic basis vs residuals
    FNFD  = F^T   Ntilde^-1 F_D     (Np, Nc, Nd)   stochastic vs deterministic
    FDNFD = F_D^T Ntilde^-1 F_D     (Np, Nd, Nd)   deterministic Gram matrix
    FDNr  = F_D^T Ntilde^-1 r       (Np, Nd)       deterministic vs residuals
    rNr   = r^T   Ntilde^-1 r       (Np,)          residual norm

``F`` is the Fourier design matrix of the stochastic processes (sine/cosine
interleaved at frequencies ``k / Tspan``), ``F_D`` the Fourier design matrix
used for deterministic signals (frequencies ``j / Tspan_ext`` on the extended
window that suppresses Gibbs ringing), and ``r`` the timing residuals.

In Prometheus' ``Data`` object these appear under the names ``FNFs``, ``FNrs``,
``TNTDs``, ``TDNTDs`` and ``TDNrs``, stored in float32 and rescaled by
``renorm``; see :mod:`pta_sims.units`.  This module keeps them in **natural
units and float64**, which is where all of the simulation arithmetic happens.

Of these six objects only the three involving ``r`` -- ``FNr``, ``FDNr`` and
``rNr`` -- depend on the data realization.  The other three are fixed by the
TOA sampling, the timing model and the white noise, and are exactly what makes
"simulate an NG15-like dataset with my own injection" a cheap operation: keep
``FNF``, ``FNFD``, ``FDNFD`` from the real array and redraw the rest.
"""

from dataclasses import dataclass, replace as _dc_replace
from typing import Optional, Sequence

import numpy as np

from . import units
from .linalg import block_matrix, condition_report


@dataclass
class InnerProducts:
    """
    Noise-weighted inner products for a PTA, in natural units and float64.

    Attributes
    ----------
    psr_names : sequence of str
        Pulsar names, in the array order used by every other attribute.
    FNF : ndarray, shape (Np, Nc, Nc)
    FNr : ndarray, shape (Np, Nc)
    FNFD : ndarray, shape (Np, Nc, Nd) or None
    FDNFD : ndarray, shape (Np, Nd, Nd) or None
    FDNr : ndarray, shape (Np, Nd) or None
    rNr : ndarray, shape (Np,) or None
        Optional; Prometheus 0.1.x does not use it, but it completes the
        sufficient statistics and later versions carry it as an additive
        log-likelihood constant.
    ntoas : ndarray, shape (Np,) or None
        TOA count per pulsar.  Only needed to simulate ``rNr``.
    ntmpars : ndarray, shape (Np,) or None
        Number of timing model basis columns projected out per pulsar.  Only
        needed to simulate ``rNr``.
    """

    psr_names: Sequence[str]
    FNF: np.ndarray
    FNr: np.ndarray
    FNFD: Optional[np.ndarray] = None
    FDNFD: Optional[np.ndarray] = None
    FDNr: Optional[np.ndarray] = None
    rNr: Optional[np.ndarray] = None
    ntoas: Optional[np.ndarray] = None
    ntmpars: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- shape
    @property
    def npsrs(self):
        return self.FNF.shape[0]

    @property
    def ncomponents(self):
        """Number of stochastic Fourier basis columns (``2 * nfreqs``)."""
        return self.FNF.shape[-1]

    @property
    def nfreqs(self):
        return self.ncomponents // 2

    @property
    def ndet(self):
        """Number of deterministic Fourier basis columns, 0 if absent."""
        return 0 if self.FDNFD is None else self.FDNFD.shape[-1]

    @property
    def has_deterministic(self):
        return self.FNFD is not None and self.FDNFD is not None

    # ----------------------------------------------------------- assembly
    def joint_matrix(self):
        """
        The stacked Gram matrix ``[F | F_D]^T Ntilde^-1 [F | F_D]``.

        Returns ``FNF`` itself when there is no deterministic basis.

        Returns
        -------
        ndarray, shape (Np, Nc + Nd, Nc + Nd)
        """
        if not self.has_deterministic:
            return np.asarray(self.FNF, dtype=np.float64)
        return block_matrix(self.FNF, self.FNFD, self.FDNFD)

    def joint_vector(self):
        """The stacked data vector ``[F | F_D]^T Ntilde^-1 r``."""
        if not self.has_deterministic:
            return np.asarray(self.FNr, dtype=np.float64)
        return np.concatenate([self.FNr, self.FDNr], axis=-1)

    def replace(self, **kwargs):
        """Return a copy with the named fields replaced."""
        return _dc_replace(self, **kwargs)

    # ------------------------------------------------------- renormalized
    def renormalized(self, dtype=units.PROMETHEUS_DTYPE):
        """
        Return the arrays in Prometheus' renormalized units and dtype.

        Returns
        -------
        dict
            Keyed by the Prometheus ``Data`` attribute names: ``FNFs``,
            ``FNrs``, ``TNTDs``, ``TDNTDs``, ``TDNrs``.
        """
        out = {
            'FNFs': units.to_renorm(self.FNF, 2).astype(dtype),
            'FNrs': units.to_renorm(self.FNr, 1).astype(dtype),
        }
        if self.has_deterministic:
            out['TNTDs'] = units.to_renorm(self.FNFD, 2).astype(dtype)
            out['TDNTDs'] = units.to_renorm(self.FDNFD, 2).astype(dtype)
            if self.FDNr is not None:
                out['TDNrs'] = units.to_renorm(self.FDNr, 1).astype(dtype)
        return out

    # ------------------------------------------------------------ loading
    @classmethod
    def from_data(cls, data, with_rNr=True):
        """
        Extract the float64 inner products from a Prometheus ``Data`` object.

        The float64 originals live in ``data.per_psr_data_dict`` (keys ``TNT``,
        ``TNr``, ``TNTD``, ``TDNTD``, ``TDNr``); they are used in preference to
        the float32 copies on the ``Data`` object itself, so nothing is ever
        derived from a single-precision array.

        Parameters
        ----------
        data : prometheus.data.Data
        with_rNr : bool
            Also record TOA and timing-model-column counts, which are what
            makes ``rNr`` simulatable.  Requires ``data.psrs`` to still be
            attached; silently skipped otherwise.

        Returns
        -------
        InnerProducts
        """
        names = list(data.psr_names)
        psd = getattr(data, 'per_psr_data_dict', None)

        if psd is not None:
            def stack(key):
                return np.stack([np.asarray(psd[n][key], dtype=np.float64) for n in names])

            FNF = stack('TNT')
            FNr = stack('TNr')
            has_det = 'TDNTD' in psd[names[0]]
            FNFD = stack('TNTD') if has_det else None
            FDNFD = stack('TDNTD') if has_det else None
            FDNr = stack('TDNr') if has_det else None
        else:
            # Fall back to the stored float32 arrays.  This loses the low-order
            # bits, so it is only a convenience for stripped-down data objects.
            FNF = units.from_renorm(np.asarray(data.FNFs), 2)
            FNr = units.from_renorm(np.asarray(data.FNrs), 1)
            has_det = hasattr(data, 'TDNTDs')
            FNFD = units.from_renorm(np.asarray(data.TNTDs), 2) if has_det else None
            FDNFD = units.from_renorm(np.asarray(data.TDNTDs), 2) if has_det else None
            FDNr = units.from_renorm(np.asarray(data.TDNrs), 1) if has_det else None

        ntoas = ntmpars = None
        if with_rNr and getattr(data, 'psrs', None):
            by_name = {p.name: p for p in data.psrs}
            if all(n in by_name for n in names):
                ntoas = np.array([len(by_name[n].toas) for n in names])
                ntmpars = np.array([by_name[n].Mmat.shape[1] for n in names])

        return cls(psr_names=names, FNF=FNF, FNr=FNr, FNFD=FNFD,
                   FDNFD=FDNFD, FDNr=FDNr, ntoas=ntoas, ntmpars=ntmpars)

    # ------------------------------------------------------------- saving
    def save(self, path, residuals_only=True, **metadata):
        """
        Write to a compressed ``.npz``.

        Parameters
        ----------
        path : str
        residuals_only : bool
            Store only the realization-dependent arrays -- ``FNr``, ``FDNr``,
            ``rNr`` -- which is ~100 kB for a 67-pulsar array against ~14 MB for
            the full set.  The Gram matrices are a property of the array, not of
            the dataset, so an injection study that varies only the signal
            should keep one copy of them and reload it with
            :meth:`load`'s ``reference`` argument.  Set False to write a
            self-contained file.
        **metadata
            Extra arrays or scalars to store alongside, e.g. injected
            parameters.  Retrievable from :meth:`load`.

        Returns
        -------
        str
            The path written.
        """
        payload = {
            'psr_names': np.asarray(list(self.psr_names), dtype=object),
            'FNr': np.asarray(self.FNr, dtype=np.float64),
            'residuals_only': np.array(residuals_only),
        }
        if self.FDNr is not None:
            payload['FDNr'] = np.asarray(self.FDNr, dtype=np.float64)
        if self.rNr is not None:
            payload['rNr'] = np.asarray(self.rNr, dtype=np.float64)
        if self.ntoas is not None:
            payload['ntoas'] = np.asarray(self.ntoas)
            payload['ntmpars'] = np.asarray(self.ntmpars)
        if not residuals_only:
            payload['FNF'] = np.asarray(self.FNF, dtype=np.float64)
            if self.has_deterministic:
                payload['FNFD'] = np.asarray(self.FNFD, dtype=np.float64)
                payload['FDNFD'] = np.asarray(self.FDNFD, dtype=np.float64)
        for key, value in metadata.items():
            payload[f'meta_{key}'] = np.asarray(value)

        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path, reference=None):
        """
        Read a file written by :meth:`save`.

        Parameters
        ----------
        path : str
        reference : InnerProducts, optional
            Supplies the Gram matrices for a residuals-only file.  Required in
            that case, and its pulsar ordering must match.

        Returns
        -------
        (inner_products, metadata) : tuple
        """
        with np.load(path, allow_pickle=True) as handle:
            stored = {k: handle[k] for k in handle.files}

        names = [str(n) for n in stored['psr_names']]
        metadata = {k[5:]: v for k, v in stored.items() if k.startswith('meta_')}

        if bool(stored['residuals_only']):
            if reference is None:
                raise ValueError(
                    f'{path} stores only the residual-dependent arrays; pass '
                    'reference=InnerProducts.from_data(base_data) to supply the '
                    'Gram matrices.')
            if list(reference.psr_names) != names:
                raise ValueError('Pulsar ordering of the reference does not '
                                 'match the stored file.')
            base = reference
        else:
            base = cls(psr_names=names, FNF=stored['FNF'], FNr=stored['FNr'],
                       FNFD=stored.get('FNFD'), FDNFD=stored.get('FDNFD'))

        inner = cls(
            psr_names=names,
            FNF=base.FNF, FNFD=base.FNFD, FDNFD=base.FDNFD,
            FNr=stored['FNr'],
            FDNr=stored.get('FDNr'),
            rNr=stored.get('rNr'),
            ntoas=stored.get('ntoas'),
            ntmpars=stored.get('ntmpars'))
        return inner, metadata

    # -------------------------------------------------------- diagnostics
    def conditioning(self):
        """
        Eigen-spectrum summary of ``FNF`` and of the joint Gram matrix.

        Returns
        -------
        dict of dict
            ``{'FNF': {...}, 'joint': {...}}`` -- see
            :func:`pta_sims.linalg.condition_report`.  Expect the joint matrix
            to be far more rank deficient than ``FNF``; that is physical, not a
            bug (see :mod:`pta_sims.linalg`).
        """
        out = {'FNF': condition_report(self.FNF)}
        if self.has_deterministic:
            out['joint'] = condition_report(self.joint_matrix())
        return out
