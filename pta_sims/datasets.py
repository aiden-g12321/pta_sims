"""
Reading, rebuilding and saving Prometheus ``Data`` objects.

A Prometheus ``Data`` object holds three kinds of thing:

* the pulsars themselves (``data.psrs``) -- large, and unchanged by simulation;
* geometry fixed by the TOAs, timing model and white noise (``FNFs``,
  ``TNTDs``, ``TDNTDs``, the Fourier frequencies, the sparse deterministic
  grid, pulsar sky positions and distances) -- also unchanged;
* everything that depends on the *realized residuals* (``FNrs``, ``TDNrs`` and
  the several cached quantities derived from them).

Simulating a dataset therefore means rebuilding only the third group, which is
what :func:`rebuild_data` does.  It shallow-copies the ``Data`` object so that
the 2 GB of pulsar objects behind ``data.psrs`` are shared rather than
duplicated, and rewrites the residual-dependent fields consistently -- not just
``FNrs`` and ``TDNrs``, but also the maximum-a-posteriori Fourier amplitudes
``a_hat`` and its several cached forms, which are stale the moment ``FNrs``
changes and would otherwise silently describe the old dataset.
"""

import copy
import pickle

import numpy as np
import scipy.linalg as sla

from . import units


def load_data(path):
    """
    Unpickle a Prometheus ``Data`` object.

    Parameters
    ----------
    path : str

    Returns
    -------
    prometheus.data.Data
    """
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def save_data(data, path, include_pulsars=True):
    """
    Pickle a ``Data`` object.

    Parameters
    ----------
    data : prometheus.data.Data
    path : str
    include_pulsars : bool
        When False the ``psrs`` list is emptied first, which takes the NG15
        object from ~2 GB to a few tens of MB.  A stripped object still drives
        the sampler, but it can no longer be used to rebuild inner products for
        a different white noise dictionary
        (:mod:`pta_sims.whitenoise`), so keep the original around.
    """
    if not include_pulsars:
        data = strip_pulsars(data)
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def strip_pulsars(data):
    """Return a shallow copy of ``data`` with ``psrs`` emptied."""
    stripped = copy.copy(data)
    stripped.psrs = []
    return stripped


def _to_array(x, like=None):
    """
    Convert to the array type and dtype a Prometheus ``Data`` object stores.

    ``Data.__init__`` casts every floating-point array it holds to float32,
    the precision its posterior is evaluated in.  Prometheus also turns JAX's
    x64 mode on globally when it is imported, so ``jnp.asarray`` no longer
    performs that cast on the way in, and the dtype a rebuilt field ends up
    with would otherwise depend on whether Prometheus happened to be imported
    first.  Cast explicitly instead: a simulated dataset must carry float32
    wherever a real one does, or the two disagree the first time a jitted
    model sees both.

    ``like`` is accepted for callers that pass the field being replaced; the
    array module is chosen the same way with or without it.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        x = x.astype(units.PROMETHEUS_DTYPE)
    try:
        import jax.numpy as jnp
        return jnp.asarray(x)
    except ImportError:
        return x


def _basis_names(data):
    """
    Names of the stochastic Fourier bases a ``Data`` object carries.

    ``Data.basis_names`` reads the per-pulsar dictionary for objects pickled
    before bases were named, which can fail on a stripped-down or hand-built
    one; such an object has the achromatic basis and nothing else.
    """
    try:
        return list(data.basis_names)
    except Exception:
        return ['red']


def rebuild_data(data, inner_products, name=None):
    """
    Return a new ``Data`` object carrying the given inner products.

    Every residual-dependent field is recomputed; every other field is shared
    with the input object by reference.  The Gram matrices ``FNFs``, ``TNTDs``
    and ``TDNTDs`` are also rewritten, so this works both for a pure
    re-injection (same geometry, new residuals) and for a dataset whose white
    noise was changed (see :mod:`pta_sims.whitenoise`).

    Parameters
    ----------
    data : prometheus.data.Data
        Template object.  Not modified.
    inner_products : pta_sims.inner_products.InnerProducts
        Natural-unit float64 inner products to install.  Pulsar ordering must
        match ``data.psr_names``.
    name : str, optional
        Name for the new dataset.  Defaults to ``data.name + '_sim'``.

    Returns
    -------
    prometheus.data.Data
    """
    ip = inner_products
    names = list(data.psr_names)
    if list(ip.psr_names) != names:
        raise ValueError('Pulsar ordering of the inner products does not match '
                         'the Data object.')

    # Each extra Fourier basis carries its own residual inner product
    # F_b^T Ntilde^-1 r in the per-pulsar dictionary, and Prometheus'
    # marginalized sampling model reads every basis through data.gram() and
    # data.basis_Nr() rather than through FNrs.  pta_sims can re-simulate the
    # DM basis (see pta_sims.dm) but not an arbitrary one, and a basis left
    # untouched would quietly keep the template's real residuals alongside the
    # simulated ones.
    unknown = [b for b in _basis_names(data) if b not in ('red', 'dm')]
    if unknown:
        raise NotImplementedError(
            f'This Data object carries Fourier bases {unknown}, whose '
            'residual inner products pta_sims does not know how to simulate; '
            "they would keep the template's real data. Only the 'red' and "
            "'dm' bases are supported.")

    new = copy.copy(data)
    new.name = name if name is not None else f'{data.name}_sim'

    renorm = ip.renormalized()
    new.FNFs = _to_array(renorm['FNFs'], getattr(data, 'FNFs', None))
    new.FNrs = _to_array(renorm['FNrs'], getattr(data, 'FNrs', None))
    if ip.has_deterministic:
        new.TNTDs = _to_array(renorm['TNTDs'], getattr(data, 'TNTDs', None))
        new.TDNTDs = _to_array(renorm['TDNTDs'], getattr(data, 'TDNTDs', None))
        if 'TDNrs' in renorm:
            new.TDNrs = _to_array(renorm['TDNrs'], getattr(data, 'TDNrs', None))

    psd = getattr(data, 'per_psr_data_dict', None)
    if psd is None:
        return new

    # ------------------------------------------------------------------
    # Rebuild the per-pulsar dictionary and the cached MAP amplitudes.
    #
    # a_hat = Sigma_inv^-1 FNr with Sigma_inv = FNF + diag(phiinv) evaluated at
    # the fixed reference power law Prometheus uses to regularize the cache.
    # Sigma_inv (and its Cholesky Li) depend only on the geometry, so when the
    # white noise is unchanged the existing factor is reused; when FNF has been
    # rebuilt it is refactored here.
    # ------------------------------------------------------------------
    refactor = not np.allclose(
        np.stack([np.asarray(psd[n]['TNT'], dtype=np.float64) for n in names]),
        ip.FNF, rtol=1e-12, atol=0.0)

    new_psd = {}
    a_hats = []
    for i, n in enumerate(names):
        entry = dict(psd[n])
        entry['TNr'] = ip.FNr[i]
        if ip.has_deterministic and ip.FDNr is not None:
            entry['TDNr'] = ip.FDNr[i]
        if ip.rNr is not None:
            entry['rNr'] = float(ip.rNr[i])

        if refactor:
            entry['TNT'] = ip.FNF[i]
            if ip.has_deterministic:
                entry['TNTD'] = ip.FNFD[i]
                entry['TDNTD'] = ip.FDNFD[i]
            Sigma_inv = ip.FNF[i] + np.diag(np.asarray(entry['phiinv'], dtype=np.float64))
            Li = sla.cholesky(Sigma_inv, lower=True)
            entry['Sigma_inv'] = Sigma_inv
            entry['Li'] = Li
            entry['Sigma'] = sla.cho_solve((Li, True), np.identity(len(Li)))
            entry['logdet'] = -2.0 * np.sum(np.log(np.diag(Li)))
        else:
            Li = np.asarray(entry['Li'], dtype=np.float64)

        a_hat = sla.cho_solve((Li, True), np.asarray(ip.FNr[i], dtype=np.float64))
        entry['a_hat'] = a_hat
        a_hats.append(a_hat)
        new_psd[n] = entry

    new.per_psr_data_dict = new_psd

    a_hats = np.stack(a_hats)                                   # (Np, Nc), seconds
    Sigma_inv_stack = np.stack(
        [np.asarray(new_psd[n]['Sigma_inv'], dtype=np.float64) for n in names])

    new.a_hat_j = _to_array(np.concatenate(a_hats).astype(np.float32),
                            getattr(data, 'a_hat_j', None))
    new.a_hat_2d_pf = _to_array((a_hats * units.RENORM).astype(np.float32),
                                getattr(data, 'a_hat_2d_pf', None))
    # Sigma_inv @ a_hat = FNr identically, so this is FNr in renormalized units
    new.Si0_a_hat_j_pf = _to_array(
        (np.einsum('pij,pj->pi', Sigma_inv_stack / units.RENORM ** 2,
                   a_hats * units.RENORM)).astype(np.float32),
        getattr(data, 'Si0_a_hat_j_pf', None))

    if refactor:
        new.Sigma_0_inv_jc = _to_array(
            (Sigma_inv_stack / units.RENORM ** 2).astype(np.float32),
            getattr(data, 'Sigma_0_inv_jc', None))
        new.Sigma_0_inv_j = _to_array(
            sla.block_diag(*[(Sigma_inv_stack[i] / units.RENORM ** 2).astype(np.float32)
                             for i in range(len(names))]),
            getattr(data, 'Sigma_0_inv_j', None))
        new.Sigma_0_logdet_j = _to_array(
            np.sum([new_psd[n]['logdet'] for n in names]),
            getattr(data, 'Sigma_0_logdet_j', None))

    new.Si0_a_hat_j = _to_array(
        np.asarray(np.asarray(new.Sigma_0_inv_j, dtype=np.float64)
                   @ np.concatenate(a_hats).astype(np.float64)
                   * units.RENORM).astype(np.float32),
        getattr(data, 'Si0_a_hat_j', None))

    if ip.rNr is not None:
        # not part of Prometheus 0.1.x's Data, but carried for completeness:
        # rNr has no basis index, so it needs no renorm rescaling.
        new.rNrs = np.asarray(ip.rNr, dtype=np.float64)

    return new
