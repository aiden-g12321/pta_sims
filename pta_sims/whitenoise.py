"""
Recomputing the inner products for a different white noise model.

The Gram matrices ``FNF``, ``FNFD`` and ``FDNFD`` are the only place the white
noise enters the compressed data, so changing EFAC / EQUAD / ECORR means
redoing the enterprise noise solves.  That is expensive -- minutes for a
67-pulsar array -- and needs ``data.psrs`` still attached, which is why
:func:`pta_sims.datasets.save_data` warns before stripping them.

The usual workflow does *not* need this: reusing NG15's measured white noise
and injecting a custom GWB or continuous wave only touches the residual side,
which :func:`pta_sims.simulate` handles without any enterprise call.  Reach for
this module when the white noise itself is the thing you want to vary.
"""

import numpy as np

from .inner_products import InnerProducts


def build_det_basis(psr, freqs_forFFT, ndet):
    """
    Rebuild the deterministic Fourier design matrix ``F_D`` for one pulsar.

    Mirrors Prometheus' construction: sine/cosine interleaved at the first
    ``ndet // 2`` non-zero FFT frequencies of the *extended* window, evaluated
    at the pulsar's own TOAs.

    Parameters
    ----------
    psr : enterprise pulsar object
    freqs_forFFT : array_like
        The extended-window FFT frequency grid, as stored on the ``Data``
        object.  Index 0 is the DC bin and is skipped.
    ndet : int
        Number of deterministic basis columns (``2 * nfreqs_det``).

    Returns
    -------
    ndarray, shape (ntoas, ndet)
    """
    freqs_forFFT = np.asarray(freqs_forFFT, dtype=np.float64)
    toas = np.asarray(psr.toas, dtype=np.float64)
    F_D = np.zeros((toas.shape[0], ndet), dtype=np.float64)
    for j in range(ndet // 2):
        phase = 2.0 * np.pi * freqs_forFFT[j + 1] * toas
        F_D[:, 2 * j] = np.sin(phase)
        F_D[:, 2 * j + 1] = np.cos(phase)
    return F_D


def recompute_inner_products(data, wn_dict=None, with_deterministic=True,
                             verbose=True):
    """
    Redo the enterprise noise solves, optionally under a new white noise model.

    All six inner products come out of a single ``ndiag.solve`` per pulsar: the
    bases and the residuals are stacked into one matrix ``B = [F | F_D | r]``
    and the blocks of ``B^T Ntilde^-1 B`` are sliced apart afterwards.  The
    solve dominates the cost, so doing it once rather than six times matters.

    The residual-side entries (``FNr``, ``FDNr``, ``rNr``) returned here are
    those of the *real* residuals under the new white noise.  Feed the result to
    :func:`pta_sims.simulate.simulate_inner_products` to replace them with a
    simulated realization.

    Parameters
    ----------
    data : prometheus.data.Data
        Must still carry ``data.psrs``.  Its ``Tspan`` defines the Fourier
        basis, so the recomputed ``F`` matches the one the object was built
        with even if the pulsar list has been subsetted.
    wn_dict : dict, optional
        White noise parameters.  Defaults to the dictionary already on
        ``data``.
    with_deterministic : bool
        Also compute the deterministic-basis blocks.
    verbose : bool
        Print per-pulsar progress.

    Returns
    -------
    InnerProducts
    """
    if not getattr(data, 'psrs', None):
        raise ValueError(
            'This Data object has no pulsars attached, so its inner products '
            'cannot be recomputed. Load the unstripped object.')

    # Only the 'red' and deterministic bases are stacked into B below, so a
    # Data object carrying further bases would come back with their Gram
    # blocks still describing the *old* white noise while FNF and FNr describe
    # the new one -- an inconsistency that nothing downstream would catch.
    from .datasets import _basis_names
    extra = [b for b in _basis_names(data) if b != 'red']
    if extra:
        raise NotImplementedError(
            f'This Data object carries the extra Fourier bases {extra}, whose '
            'noise-weighted inner products this function does not recompute. '
            'Changing the white noise would leave them inconsistent with the '
            "'red' and deterministic blocks, so rebuild the Data object from "
            'its pulsars under the new white noise instead.')

    wn_dict = wn_dict if wn_dict is not None else data.wn_dict
    names = list(data.psr_names)
    by_name = {p.name: p for p in data.psrs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f'Pulsars missing from data.psrs: {missing}')

    # The Fourier basis is defined on the *PTA* baseline, which the Data object
    # already fixed when it was built. Re-deriving it from data.psrs would give
    # a different grid for any object whose pulsar list has since been
    # subsetted, and the resulting F would silently not match the stored one.
    Tspan = float(data.Tspan)

    nc = 2 * data.nfreqs
    nd = data.num_coeff_det if with_deterministic else 0

    FNF, FNr, FNFD, FDNFD, FDNr, rNr = [], [], [], [], [], []
    ntoas, ntmpars = [], []

    for i, name in enumerate(names):
        psr = by_name[name]
        pta = data.build_single_psr_enterprise_model(psr, Tspan)
        pta.set_default_params(wn_dict)
        params = {}

        F = np.asarray(pta.get_basis(params)[0], dtype=np.float64)
        if F.shape[1] != nc:
            raise ValueError(
                f'{name}: basis has {F.shape[1]} columns but the Data object '
                f'expects {nc}. The Fourier basis of the rebuilt enterprise '
                f'model does not match the one the Data was built with.')

        collection = pta._signalcollections[0]
        ndiag = collection.get_ndiag(params)
        res = np.asarray(collection.get_detres(params), dtype=np.float64)

        if with_deterministic:
            F_D = build_det_basis(
                psr, data.per_psr_data_dict[name]['freqs_forFFT'], nd)
            B = np.concatenate([F, F_D, res[:, None]], axis=1)
        else:
            B = np.concatenate([F, res[:, None]], axis=1)

        BNB = np.asarray(ndiag.solve(B, left_array=B), dtype=np.float64)

        s_f = slice(0, nc)
        s_d = slice(nc, nc + nd)
        i_r = nc + nd

        FNF.append(BNB[s_f, s_f])
        FNr.append(BNB[s_f, i_r])
        rNr.append(BNB[i_r, i_r])
        if with_deterministic:
            FNFD.append(BNB[s_f, s_d])
            FDNFD.append(BNB[s_d, s_d])
            FDNr.append(BNB[s_d, i_r])

        ntoas.append(len(psr.toas))
        ntmpars.append(psr.Mmat.shape[1])

        if verbose:
            print(f'[{i + 1}/{len(names)}] {name}: {len(psr.toas)} TOAs',
                  flush=True)

    return InnerProducts(
        psr_names=names,
        FNF=np.stack(FNF),
        FNr=np.stack(FNr),
        FNFD=np.stack(FNFD) if with_deterministic else None,
        FDNFD=np.stack(FDNFD) if with_deterministic else None,
        FDNr=np.stack(FDNr) if with_deterministic else None,
        rNr=np.array(rNr),
        ntoas=np.array(ntoas),
        ntmpars=np.array(ntmpars),
    )


def scale_white_noise(wn_dict, efac_scale=1.0, log10_equad_shift=0.0,
                      log10_ecorr_shift=0.0, psr_names=None):
    """
    Perturb a white noise dictionary.

    Parameters
    ----------
    wn_dict : dict
        Enterprise-style ``{psrname}_{backend}_{param}`` dictionary.
    efac_scale : float
        Multiplies every EFAC.
    log10_equad_shift, log10_ecorr_shift : float
        Added to every ``log10_t2equad`` / ``log10_ecorr``.
    psr_names : sequence of str, optional
        Restrict the change to these pulsars.

    Returns
    -------
    dict
        A new dictionary; the input is not modified.
    """
    out = dict(wn_dict)
    for key, value in wn_dict.items():
        if psr_names is not None and key.split('_')[0] not in set(psr_names):
            continue
        if key.endswith('_efac'):
            out[key] = value * efac_scale
        elif key.endswith('_log10_t2equad'):
            out[key] = value + log10_equad_shift
        elif key.endswith('_log10_ecorr'):
            out[key] = value + log10_ecorr_shift
    return out
