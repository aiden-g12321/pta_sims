"""
High-level entry point: a ``PTAModel`` plus hyper-parameters in, a statistically
consistent ``Data`` object out.

The model you hand in defines *everything* about the simulation except the
values of the hyper-parameters: which stochastic processes exist and how their
spectra depend on parameters, whether the background is Hellings-Downs or
uncorrelated, which deterministic signal is present and how its parameters map
to Fourier amplitudes.  Its ``Data`` supplies the array geometry -- TOA
sampling, timing model and white noise -- through the Gram matrices, and is
otherwise reused untouched.  That is what makes "NG15 pulsars, NG15 white
noise, my own GWB and my own CW" a one-liner: build the model on the real NG15
``Data`` and simulate.

The returned object carries the new ``Data``, a ``PTAModel`` rebound to it, the
injected truth (hyper-parameters, Fourier amplitudes, SNR) and a bundle of
numerical diagnostics.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import dm as _dm
from . import units
from .coefficients import (_as_rng, deterministic_coefficients,
                           draw_from_blocks, draw_from_phi_cube)
from .datasets import rebuild_data
from .inner_products import InnerProducts
from .linalg import DEFAULT_RTOL
from .simulate import (CompressedNoise, deterministic_metric,
                       matched_filter_snr, simulate_inner_products)


@dataclass
class SimulatedDataset:
    """
    Everything produced by :func:`simulate`.

    Attributes
    ----------
    data : prometheus.data.Data
        The simulated dataset, ready to hand to a ``PTAModel``.
    pta_model : prometheus.pta_model.PTAModel
        A copy of the input model rebound to ``data``.
    inner_products : InnerProducts
        The simulated inner products in natural units and float64, before the
        renormalization and float32 cast.  Keep these if you want to compare
        against what actually landed in ``data``.
    reference_inner_products : InnerProducts
        The inner products extracted from the input model's ``Data``.
    truth : dict
        ``params``   -- the hyper-parameters used, as passed in;
        ``a``        -- injected stochastic Fourier amplitudes (Np, Nc) in ns;
        ``a_det``    -- injected deterministic amplitudes (Np, Nd) in ns, or None;
        ``a_dm``     -- injected DM Gaussian process amplitudes (Np, Ndm) in ns
        (delays at the DM reference radio frequency), or None;
        ``psr_phases``, ``psr_dists`` -- per-pulsar deterministic parameters;
        ``phi_cube`` -- the prior covariance the amplitudes were drawn from;
        ``snr``      -- white-noise matched-filter SNR of the deterministic signal;
        ``snr_marginalized`` -- the same, marginalizing the stochastic processes.
    diagnostics : dict
        See :func:`simulate` and :func:`pta_sims.diagnostics.report`.
    """

    data: Any
    pta_model: Any
    inner_products: InnerProducts
    reference_inner_products: InnerProducts
    truth: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def summary(self):
        """Human-readable one-screen summary of the simulation."""
        from .diagnostics import format_report
        return format_report(self)


def simulate(pta_model, params=None, seed=None, *,
             include_white_noise=True,
             include_stochastic=True,
             include_deterministic=True,
             a=None, a_det=None, a_dm=None, phi_cube=None,
             psr_phases=None, psr_dists=None, noise=None,
             rtol=DEFAULT_RTOL, simulate_rNr=True, name=None):
    """
    Simulate a Prometheus dataset consistent with a model and hyper-parameters.

    Parameters
    ----------
    pta_model : prometheus.pta_model.PTAModel
        Defines the spectral and deterministic models, and supplies the array
        geometry through ``pta_model.data``.
    params : dict, optional
        Hyper-parameter values keyed by model name, matching what
        ``pta_model.get_param_names_and_shapes()`` reports:

        * standard mode -- ``{psr_model.name: (Np, nparams), gwb_model.name: (nparams,)}``
        * custom mode   -- ``{spectral_model.name: (nparams,)}``
        * deterministic -- ``{det_model.name: (nparams,)}``, plus optionally
          ``'psr_phases'`` (Np,) and ``'psr_dists'`` (Np, kpc) or
          ``'standard_psr_dists'`` (Np, in units of the distance uncertainty).

        Omit only if you are supplying ``a`` and ``a_det`` directly.
    seed : int, numpy Generator, or None
        Everything random in the simulation is drawn from this one stream:
        the Fourier amplitudes, the white noise projection, any pulsar phases
        left unspecified, and the unobserved part of ``rNr``.
    include_white_noise : bool
        When False the noise projection is replaced by its mean.  The result is
        a noiseless dataset whose posterior peaks near -- but, because the prior
        still shrinks the amplitudes, not exactly at -- the injected values.
    include_stochastic, include_deterministic : bool
        Switch off the intrinsic-noise-plus-GWB draw, or the deterministic
        injection, without changing the model.
    a : array_like, shape (Np, Nc), optional
        Inject these stochastic amplitudes (nanoseconds) instead of drawing
        them.  Bypasses ``phi_cube`` entirely.
    a_det : array_like, shape (Np, Nd), optional
        Inject these deterministic amplitudes (nanoseconds) instead of
        evaluating the deterministic model.
    a_dm : array_like, shape (Np, Ndm), optional
        Inject these DM Gaussian process amplitudes (nanoseconds, delays at the
        DM reference radio frequency) instead of drawing them from the model's
        ``dm_model``.  Only for models with a DM Gaussian process.
    phi_cube : array_like, shape (Nc, Np, Np), optional
        Use this prior covariance (ns^2) instead of evaluating the spectral
        models.  Useful to supply a float64 covariance where the default
        float32 evaluation is not good enough.
    psr_phases : array_like, shape (Np,), optional
        CW phase at each pulsar.  Defaults to ``params['psr_phases']`` if
        present, otherwise drawn uniformly on [0, 2 pi), which is the model's
        own prior.
    psr_dists : array_like, shape (Np,), optional
        Pulsar distances in kpc.  Defaults to ``params['psr_dists']``, or to
        ``params['standard_psr_dists']`` mapped through the measured distance
        and uncertainty, or to the measured distances themselves.
    noise : CompressedNoise, optional
        Pre-built factorization of the joint Gram matrix, from
        ``CompressedNoise.from_inner_products(InnerProducts.from_data(data))``.
        The factorization dominates the cost of a simulation, so building it
        once and passing it in turns an ensemble of realizations from minutes
        into seconds.  It is also returned on the result as
        ``sim.diagnostics['noise']``.  For a model with a DM Gaussian process
        build it from ``pta_sims.dm.augmented_inner_products(data)`` instead.
    rtol : float
        Eigenvalue truncation used for every PSD factorization.  Ignored when
        ``noise`` is supplied.
    simulate_rNr : bool
        Also simulate ``r^T Ntilde^-1 r``.
    name : str, optional
        Name for the simulated dataset.

    Returns
    -------
    SimulatedDataset
    """
    rng = _as_rng(seed)
    params = dict(params or {})
    data = pta_model.data

    ref_ip = InnerProducts.from_data(data)
    dm_model = getattr(pta_model, 'dm_model', None)

    # ------------------------------------------------------------------
    # stochastic Fourier amplitudes
    # ------------------------------------------------------------------
    phi_info = {}
    supplied_phi = (None if phi_cube is None
                    else np.asarray(phi_cube, dtype=np.float64))
    if a is not None:
        # amplitudes given outright; a phi_cube passed alongside is still worth
        # keeping, since the marginalized SNR is computed from it
        a_ns = np.asarray(a, dtype=np.float64)
        phi_cube_used = supplied_phi
    elif not include_stochastic:
        a_ns = np.zeros((ref_ip.npsrs, ref_ip.ncomponents))
        phi_cube_used = supplied_phi
    elif supplied_phi is not None:
        phi_cube_used = supplied_phi
        a_ns, phi_info = draw_from_phi_cube(phi_cube_used, rng=rng, rtol=rtol)
    elif getattr(pta_model, 'freq_corrs', False) or dm_model is not None:
        # Inter-frequency correlations or a second basis put Prometheus in its
        # marginalized sampling model, whose per-pulsar prior is dense and no
        # longer factorizes over frequency component; mirror that.  Where the
        # prior does factorize the two routes draw from the same distribution,
        # since F a_psr + F a_gwb = F (a_psr + a_gwb), so the branch below is
        # the cheaper way to say the same thing.
        a_ns, phi_info = _draw_with_freq_corrs(pta_model, params, rng, rtol)
        phi_cube_used = None
    else:
        phi_cube_used = build_phi_cube(pta_model, params)
        a_ns, phi_info = draw_from_phi_cube(phi_cube_used, rng=rng, rtol=rtol)

    # ------------------------------------------------------------------
    # deterministic Fourier amplitudes
    # ------------------------------------------------------------------
    a_det_ns = None
    phases_used = dists_used = None
    if ref_ip.has_deterministic and include_deterministic:
        if a_det is not None:
            a_det_ns = np.asarray(a_det, dtype=np.float64)
        elif pta_model.det_model is not None:
            det_params = _require(params, pta_model.det_model.name,
                                  'deterministic model parameters')
            phases_used, dists_used = _resolve_psr_params(
                pta_model, params, psr_phases, psr_dists, rng)
            a_det_ns = deterministic_coefficients(
                pta_model.det_model, det_params, phases_used, dists_used)
    if ref_ip.has_deterministic and a_det_ns is None:
        a_det_ns = np.zeros((ref_ip.npsrs, ref_ip.ndet))

    if a_det_ns is not None and not np.all(np.isfinite(a_det_ns)):
        raise ValueError(
            'The deterministic model returned non-finite Fourier amplitudes. '
            'For the evolving CW models this usually means the binary merges '
            'within the observation, which the waveform flags with NaN.')

    # ------------------------------------------------------------------
    # DM Gaussian process amplitudes: the DM basis joins the stochastic one
    # ------------------------------------------------------------------
    a_dm_ns = None
    if dm_model is not None:
        if a_dm is not None:
            a_dm_ns = np.asarray(a_dm, dtype=np.float64)
        elif not include_stochastic:
            a_dm_ns = np.zeros((ref_ip.npsrs, data.ncomponents_dm))
        else:
            a_dm_ns, phi_info_dm = _dm.draw_dm_amplitudes(pta_model, params,
                                                          rng=rng, rtol=rtol)
            phi_info = dict(phi_info, dm=phi_info_dm)
        sim_ip = _dm.augment(ref_ip, _dm.DMInnerProducts.from_data(data))
        a_sim = np.concatenate([a_ns, a_dm_ns], axis=-1)
    elif a_dm is not None:
        raise ValueError('a_dm was supplied but the model has no dm_model.')
    else:
        sim_ip = ref_ip
        a_sim = a_ns

    # ------------------------------------------------------------------
    # the simulation proper
    # ------------------------------------------------------------------
    if noise is None:
        noise = CompressedNoise.from_inner_products(sim_ip, rtol=rtol)

    sim = simulate_inner_products(
        sim_ip, a=a_sim, a_det=a_det_ns, rng=rng,
        include_white_noise=include_white_noise,
        simulate_rNr=simulate_rNr, noise=noise)

    if dm_model is not None:
        sim_base_ip, sim_dm_ip = _dm.split(sim.inner_products, ref_ip.ncomponents)
        new_data = _dm.install(rebuild_data(data, sim_base_ip, name=name), sim_dm_ip)
    else:
        sim_base_ip = sim.inner_products
        new_data = rebuild_data(data, sim_base_ip, name=name)
    new_model = rebind_model(pta_model, new_data)

    # ------------------------------------------------------------------
    # diagnostics and truth
    # ------------------------------------------------------------------
    diagnostics = dict(sim.diagnostics)
    diagnostics['phi_factor'] = phi_info
    diagnostics['noise'] = noise
    diagnostics['downcast'] = _downcast_report(sim_base_ip, new_data)

    snr = snr_marg = 0.0
    if ref_ip.has_deterministic and a_det_ns is not None and np.any(a_det_ns):
        snr = matched_filter_snr(ref_ip, a_det_ns)
        if phi_cube_used is not None:
            phi_diag = np.einsum('cpp->pc', phi_cube_used)
            snr_marg = matched_filter_snr(ref_ip, a_det_ns, phi=phi_diag)

    truth = {
        'params': params,
        'a': a_ns,
        'a_det': a_det_ns,
        'a_dm': a_dm_ns,
        'psr_phases': phases_used,
        'psr_dists': dists_used,
        'phi_cube': phi_cube_used,
        'snr': snr,
        'snr_marginalized': snr_marg,
        'seed': seed,
    }

    return SimulatedDataset(
        data=new_data,
        pta_model=new_model,
        inner_products=sim_base_ip,
        reference_inner_products=ref_ip,
        truth=truth,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# prior covariance
# ---------------------------------------------------------------------------

def build_phi_cube(pta_model, params):
    """
    Evaluate the total prior covariance of the stochastic Fourier amplitudes.

    Sums the contribution of every spectral model in the ``PTAModel``, using
    the models' own ``get_phi_cube``, so the covariance the amplitudes are
    drawn from is exactly the one the sampler will evaluate.

    Prometheus evaluates its posterior in float32 and rounds every spectral
    model's output to it, so this evaluation carries a relative error of order
    1e-7 however the ambient JAX precision is set.  That is far below the
    sampling scatter of a single Gaussian process realization and is
    irrelevant to the simulation; the result is upcast to float64 before it is
    factored.  Pass an explicit ``phi_cube`` to :func:`simulate` if you want it
    computed otherwise.

    Parameters
    ----------
    pta_model : prometheus.pta_model.PTAModel
    params : dict

    Returns
    -------
    ndarray, shape (Nc, Np, Np), float64, ns^2
    """
    if pta_model.mode != 'standard':
        spectral_model = pta_model.spectral_model
        spectral_params = _require(params, spectral_model.name,
                                   'spectral parameters')
        return np.asarray(
            spectral_model.get_phi_cube(spectral_params, spectral_model.freqs),
            dtype=np.float64)

    # every process contributing to the shared basis: the per-pulsar ones
    # (Prometheus allows several) and, when one is modelled, the common one
    models = list(pta_model.psr_models)
    if pta_model.gwb_model is not None:
        models.append(pta_model.gwb_model)

    off_basis = {m.name: m.basis for m in models if m.basis != 'red'}
    if off_basis:
        raise ValueError(
            f'build_phi_cube describes a single shared Fourier basis, but '
            f'{off_basis} live on another one. Processes spanning several '
            'bases -- a DM Gaussian process, say -- are simulated through '
            'pta_sims.dm.augmented_inner_products instead; pta_sims.simulate '
            'takes that route by itself.')

    phi = 0.0
    for model in models:
        values = _require(params, model.name, f"parameters of '{model.name}'")
        # each model's own ``freqs``, not ``data.freqs``: they are the
        # harmonics of that model's basis, and they are what the sampler
        # evaluates the spectrum on
        phi = phi + np.asarray(model.get_phi_cube(values, model.freqs),
                               dtype=np.float64)
    return phi


def _draw_with_freq_corrs(pta_model, params, rng, rtol):
    """
    Amplitude draw for a model with inter-frequency correlations.

    Mirrors ``PTAModel.sampling_model_marginalized`` on the shared ``'red'``
    basis: the per-pulsar processes contribute a dense ``(Nc, Nc)`` block per
    pulsar -- several of them adding their covariances -- and the common
    process contributes ``kron(correlation_matrix, phi_spec)`` over the
    components it covers.  Processes on other bases are drawn separately and
    stacked by the caller (see :mod:`pta_sims.dm`), which is what the
    marginalized model's block-diagonal ``phi_B`` does.
    """
    from jax import vmap
    import jax.numpy as jnp

    red_models = [m for m in pta_model.psr_models if m.basis == 'red']
    if not red_models:
        raise ValueError(
            "No per-pulsar process lives on the 'red' basis, so there are no "
            'stochastic amplitudes to draw on it.')

    per_pulsar_phi = 0.0
    for model in red_models:
        values = _require(params, model.name, f"parameters of '{model.name}'")
        per_pulsar_phi = per_pulsar_phi + np.asarray(
            vmap(model.get_phi_func, in_axes=(0, None))(
                jnp.asarray(values), model.freqs), dtype=np.float64)

    nc = per_pulsar_phi.shape[-1]
    gwb_model = pta_model.gwb_model
    if gwb_model is None or gwb_model.basis != 'red':
        # no common process on this basis: the pulsars are independent
        corr = np.eye(per_pulsar_phi.shape[0], dtype=np.float64)
        padded = np.zeros((nc, nc), dtype=np.float64)
    else:
        gwb_params = _require(params, gwb_model.name,
                              'common-process spectral parameters')
        gwb_na = 2 * gwb_model.nfreqs
        common = np.asarray(
            gwb_model.get_phi_func(jnp.asarray(gwb_params), gwb_model.freqs),
            dtype=np.float64)[:gwb_na, :gwb_na]
        padded = np.zeros((nc, nc), dtype=np.float64)
        padded[:gwb_na, :gwb_na] = common
        corr = np.asarray(gwb_model.correlation_matrix, dtype=np.float64)

    return draw_from_blocks(per_pulsar_phi, corr, padded, rng=rng, rtol=rtol)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require(params, key, what):
    if key not in params:
        raise KeyError(
            f"Missing {what} under key '{key}'. Supplied keys: "
            f'{sorted(params)}. Model parameter names and shapes are reported '
            f'by PTAModel.get_param_names_and_shapes().')
    return np.asarray(params[key], dtype=np.float64)


def _resolve_psr_params(pta_model, params, psr_phases, psr_dists, rng):
    """
    Settle the per-pulsar deterministic parameters (CW phase and distance).

    Returns ``(None, None)`` for a deterministic model that does not use them.
    """
    det_model = pta_model.det_model
    if det_model is None or not det_model.with_psr_params:
        return None, None

    data = pta_model.data
    npsrs = data.npsrs

    if psr_phases is None:
        psr_phases = params.get('psr_phases')
    if psr_phases is None:
        # the model's own prior
        psr_phases = rng.uniform(0.0, 2.0 * np.pi, size=npsrs)
    psr_phases = np.asarray(psr_phases, dtype=np.float64)

    measured = np.asarray(data.psr_dists_measured, dtype=np.float64)
    if psr_dists is None:
        psr_dists = params.get('psr_dists')
    if psr_dists is None and 'standard_psr_dists' in params:
        std = np.asarray(data.psr_dists_std, dtype=np.float64)
        psr_dists = measured + np.asarray(params['standard_psr_dists'],
                                          dtype=np.float64) * std
    if psr_dists is None:
        psr_dists = measured
    psr_dists = np.asarray(psr_dists, dtype=np.float64)

    if psr_phases.shape != (npsrs,) or psr_dists.shape != (npsrs,):
        raise ValueError(f'psr_phases and psr_dists must have shape ({npsrs},).')
    return psr_phases, psr_dists


def rebind_model(pta_model, new_data):
    """
    Return a copy of ``pta_model`` whose sub-models point at ``new_data``.

    The Gram matrices, frequency grids and sparse deterministic grid are shared
    by reference between the old and new ``Data``, so nothing the sub-models
    precomputed at construction (the Hellings-Downs matrix, the Tukey window,
    the jitted coefficient function) becomes stale.

    Prometheus takes any number of per-pulsar processes, each on its own
    Fourier basis, so the whole of ``pta_model.psr_models`` is rebound rather
    than ``psr_model`` alone -- passing only the first silently drops the rest.
    A DM process is already one of them and is rediscovered from its basis, so
    it must not also be passed as ``dm_model``, which would append it twice.
    The choice of sampling model is carried over explicitly instead of being
    re-derived, so a model built with ``sampling='marginalized'`` stays
    marginalized.
    """
    from prometheus.pta_model import PTAModel

    def rebound(model):
        if model is None:
            return None
        clone = copy.copy(model)
        clone.data = new_data
        return clone

    psr_models = [rebound(m) for m in pta_model.psr_models]

    return PTAModel(psr_model=psr_models or None,
                    gwb_model=rebound(pta_model.gwb_model),
                    spectral_model=rebound(pta_model.spectral_model),
                    det_model=rebound(pta_model.det_model),
                    sampling=pta_model.sampling)


def _downcast_report(ip, data):
    """
    How much the float32 cast into the ``Data`` object actually cost.

    Compares each stored array against the float64 original.  Relative errors
    of order 1e-7 are the float32 epsilon and are expected; anything larger,
    or any non-finite entry, means the dynamic range of the simulated data has
    outrun single precision.
    """
    out = {}
    pairs = [('FNrs', units.to_renorm(ip.FNr, 1)),
             ('FNFs', units.to_renorm(ip.FNF, 2))]
    if ip.has_deterministic and ip.FDNr is not None:
        pairs += [('TDNrs', units.to_renorm(ip.FDNr, 1)),
                  ('TNTDs', units.to_renorm(ip.FNFD, 2)),
                  ('TDNTDs', units.to_renorm(ip.FDNFD, 2))]

    for attr, exact in pairs:
        stored = np.asarray(getattr(data, attr), dtype=np.float64)
        scale = np.max(np.abs(exact))
        denom = np.where(np.abs(exact) > 0, np.abs(exact), np.inf)
        out[attr] = {
            'max_rel_err': float(np.max(np.abs(stored - exact) / denom)),
            'max_rel_err_vs_scale': float(np.max(np.abs(stored - exact)) / scale)
            if scale > 0 else 0.0,
            'n_nonfinite': int(np.sum(~np.isfinite(stored))),
            'n_underflow': int(np.sum((stored == 0) & (exact != 0))),
            'abs_max': float(scale),
            'abs_min_nonzero': float(np.min(np.abs(exact[exact != 0])))
            if np.any(exact != 0) else 0.0,
        }
    return out
