"""
Convenience constructors for the Prometheus models a simulation needs.

Nothing here is required -- :func:`pta_sims.simulate` accepts any ``PTAModel``
you build yourself.  These are shortcuts for the two configurations that come
up constantly: a power-law intrinsic red noise plus a correlated power-law
background, and an evolving continuous wave from a single binary.
"""

import numpy as np


DEFAULT_RN_BOUNDS = [[-20.0, -11.0], [0.0, 7.0]]
DEFAULT_GWB_BOUNDS = [[-18.0, -11.0], [0.0, 7.0]]
DEFAULT_DM_BOUNDS = [[-20.0, -11.0], [0.0, 7.0]]

#: Parameter order of the Prometheus CW waveforms, with the bounds the
#: ``cw_delay_full_prior_float32`` implementation is validated over.
CW_PARAM_NAMES = ('log10_mc', 'log10_fgw', 'cos_inc', 'psi',
                  'log10_h', 'cos_gwtheta', 'gwphi', 'phase0')
DEFAULT_CW_BOUNDS = [
    [7.0, 10.0],            # log10(Mc / Msun)
    [-9.0, -7.0],           # log10(f_gw / Hz)
    [-1.0, 1.0],            # cos(inclination)
    [0.0, np.pi],           # polarization
    [-18.0, -11.0],         # log10(strain)
    [-1.0, 1.0],            # cos(theta_gw)
    [0.0, 2.0 * np.pi],     # phi_gw
    [0.0, 2.0 * np.pi],     # initial phase
]


def build_standard_model(data, correlations='HD', gwb_nfreqs=None,
                         rn_bounds=None, gwb_bounds=None,
                         rn_spectrum=None, gwb_spectrum=None,
                         rn_name='red_noise', gwb_name='gwb',
                         det_model=None, freq_corrs=False,
                         dm=False, dm_bounds=None, dm_spectrum=None,
                         dm_name='dm_gp', dm_freq_corrs=None):
    """
    Build a ``PTAModel`` with per-pulsar red noise and a common background.

    Parameters
    ----------
    data : prometheus.data.Data
    correlations : {'HD', 'CURN'}, ndarray, or None
        Inter-pulsar correlation pattern of the common process.  ``None``
        leaves the common process out altogether, giving a pulsar-noise-only
        model.
    gwb_nfreqs : int, optional
        Number of frequencies the background occupies.  Defaults to all of
        ``data.nfreqs``.
    rn_bounds, gwb_bounds : array_like, shape (nparams, 2), optional
        Parameter bounds.  Defaults are a log-uniform amplitude and a uniform
        spectral index over the usual search ranges.
    rn_spectrum, gwb_spectrum : callable, optional
        ``get_phi_diag_func`` for each process.  Default
        ``prometheus.spectra.power_law``.
    rn_name, gwb_name : str
        Names the hyper-parameters are keyed by.
    det_model : prometheus.deterministic_models.DeterministicModel, optional
    freq_corrs : bool
        Model inter-frequency correlations.  The prior then no longer
        factorizes over frequency components, and :func:`pta_sims.simulate`
        switches to assembling and factoring the full ``(Np * Nc)`` covariance
        rather than one ``(Np, Np)`` block per component.
    dm : bool
        Add a per-pulsar DM Gaussian process (``prometheus.spectral_models.
        DMSpectralModel``).  Needs a ``Data`` built with ``nfreqs_dm > 0``.
        Prometheus then always uses its marginalized sampling model.
    dm_bounds, dm_spectrum, dm_name
        As for the red noise; defaults are :data:`DEFAULT_DM_BOUNDS`,
        ``prometheus.spectra.power_law`` and ``'dm_gp'``.
    dm_freq_corrs : bool, optional
        Inter-frequency correlations for the DM process.  Defaults to
        ``freq_corrs``.

    Returns
    -------
    prometheus.pta_model.PTAModel
    """
    from prometheus import spectra
    from prometheus.pta_model import PTAModel
    from prometheus.spectral_models import (CommonSpectralModel,
                                            IndependentSpectralModel)

    psr_model = IndependentSpectralModel(
        name=rn_name,
        parameter_bounds=np.asarray(rn_bounds if rn_bounds is not None
                                    else DEFAULT_RN_BOUNDS),
        data=data,
        get_phi_diag_func=rn_spectrum or spectra.power_law,
        freq_corrs=freq_corrs)

    gwb_model = None
    if correlations is not None:
        gwb_model = CommonSpectralModel(
            name=gwb_name,
            parameter_bounds=np.asarray(gwb_bounds if gwb_bounds is not None
                                        else DEFAULT_GWB_BOUNDS),
            data=data,
            correlation_matrix=correlations,
            get_phi_diag_func=gwb_spectrum or spectra.power_law,
            nfreqs=gwb_nfreqs,
            freq_corrs=freq_corrs)

    kwargs = {}
    if dm:
        from prometheus.spectral_models import DMSpectralModel
        kwargs['dm_model'] = DMSpectralModel(
            name=dm_name,
            parameter_bounds=np.asarray(dm_bounds if dm_bounds is not None
                                        else DEFAULT_DM_BOUNDS),
            data=data,
            get_phi_diag_func=dm_spectrum or spectra.power_law,
            freq_corrs=freq_corrs if dm_freq_corrs is None else dm_freq_corrs)

    return PTAModel(psr_model=psr_model, gwb_model=gwb_model,
                    det_model=det_model, **kwargs)


def build_cw_model(data, name='cw', bounds=None, waveform=None):
    """
    Build a continuous-wave ``DeterministicModel``.

    Parameters
    ----------
    data : prometheus.data.Data
    name : str
        Key the CW parameters are supplied under.
    bounds : array_like, shape (8, 2), optional
        Defaults to :data:`DEFAULT_CW_BOUNDS`; the parameter order is
        :data:`CW_PARAM_NAMES`.
    waveform : callable, optional
        Delay function.  Defaults to
        ``prometheus.deterministic.cw_delay_full_prior_float32``, which is the
        one written to stay valid across the whole default prior.

    Returns
    -------
    prometheus.deterministic_models.DeterministicModel
    """
    from prometheus import deterministic
    from prometheus.deterministic_models import DeterministicModel

    return DeterministicModel(
        name=name,
        data=data,
        get_delays_func=waveform or deterministic.cw_delay_full_prior_float32,
        parameter_bounds=np.asarray(bounds if bounds is not None
                                    else DEFAULT_CW_BOUNDS),
        with_psr_params=True)


def cw_psr_phases(data, params, psr_dists=None):
    """
    Pulsar-term phases implied by a CW source, for a physically consistent
    injection.

    Prometheus samples the per-pulsar CW phase as a free parameter, because the
    pulsar distances are too poorly known to predict it.  An *injection*,
    though, knows the true distances, so the phases are not free: they follow
    from the source parameters and the light-travel delay to each pulsar.
    Drawing them uniformly instead (which is what :func:`pta_sims.simulate`
    does when none are supplied) injects a signal that is still inside the
    model, but is not the signal a real binary would produce.

    Parameters
    ----------
    data : prometheus.data.Data
    params : array_like, shape (8,)
        CW parameters in the order of :data:`CW_PARAM_NAMES`.
    psr_dists : array_like, shape (Np,), optional
        True pulsar distances in kpc.  Defaults to the measured distances on
        the ``Data`` object.

    Returns
    -------
    ndarray, shape (Np,)
        Phases on [0, 2 pi).
    """
    import jax.numpy as jnp
    from jax import vmap
    from prometheus.deterministic import get_psr_phase

    if psr_dists is None:
        psr_dists = data.psr_dists_measured
    phases = vmap(get_psr_phase, in_axes=(None, 0, 0))(
        jnp.asarray(params), jnp.asarray(data.psrpos), jnp.asarray(psr_dists))
    return np.asarray(phases, dtype=np.float64)


def cw_params(log10_mc, log10_fgw, cos_inc, psi, log10_h, cos_gwtheta,
              gwphi, phase0):
    """Assemble a CW parameter vector in the order the waveform expects."""
    return np.array([log10_mc, log10_fgw, cos_inc, psi, log10_h,
                     cos_gwtheta, gwphi, phase0], dtype=np.float64)


def power_law_params(log10_A, gamma, npsrs=None):
    """
    Assemble power-law hyper-parameters.

    Parameters
    ----------
    log10_A, gamma : float or array_like
        Scalars broadcast across the array, or per-pulsar arrays of length
        ``npsrs``.
    npsrs : int, optional
        When given, returns a ``(npsrs, 2)`` array suitable for an
        ``IndependentSpectralModel``; otherwise a ``(2,)`` array for a
        ``CommonSpectralModel``.

    Returns
    -------
    ndarray
    """
    if npsrs is None:
        return np.array([log10_A, gamma], dtype=np.float64)
    log10_A = np.broadcast_to(np.asarray(log10_A, dtype=np.float64), (npsrs,))
    gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (npsrs,))
    return np.stack([log10_A, gamma], axis=-1)
