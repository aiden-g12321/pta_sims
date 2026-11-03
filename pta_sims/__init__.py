"""
pta_sims -- simulate compressed PTA datasets for Prometheus.

Rather than simulating TOAs, refitting a timing model and compressing the
result, this package simulates the compressed statistics directly:

    FNr  = F^T   Ntilde^-1 r
    FDNr = F_D^T Ntilde^-1 r
    rNr  = r^T   Ntilde^-1 r

conditioned on injected intrinsic pulsar noise, a gravitational wave
background, and a deterministic signal, with the white noise entering exactly
through the Gram matrices ``FNF``, ``FNFD`` and ``FDNFD`` that the real dataset
already carries.  See :mod:`pta_sims.simulate` for the derivation and
:mod:`pta_sims.linalg` for why the rank deficiency of those Gram matrices
dictates how the noise is drawn.

Quick start
-----------
::

    import pta_sims

    data = pta_sims.load_data('data/NG15/data.pkl')
    model = pta_sims.build_standard_model(
        data, correlations='HD',
        det_model=pta_sims.build_cw_model(data))

    sim = pta_sims.simulate(
        model,
        params={
            'red_noise': pta_sims.power_law_params(-14.3, 3.2, npsrs=data.npsrs),
            'gwb':       pta_sims.power_law_params(-14.6, 4.33),
            'cw':        pta_sims.cw_params(9.0, -8.0, 0.5, 1.0,
                                            -14.0, 0.3, 2.0, 1.5),
        },
        seed=42)

    print(sim.summary())
    sim.data          # a statistically consistent prometheus.data.Data
    sim.pta_model     # the same model, rebound to it
"""

from . import dm
from .api import SimulatedDataset, build_phi_cube, rebind_model, simulate
from .coefficients import (deterministic_coefficients, draw_from_blocks,
                           draw_from_phi_cube)
from .diagnostics import posterior_summary, sky_summary
from .datasets import load_data, rebuild_data, save_data, strip_pulsars
from .inner_products import InnerProducts
from .sampling import run_nuts_monitored
from .models import (build_cw_model, build_standard_model, cw_params,
                     cw_psr_phases, power_law_params)
from .simulate import (CompressedNoise, InnerProductSimulation,
                       deterministic_metric, deterministic_statistic,
                       marginal_delta_ln_likelihood, matched_filter_snr,
                       simulate_inner_products, whiten)
from . import detectability, diagnostics, linalg, sampling, units, whitenoise

__version__ = '0.1.0'

__all__ = [
    'CompressedNoise',
    'InnerProducts',
    'InnerProductSimulation',
    'SimulatedDataset',
    'build_cw_model',
    'build_phi_cube',
    'build_standard_model',
    'cw_params',
    'cw_psr_phases',
    'deterministic_coefficients',
    'deterministic_metric',
    'deterministic_statistic',
    'diagnostics',
    'draw_from_blocks',
    'draw_from_phi_cube',
    'linalg',
    'run_nuts_monitored',
    'load_data',
    'posterior_summary',
    'marginal_delta_ln_likelihood',
    'matched_filter_snr',
    'power_law_params',
    'rebind_model',
    'rebuild_data',
    'save_data',
    'sky_summary',
    'simulate',
    'simulate_inner_products',
    'strip_pulsars',
    'sampling',
    'units',
    'whiten',
    'whitenoise',
]
