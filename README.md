# pta_sims

Simulate pulsar timing array datasets for
[Prometheus](https://github.com/XGI-MSU/prometheus) at the level the sampler
actually consumes — the noise-weighted inner products — rather than by
generating TOAs and refitting a timing model.

Hand it a `prometheus.pta_model.PTAModel` and a set of hyper-parameters, and get
back a statistically consistent `prometheus.data.Data`, ready to sample. The
array geometry — TOA sampling, timing model, measured white noise — is reused
untouched, so "NG15 pulsars, NG15 white noise, my own background and my own
continuous wave" is a handful of lines.


## Install

```bash
pip install -e .
```

Prometheus comes with it, from its repository. Add `'.[examples]'` for what the
notebooks import, or `'.[test]'` for pytest.


## Usage

```python
import pta_sims

data  = pta_sims.load_data('data/NG15/data.pkl')
model = pta_sims.build_standard_model(
    data, correlations='HD', det_model=pta_sims.build_cw_model(data))

sim = pta_sims.simulate(model, seed=42, params={
    'red_noise': pta_sims.power_law_params(-15.5, 2.5, npsrs=data.npsrs),
    'gwb':       pta_sims.power_law_params(-14.6, 13 / 3),
    'cw':        pta_sims.cw_params(log10_mc=9.5, log10_fgw=-8.4, cos_inc=0.6,
                                    psi=0.9, log10_h=-14.3, cos_gwtheta=0.04,
                                    gwphi=4.06, phase0=1.2),
})

print(sim.summary())

sim.data       # prometheus.data.Data, ready to sample
sim.pta_model  # the same model, rebound to it
sim.truth      # injected amplitudes, SNR, hyper-parameters
```

The model defines everything about the simulation except the hyper-parameter
values, and it is also the model you then sample, so the injection and the
recovery cannot drift apart. `load_data` takes any Prometheus `Data`; the NG15
pickle above is ~2 GB and is not distributed here.

Worked examples, injection through recovery:
[`ng15_stochastic_pe.ipynb`](examples/ng15_stochastic_pe.ipynb) for red noise
and a Hellings–Downs background, and
[`ng15_cw_pe.ipynb`](examples/ng15_cw_pe.ipynb) for the same with a continuous
wave on top.


## How it works

Prometheus never touches TOAs while sampling. Everything it needs is:

| object | meaning | shape | `Data` attribute |
|---|---|---|---|
| `FNF`   | `F^T Ñ⁻¹ F`     | (Np, Nc, Nc) | `FNFs` |
| `FNr`   | `F^T Ñ⁻¹ r`     | (Np, Nc)     | `FNrs` |
| `FNFD`  | `F^T Ñ⁻¹ F_D`   | (Np, Nc, Nd) | `TNTDs` |
| `FDNFD` | `F_D^T Ñ⁻¹ F_D` | (Np, Nd, Nd) | `TDNTDs` |
| `FDNr`  | `F_D^T Ñ⁻¹ r`   | (Np, Nd)     | `TDNrs` |
| `rNr`   | `r^T Ñ⁻¹ r`     | (Np,)        | — |

`F` is the stochastic Fourier design matrix, `F_D` the deterministic one, `r`
the residuals, and `Ñ⁻¹` the white-noise inverse projected orthogonal to the
linearized timing model. Only the three objects involving `r` depend on the data
realization; the rest are fixed by the array, which is what makes re-injection
cheap.

For `r = F a + F_D a_det + n` with `n ~ N(0, N)`, the noise left in the two
channels after compression is exactly Gaussian, with covariance the compressed
Gram matrix:

```
(ξ_F, ξ_D) ~ N(0, C),   C = [[FNF, FNFD], [FNFDᵀ, FDNFD]]
```

The two channels are therefore drawn jointly from `C` itself, and `rNr` follows
in closed form. Everything runs in float64; the float32 arrays Prometheus stores
are produced once, at the end.
