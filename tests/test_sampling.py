"""
The monitored NUTS driver.

Exercised on a small analytically-known numpyro model rather than the PTA
posterior, so the tests run on CPU in seconds and any failure is a failure of
the driver rather than of the model.
"""

import numpy as np
import pytest

pytest.importorskip('numpyro')

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from pta_sims.sampling import (effective_sample_size, run_nuts_monitored,
                               unconstrained_init)


TRUE_MU = np.array([0.35, -0.20, 0.60])
SIGMA = 0.15


def toy_model():
    """Three bounded parameters with a Gaussian likelihood pinning them."""
    mu = numpyro.sample('mu', dist.Uniform(-1.0, 1.0).expand([3]))
    scale = numpyro.sample('scale', dist.Uniform(0.05, 2.0))
    numpyro.factor('lik', -0.5 * jnp.sum((mu - TRUE_MU) ** 2) / SIGMA ** 2
                   - 0.5 * (scale - 1.0) ** 2 / 0.3 ** 2)


def test_unconstrained_init_round_trips():
    values = {'mu': TRUE_MU, 'scale': 0.8}
    init, potential_fn, transforms = unconstrained_init(toy_model, values)

    assert set(init) == {'mu', 'scale'}
    assert np.allclose(np.asarray(transforms['mu'](init['mu'])), TRUE_MU,
                       atol=1e-5)
    assert np.isclose(float(transforms['scale'](init['scale'])), 0.8, atol=1e-5)
    assert np.isfinite(float(potential_fn(init)))


def test_recovers_the_truth():
    samples, diag = run_nuts_monitored(
        toy_model, {'mu': TRUE_MU, 'scale': 0.8},
        num_warmup=300, num_samples=800, seed=1, report_every=10_000)

    assert samples['mu'].shape == (800, 3)
    assert samples['scale'].shape == (800,)
    assert np.allclose(np.median(samples['mu'], axis=0), TRUE_MU, atol=0.05)
    # bounded priors must be respected after the transform back
    assert samples['mu'].min() > -1.0 and samples['mu'].max() < 1.0
    assert samples['scale'].min() > 0.05 and samples['scale'].max() < 2.0

    assert 0.5 < diag.mean_accept_prob <= 1.0
    assert diag.step_size > 1e-3
    assert diag.warmup_step_size.shape == (300,)
    assert len(diag.attempts) == 1 and diag.attempts[0].outcome == 'ok'


def test_matches_numpyro_mcmc_in_distribution():
    """The hand-rolled loop must sample the same posterior numpyro's does."""
    import jax.random as jr
    from numpyro.infer import MCMC, NUTS, init_to_value

    mcmc = MCMC(NUTS(toy_model, init_strategy=init_to_value(
        values={'mu': TRUE_MU, 'scale': 0.8})),
        num_warmup=300, num_samples=1500, progress_bar=False)
    mcmc.run(jr.PRNGKey(3))
    reference = mcmc.get_samples()

    ours, _ = run_nuts_monitored(toy_model, {'mu': TRUE_MU, 'scale': 0.8},
                                 num_warmup=300, num_samples=1500, seed=3,
                                 report_every=10_000)

    ref_mu = np.asarray(reference['mu'])
    assert np.allclose(np.mean(ours['mu'], axis=0), np.mean(ref_mu, axis=0),
                       atol=0.02)
    assert np.allclose(np.std(ours['mu'], axis=0), np.std(ref_mu, axis=0),
                       rtol=0.15)


def test_progress_callback_reports_live_diagnostics():
    seen = []
    run_nuts_monitored(toy_model, {'mu': TRUE_MU}, num_warmup=100,
                       num_samples=100, seed=2, report_every=25,
                       progress=seen.append)

    assert {r['phase'] for r in seen} == {'warmup', 'sample'}
    assert all(np.isfinite(r['step_size']) for r in seen)
    assert all(r['tree_depth'] >= 1 for r in seen)
    assert seen[-1]['iteration'] == 200


def test_watchdog_fires_and_gives_up():
    """
    An impossible floor makes every attempt look collapsed, so the driver must
    exhaust its attempts and say so rather than returning a bad chain.
    """
    with pytest.raises(RuntimeError, match='failed to warm up in 2 attempts'):
        run_nuts_monitored(toy_model, {'mu': TRUE_MU}, num_warmup=300,
                           num_samples=50, seed=4, max_attempts=2,
                           step_size_floor=1e9, saturation_fraction=0.0,
                           check_after=50, report_every=10_000)


def test_watchdog_needs_both_signals():
    """A collapsed step size alone is not enough; the tree must saturate too."""
    samples, diag = run_nuts_monitored(
        toy_model, {'mu': TRUE_MU}, num_warmup=200, num_samples=100, seed=5,
        step_size_floor=1e9,            # every step size counts as collapsed
        saturation_fraction=1.01,       # but saturation can never be met
        check_after=50, report_every=10_000)
    assert samples['mu'].shape == (100, 3)
    assert diag.attempts[0].outcome == 'ok'


def test_watchdog_never_fires_after_warmup():
    """Post-warm-up iterations are exempt, so a short warm-up always finishes."""
    samples, diag = run_nuts_monitored(
        toy_model, {'mu': TRUE_MU}, num_warmup=60, num_samples=200, seed=6,
        step_size_floor=1e9, saturation_fraction=0.0, check_after=1000,
        report_every=10_000)
    assert samples['mu'].shape == (200, 3)


def test_wall_clock_limit_is_enforced():
    with pytest.raises(RuntimeError, match='exceeded'):
        run_nuts_monitored(toy_model, {'mu': TRUE_MU}, num_warmup=100_000,
                           num_samples=10, seed=8, max_attempts=1,
                           wall_clock_limit=1.0, report_every=10_000)


def test_collect_restricts_what_is_stored():
    samples, _ = run_nuts_monitored(toy_model, {'mu': TRUE_MU}, num_warmup=80,
                                    num_samples=80, seed=9, collect=['scale'],
                                    report_every=10_000)
    assert set(samples) == {'scale'}


def test_effective_sample_size():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4000)
    assert effective_sample_size(white) > 2000

    # AR(1) with rho = 0.8 has ESS ~ N (1 - rho) / (1 + rho) = N / 9
    correlated = np.empty(4000)
    correlated[0] = rng.standard_normal()
    for i in range(1, 4000):
        correlated[i] = 0.8 * correlated[i - 1] + rng.standard_normal()
    ess = effective_sample_size(correlated)
    assert 4000 / 20 < ess < 4000 / 4
    assert effective_sample_size(np.ones(100)) == 100.0


def test_checkpoint_receives_growing_prefixes_of_the_final_chain():
    seen = []
    samples, _ = run_nuts_monitored(
        toy_model, {'mu': jnp.zeros(3), 'scale': jnp.float32(1.0)},
        num_warmup=50, num_samples=120, seed=3, report_every=1000,
        checkpoint=lambda s, info: seen.append((info['iteration'], {k: v.copy() for k, v in s.items()})),
        checkpoint_every=50)
    iterations = [it for it, _ in seen]
    assert iterations == [100, 150, 170]          # 50 warm-up + every 50 samples, and the end
    for _, partial in seen:
        n = partial['mu'].shape[0]
        assert np.array_equal(partial['mu'], samples['mu'][:n])
    assert seen[-1][1]['mu'].shape[0] == 120


def test_mixed_dtype_starting_values_are_accepted():
    """
    A starting point is assembled by hand, so its dtypes need not agree.  They
    used to agree by accident, because JAX ran x64 off and cast everything to
    float32; Prometheus now turns x64 on globally, and a mixed dict would reach
    NUTS intact and fail inside its tree-doubling loop with an error naming
    neither the site nor the caller.
    """
    values = {'mu': np.float64(TRUE_MU),      # float64
              'scale': jnp.float32(1.0)}      # float32
    samples, _ = run_nuts_monitored(toy_model, values, num_warmup=60,
                                    num_samples=80, seed=11,
                                    report_every=10_000)
    assert samples['mu'].shape == (80, 3)
    assert np.all(np.isfinite(samples['scale']))
