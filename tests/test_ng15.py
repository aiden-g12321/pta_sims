"""
Integration tests against the real NG15 dataset.

Skipped when ``data/NG15/data.pkl`` or Prometheus is unavailable.  These are
the tests that exercise the realistic failure modes -- Gram matrices that are
rank deficient by a factor of ten, eigenvalues spanning fourteen orders of
magnitude, and a final cast to float32 -- which no synthetic fixture reproduces
faithfully.
"""

import os

import numpy as np
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'NG15', 'data.pkl')

pytest.importorskip('prometheus')
pytestmark = pytest.mark.skipif(not os.path.exists(DATA_PATH),
                                reason='NG15 data.pkl not available')


@pytest.fixture(scope='module')
def ng15():
    import pta_sims
    return pta_sims.load_data(DATA_PATH)


@pytest.fixture(scope='module')
def ng15_ip(ng15):
    import pta_sims
    return pta_sims.InnerProducts.from_data(ng15)


@pytest.fixture(scope='module')
def ng15_model(ng15):
    import pta_sims
    return pta_sims.build_standard_model(
        ng15, correlations='HD', det_model=pta_sims.build_cw_model(ng15))


@pytest.fixture(scope='module')
def params(ng15):
    import pta_sims
    return {
        'red_noise': pta_sims.power_law_params(-14.3, 3.2, npsrs=ng15.npsrs),
        'gwb': pta_sims.power_law_params(-14.6, 13 / 3),
        'cw': pta_sims.cw_params(9.0, -8.0, 0.5, 1.0, -14.0, 0.3, 2.0, 1.5),
    }


def test_real_data_lies_in_the_gram_column_space(ng15_ip):
    """
    The identity that justifies using the Gram matrix as the noise covariance,
    tested on the real compressed data rather than on a simulation of it.
    """
    from pta_sims.diagnostics import range_consistency
    frac = range_consistency(ng15_ip)['out_of_range_frac']
    assert np.max(frac) < 1e-10


def test_the_gram_matrices_are_severely_rank_deficient(ng15_ip):
    """
    Guards the design decision: if this ever stopped being true, a plain
    Cholesky would do and the eigen-truncation would be unnecessary machinery.
    """
    from pta_sims.simulate import CompressedNoise
    noise = CompressedNoise.from_inner_products(ng15_ip)
    assert np.max(noise.rank) < 0.6 * noise.size
    assert np.min(noise.rank) < 0.2 * noise.size


def test_end_to_end_simulation(ng15, ng15_model, params):
    import pta_sims
    sim = pta_sims.simulate(ng15_model, params, seed=0)

    assert np.asarray(sim.data.FNrs).shape == np.asarray(ng15.FNrs).shape
    assert np.asarray(sim.data.TDNrs).shape == np.asarray(ng15.TDNrs).shape
    assert np.all(np.isfinite(np.asarray(sim.data.FNrs)))
    assert np.all(np.isfinite(np.asarray(sim.data.TDNrs)))
    assert np.asarray(sim.data.FNrs).dtype == np.float32
    # geometry is shared, not copied
    assert np.array_equal(np.asarray(sim.data.FNFs), np.asarray(ng15.FNFs))
    assert sim.data.psrs is ng15.psrs
    assert sim.data.name != ng15.name


def test_float32_cast_costs_only_float32_epsilon(ng15_model, params):
    import pta_sims
    sim = pta_sims.simulate(ng15_model, params, seed=1)
    for key, rep in sim.diagnostics['downcast'].items():
        assert rep['n_nonfinite'] == 0, key
        assert rep['n_underflow'] == 0, key
        assert rep['max_rel_err'] < 1e-6, key


def test_white_noise_only_is_chi_squared(ng15_ip):
    from pta_sims.diagnostics import whitened_statistics
    from pta_sims.simulate import CompressedNoise, simulate_inner_products

    noise = CompressedNoise.from_inner_products(ng15_ip)
    chi2 = np.array([
        whitened_statistics(
            simulate_inner_products(ng15_ip, rng=s, noise=noise,
                                    simulate_rNr=False).inner_products,
            noise=noise)['chi2_per_dof']
        for s in range(60)])
    dof = int(np.sum(noise.rank))
    assert abs(chi2.mean() - 1.0) < 5 * np.sqrt(2.0 / dof / 60)


def test_recomputed_inner_products_reproduce_the_stored_ones(ng15):
    """
    Rebuilding F, F_D and the noise solve from scratch must return exactly the
    arrays the Data object was built with.  This is the check that our model of
    the compression -- which basis, which frequencies, which noise metric --
    matches Prometheus' own, and it also validates the N_TOA - N_tmpars degrees
    of freedom used to simulate rNr, since the real residuals are white-noise
    dominated.
    """
    import copy

    pytest.importorskip('enterprise')
    from pta_sims.inner_products import InnerProducts
    from pta_sims.whitenoise import recompute_inner_products

    keep = list(ng15.psr_names[:3])
    subset = copy.copy(ng15)
    subset.psrs = [p for p in ng15.psrs if p.name in keep]
    subset.psr_names = keep
    subset.per_psr_data_dict = {k: ng15.per_psr_data_dict[k] for k in keep}

    fresh = recompute_inner_products(subset, verbose=False)
    stored = InnerProducts.from_data(subset)

    for name in ('FNF', 'FNr', 'FNFD', 'FDNFD', 'FDNr'):
        new = getattr(fresh, name)
        old = getattr(stored, name)
        assert np.max(np.abs(new - old)) < 1e-10 * np.max(np.abs(old)), name

    ratio = fresh.rNr / (fresh.ntoas - fresh.ntmpars)
    assert np.all(np.abs(ratio - 1.0) < 0.2)


def test_rebuilt_data_is_internally_consistent(ng15_model, params):
    """Sigma_inv @ a_hat == FNr must still hold after the rebuild."""
    import pta_sims
    sim = pta_sims.simulate(ng15_model, params, seed=2)
    # equal up to the float32 cast: one side is FNr / renorm rounded, the
    # other Sigma_inv @ a_hat computed in float64 and then rounded
    assert np.allclose(np.asarray(sim.data.Si0_a_hat_j_pf),
                       np.asarray(sim.data.FNrs), rtol=1e-5, atol=0.0)
    names = list(sim.data.psr_names)
    stacked = np.concatenate([sim.data.per_psr_data_dict[n]['a_hat'] for n in names])
    assert np.allclose(np.asarray(sim.data.a_hat_j), stacked.astype(np.float32))


def test_prometheus_posterior_is_finite_on_simulated_data(ng15, ng15_model, params):
    import jax
    import jax.numpy as jnp
    from numpyro.infer.util import log_density

    import pta_sims
    sim = pta_sims.simulate(ng15_model, params, seed=3)
    site = {
        'red_noise': jnp.asarray(params['red_noise'], jnp.float32),
        'gwb': jnp.asarray(params['gwb'], jnp.float32),
        'cw': jnp.asarray(params['cw'], jnp.float32),
        'psr_phases': jnp.asarray(sim.truth['psr_phases'], jnp.float32),
        'standard_psr_dists': jnp.zeros(ng15.npsrs, jnp.float32),
        'z': jnp.zeros((ng15.npsrs, ng15.ncomponents), jnp.float32),
    }
    value, _ = log_density(sim.pta_model.sampling_model, (), {}, site)
    assert np.isfinite(float(value))
    grads = jax.grad(lambda s: log_density(sim.pta_model.sampling_model,
                                           (), {}, s)[0])(site)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads.values())


def test_chaining_from_a_simulated_dataset(ng15, ng15_model, params):
    """NG15-like data can itself be used as the template for another sim."""
    import pta_sims
    first = pta_sims.simulate(ng15_model, params, seed=4)
    second_model = pta_sims.build_standard_model(
        first.data, correlations='HD',
        det_model=pta_sims.build_cw_model(first.data))
    second = pta_sims.simulate(second_model, params, seed=5)

    assert np.array_equal(np.asarray(second.data.FNFs), np.asarray(ng15.FNFs))
    assert not np.array_equal(np.asarray(second.data.FNrs),
                              np.asarray(first.data.FNrs))


def test_noise_factorization_is_reusable(ng15_ip, ng15_model, params):
    import pta_sims
    from pta_sims.simulate import CompressedNoise
    noise = CompressedNoise.from_inner_products(ng15_ip)
    a = pta_sims.simulate(ng15_model, params, seed=6, noise=noise)
    b = pta_sims.simulate(ng15_model, params, seed=6)
    assert np.array_equal(np.asarray(a.data.FNrs), np.asarray(b.data.FNrs))


def test_matched_pairs_share_a_noise_realization(ng15, ng15_model, params):
    """
    Through the high-level entry point: handing in ``a`` and ``a_det`` outright
    leaves the random stream untouched until the noise draw, so a null dataset
    and a continuous-wave dataset built from the same seed differ by exactly
    the injected wave.  Downstream injection campaigns depend on this being
    bit-exact.
    """
    import pta_sims

    npsrs = ng15.npsrs
    rng = np.random.default_rng(0)
    a = rng.standard_normal((npsrs, ng15.ncomponents)) * 50.0
    a_det = rng.standard_normal((npsrs, ng15.num_coeff_det)) * 5.0
    zero_det = np.zeros_like(a_det)

    from pta_sims.diagnostics import signal_vector

    common = dict(seed=1234, a=a, simulate_rNr=False)
    null = pta_sims.simulate(ng15_model, params, a_det=zero_det, **common)
    signal = pta_sims.simulate(ng15_model, params, a_det=a_det, **common)

    difference = (signal.inner_products.joint_vector()
                  - null.inner_products.joint_vector())
    expected = signal_vector(null.reference_inner_products,
                             np.zeros_like(a), a_det)

    # The joint vector spans eight orders of magnitude on this array, so a
    # bare rtol would demand 1e-10 relative agreement on the handful of
    # elements that cancellation leaves near zero.  Hold the comparison to
    # float64 rounding of the vector's own scale instead.
    atol = 1e-10 * np.max(np.abs(expected))
    assert np.allclose(difference, expected, rtol=1e-10, atol=atol)


# ---------------------------------------------------------------------------
# Compatibility with Prometheus' current data and model API.
#
# Prometheus now puts each stochastic process on a *named* Fourier basis, takes
# any number of per-pulsar processes, makes the common process optional, and
# enables JAX's x64 mode globally while still storing its ``Data`` arrays in
# float32.  Each of these tests pins one place where pta_sims has to follow.
# ---------------------------------------------------------------------------

def test_rebind_keeps_every_per_pulsar_process(ng15):
    """
    ``psr_model`` is only the first of ``psr_models``.  Rebinding from it alone
    silently drops the others, which costs the simulation a whole process.
    """
    import pta_sims
    from prometheus import spectra
    from prometheus.pta_model import PTAModel
    from prometheus.spectral_models import (CommonSpectralModel,
                                            IndependentSpectralModel)

    def independent(name):
        return IndependentSpectralModel(
            name=name, parameter_bounds=[[-20., -11.], [0., 7.]], data=ng15,
            get_phi_diag_func=spectra.power_law)

    model = PTAModel(
        psr_model=[independent('rn_a'), independent('rn_b')],
        gwb_model=CommonSpectralModel(
            name='gwb', parameter_bounds=[[-18., -11.], [0., 7.]], data=ng15,
            get_phi_diag_func=spectra.power_law, correlation_matrix='HD'))

    rebound = pta_sims.rebind_model(model, ng15)
    assert [m.name for m in rebound.psr_models] == ['rn_a', 'rn_b']
    assert rebound.sampling == model.sampling
    assert all(m.data is ng15 for m in rebound.psr_models)


def test_simulate_without_a_common_process(ng15):
    """Prometheus allows gwb_model=None; a noise-only model must simulate."""
    import pta_sims

    model = pta_sims.build_standard_model(ng15, correlations=None)
    assert model.gwb_model is None

    sim = pta_sims.simulate(
        model, {'red_noise': pta_sims.power_law_params(-14.3, 3.2,
                                                       npsrs=ng15.npsrs)},
        seed=5)
    assert sim.truth['a'].shape == (ng15.npsrs, ng15.ncomponents)
    assert np.all(np.isfinite(np.asarray(sim.data.FNrs)))


def test_build_phi_cube_rejects_a_process_on_another_basis(ng15):
    """
    A single ``(Nc, Np, Np)`` cube cannot describe processes living on
    different bases.  Saying so beats returning a cube that quietly leaves one
    of them out.
    """
    import pta_sims

    model = pta_sims.build_standard_model(ng15, correlations='HD')
    off_basis = model.psr_model
    off_basis.basis = 'dm'      # pretend it lives elsewhere

    with pytest.raises(ValueError, match='single shared Fourier basis'):
        pta_sims.build_phi_cube(model, {
            model.psr_model.name: pta_sims.power_law_params(-14.3, 3.2,
                                                            npsrs=ng15.npsrs),
            model.gwb_model.name: pta_sims.power_law_params(-14.6, 13 / 3)})

    off_basis.basis = 'red'


def test_simulated_data_has_the_dtypes_a_real_one_has(ng15_model, params, ng15):
    """
    Prometheus stores its ``Data`` arrays in float32 but now runs JAX in x64
    mode, so nothing downcasts a rebuilt field by accident any more.  A
    simulated dataset that carried float64 where a real one carries float32
    would promote the dtype of every jitted model that saw it.
    """
    import pta_sims

    sim = pta_sims.simulate(ng15_model, params, seed=6)
    for name in ('FNFs', 'FNrs', 'TNTDs', 'TDNTDs', 'TDNrs'):
        reference = getattr(ng15, name)
        assert getattr(sim.data, name).dtype == reference.dtype, name


def test_rebuild_refuses_a_basis_it_cannot_simulate(ng15, ng15_ip):
    """
    Every extra basis has its own F_b^T N^-1 r, which the marginalized sampling
    model reads through ``data.basis_Nr``.  pta_sims can re-simulate 'red' and
    'dm'; leaving any other basis alone would mix real residuals into the
    simulation, so it must refuse rather than return that.
    """
    import copy

    from pta_sims.datasets import rebuild_data

    stub = copy.copy(ng15)
    stub.extra_bases = list(getattr(ng15, 'extra_bases', [])) + [
        type('Basis', (), {'name': 'scattering', 'nfreqs': 4})()]

    with pytest.raises(NotImplementedError, match='scattering'):
        rebuild_data(stub, ng15_ip)
