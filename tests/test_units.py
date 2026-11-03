"""The renorm scaling convention."""

import numpy as np

from pta_sims import units


def test_round_trip():
    x = np.array([1.0, 2.0, 3.0])
    for n in (0, 1, 2):
        assert np.allclose(units.from_renorm(units.to_renorm(x, n), n), x)


def test_basis_index_counting():
    x = np.array([1.0])
    assert np.allclose(units.to_renorm(x, 1), 1.0 / units.RENORM)
    assert np.allclose(units.to_renorm(x, 2), 1.0 / units.RENORM ** 2)


def test_quadratic_forms_are_invariant():
    """a . FNr and a^T FNF a must not care about the renormalization."""
    rng = np.random.default_rng(0)
    n = 6
    a_sec = rng.standard_normal(n) * 1e-7
    FNF = rng.standard_normal((n, n)) * 1e12
    FNF = FNF @ FNF.T
    FNr = FNF @ a_sec

    a_ns = units.seconds_to_ns(a_sec)
    assert np.isclose(a_ns @ units.to_renorm(FNr, 1), a_sec @ FNr, rtol=1e-12)
    assert np.isclose(a_ns @ units.to_renorm(FNF, 2) @ a_ns,
                      a_sec @ FNF @ a_sec, rtol=1e-12)


def test_matches_prometheus_renorm():
    prometheus = __import__('importlib').util.find_spec('prometheus')
    if prometheus is None:
        return
    from prometheus import utilities as putils
    assert float(putils.renorm) == units.RENORM


def test_rebuilt_fields_are_cast_to_the_dtype_prometheus_stores():
    """
    ``Data.__init__`` casts every float array it holds to float32.  Rebuilt
    fields used to inherit that cast from ``jnp.asarray``, which downcast
    silently while JAX ran with x64 off; Prometheus now turns x64 on globally,
    so the cast has to be explicit or the dtype depends on import order.
    """
    import pytest
    pytest.importorskip('jax')

    import jax
    from pta_sims.datasets import _to_array

    with jax.experimental.enable_x64():
        out = _to_array(np.array([1.0, 2.0], dtype=np.float64))
    assert out.dtype == units.PROMETHEUS_DTYPE

    # integer fields are not floating point and must come through untouched
    assert _to_array(np.array([1, 2], dtype=np.int32)).dtype == np.int32
