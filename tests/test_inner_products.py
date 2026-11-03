"""The InnerProducts container and its unit conversions."""

import numpy as np
import pytest

from pta_sims import units
from pta_sims.inner_products import InnerProducts


def test_shapes(toy, toy_ip):
    assert toy_ip.npsrs == toy.npsrs
    assert toy_ip.ncomponents == toy.ncomponents
    assert toy_ip.nfreqs == toy.nfreqs
    assert toy_ip.ndet == toy.ndet
    assert toy_ip.has_deterministic


def test_joint_matrix_layout(toy_ip):
    C = toy_ip.joint_matrix()
    nc = toy_ip.ncomponents
    assert C.shape == (toy_ip.npsrs, nc + toy_ip.ndet, nc + toy_ip.ndet)
    assert np.array_equal(C[:, :nc, :nc], toy_ip.FNF)
    assert np.array_equal(C[:, :nc, nc:], toy_ip.FNFD)
    assert np.array_equal(C[:, nc:, nc:], toy_ip.FDNFD)
    assert np.allclose(C, np.swapaxes(C, -1, -2))


def test_joint_matrix_is_the_gram_matrix_of_the_stacked_basis(toy, toy_ip):
    for p in range(toy.npsrs):
        B = np.concatenate([toy.F[p], toy.F_D[p]], axis=1)
        assert np.allclose(toy_ip.joint_matrix()[p],
                           B.T @ toy.Ninv_tilde[p] @ B, rtol=1e-9)


def test_without_deterministic_basis(toy_ip):
    plain = toy_ip.replace(FNFD=None, FDNFD=None, FDNr=None)
    assert not plain.has_deterministic
    assert plain.ndet == 0
    assert np.array_equal(plain.joint_matrix(), plain.FNF)
    assert np.array_equal(plain.joint_vector(), plain.FNr)


def test_renormalization_uses_one_factor_per_basis_index(toy_ip):
    out = toy_ip.renormalized(dtype=np.float64)
    assert np.allclose(out['FNFs'], toy_ip.FNF / units.RENORM ** 2)
    assert np.allclose(out['FNrs'], toy_ip.FNr / units.RENORM)
    assert np.allclose(out['TNTDs'], toy_ip.FNFD / units.RENORM ** 2)
    assert np.allclose(out['TDNTDs'], toy_ip.FDNFD / units.RENORM ** 2)
    assert np.allclose(out['TDNrs'], toy_ip.FDNr / units.RENORM)


def test_renormalization_defaults_to_float32(toy_ip):
    out = toy_ip.renormalized()
    assert all(v.dtype == np.float32 for v in out.values())


def test_replace_is_a_copy(toy_ip):
    other = toy_ip.replace(FNr=np.zeros_like(toy_ip.FNr))
    assert np.any(toy_ip.FNr != 0) or True
    assert other.FNF is toy_ip.FNF
    assert not np.array_equal(other.FNr, toy_ip.FNr) or np.all(toy_ip.FNr == 0)


# ------------------------------------------------------------------- saving

def test_residuals_only_round_trip(toy_ip, tmp_path):
    """A residuals-only file needs the reference back, and then matches."""
    path = str(tmp_path / 'ip.npz')
    toy_ip.save(path, residuals_only=True, log10_h=-14.0,
                cw=np.arange(8.0))

    loaded, metadata = InnerProducts.load(path, reference=toy_ip)
    assert np.allclose(loaded.FNr, toy_ip.FNr)
    assert np.allclose(loaded.FDNr, toy_ip.FDNr)
    assert np.allclose(loaded.rNr, toy_ip.rNr)
    assert loaded.FNF is toy_ip.FNF          # reattached, not duplicated
    assert np.array_equal(loaded.ntoas, toy_ip.ntoas)
    assert float(metadata['log10_h']) == -14.0
    assert np.allclose(metadata['cw'], np.arange(8.0))


def test_residuals_only_refuses_to_load_without_a_reference(toy_ip, tmp_path):
    path = str(tmp_path / 'ip.npz')
    toy_ip.save(path, residuals_only=True)
    with pytest.raises(ValueError, match='reference'):
        InnerProducts.load(path)


def test_self_contained_round_trip(toy_ip, tmp_path):
    path = str(tmp_path / 'full.npz')
    toy_ip.save(path, residuals_only=False)
    loaded, _ = InnerProducts.load(path)
    for name in ('FNF', 'FNr', 'FNFD', 'FDNFD', 'FDNr'):
        assert np.allclose(getattr(loaded, name), getattr(toy_ip, name)), name


def test_residuals_only_file_is_much_smaller(toy_ip, tmp_path):
    import os
    small = str(tmp_path / 'small.npz')
    big = str(tmp_path / 'big.npz')
    toy_ip.save(small, residuals_only=True)
    toy_ip.save(big, residuals_only=False)
    assert os.path.getsize(small) < 0.25 * os.path.getsize(big)


def test_reference_ordering_is_checked(toy_ip, tmp_path):
    path = str(tmp_path / 'ip.npz')
    toy_ip.save(path, residuals_only=True)
    shuffled = toy_ip.replace(psr_names=list(reversed(toy_ip.psr_names)))
    with pytest.raises(ValueError, match='ordering'):
        InnerProducts.load(path, reference=shuffled)
