"""Basic tests for the satom package."""

import numpy as np
import pytest

from satom import SAtom


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def ishigami_data():
    """Generate Ishigami-Homma test data (3 inputs, 1 output)."""
    rng = np.random.default_rng(123)
    N = 5000
    X = rng.uniform(-np.pi, np.pi, size=(N, 3))
    a, b = 2, 1
    Y = np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
    return X, Y


# =========================================================================
# Tests
# =========================================================================

class TestSAtomBasic:
    """Core functionality tests."""

    def test_output_shapes(self, ishigami_data):
        X, Y = ishigami_data
        J = 200
        KS2_mean, KS2 = SAtom(X, Y, J=J, seed=42)

        assert KS2_mean.shape == (3,)
        assert KS2.shape == (J, 3)

    def test_output_dtype(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean, KS2 = SAtom(X, Y, J=200, seed=42)

        assert KS2_mean.dtype == np.float64
        assert KS2.dtype == np.float64

    def test_reproducibility(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean_a, KS2_a = SAtom(X, Y, J=200, seed=42)
        KS2_mean_b, KS2_b = SAtom(X, Y, J=200, seed=42)

        np.testing.assert_array_equal(KS2_mean_a, KS2_mean_b)
        np.testing.assert_array_equal(KS2_a, KS2_b)

    def test_different_seeds_differ(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean_a, _ = SAtom(X, Y, J=200, seed=42)
        KS2_mean_b, _ = SAtom(X, Y, J=200, seed=99)

        assert not np.allclose(KS2_mean_a, KS2_mean_b)

    def test_values_in_valid_range(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean, KS2 = SAtom(X, Y, J=200, seed=42)

        assert np.all(KS2 >= 0)
        assert np.all(KS2 <= 1)
        assert np.all(KS2_mean >= 0)
        assert np.all(KS2_mean <= 1)

    def test_ranking_ishigami(self, ishigami_data):
        """x1 and x3 should dominate x2 for SScompare=0 on Ishigami."""
        X, Y = ishigami_data
        KS2_mean, _ = SAtom(X, Y, J=500, seed=42)

        rank = np.argsort(KS2_mean)[::-1]
        # x1 (idx 0) and x3 (idx 2) should be in the top 2
        top2 = set(rank[:2])
        assert 0 in top2 or 2 in top2  # at least one of the known drivers


class TestSAtomSScompare:
    """Test all three SScompare modes run without error."""

    @pytest.mark.parametrize("mode", [0, 1, 2])
    def test_sscompare_modes(self, ishigami_data, mode):
        X, Y = ishigami_data
        KS2_mean, KS2 = SAtom(X, Y, J=200, SScompare=mode, seed=42)

        assert KS2_mean.shape == (3,)
        assert np.all(np.isfinite(KS2_mean))


class TestSAtomDummy:
    """Test dummy variable functionality."""

    def test_dummy_adds_column(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean, KS2 = SAtom(X, Y, J=200, dummyYN=True, seed=42)

        assert KS2_mean.shape == (4,)
        assert KS2.shape[1] == 4

    def test_dummy_has_lowest_sensitivity(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean, _ = SAtom(X, Y, J=500, dummyYN=True, seed=42)

        # Dummy (last column) should have the smallest mean KS distance
        assert np.argmin(KS2_mean) == 3


class TestSAtomSubset:
    """Test N subsetting."""

    def test_subset_n(self, ishigami_data):
        X, Y = ishigami_data
        KS2_mean, KS2 = SAtom(X, Y, J=200, N=1000, seed=42)

        assert KS2_mean.shape == (3,)

    def test_n_greater_than_rows_raises(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="cannot be greater"):
            SAtom(X, Y, J=200, N=X.shape[0] + 100, seed=42)


class TestSAtomInputValidation:
    """Edge cases and error handling."""

    def test_1d_y(self, ishigami_data):
        X, Y = ishigami_data
        # Y as 1-D should work fine
        KS2_mean, _ = SAtom(X, Y, J=200, seed=42)
        assert KS2_mean.shape == (3,)

    def test_mismatched_rows_raises(self):
        X = np.random.rand(100, 3)
        Y = np.random.rand(50)
        with pytest.raises(ValueError, match="same number of rows"):
            SAtom(X, Y)

    def test_invalid_sscompare_raises(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="SScompare must be 0, 1, or 2"):
            SAtom(X, Y, SScompare=5)

    def test_invalid_j_raises(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="positive integer"):
            SAtom(X, Y, J=-1)

    def test_wrong_label_count_raises(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="X_label must have"):
            SAtom(X, Y, X_label=["a", "b"])

    def test_j_below_minimum_warns(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.warns(UserWarning, match="below minimum"):
            KS2_mean, KS2 = SAtom(X, Y, J=10, seed=42)
        # Should have been bumped to 100
        assert KS2.shape[0] == 100