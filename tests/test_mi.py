import numpy as np
import pytest

from infoplane.mi import (
    bin_activations,
    entropy,
    joint_symbols,
    layer_information,
    mutual_information,
)


def test_entropy_of_uniform_is_log2_n():
    assert entropy(np.arange(4096)) == pytest.approx(12.0)
    assert entropy(np.zeros(100, dtype=int)) == pytest.approx(0.0)


def test_mi_of_a_variable_with_itself_is_its_entropy():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 7, size=5000)
    assert mutual_information(a, a) == pytest.approx(entropy(a))


def test_mi_of_independent_variables_is_zero_for_a_product_distribution():
    # Exact product distribution (every pair equally often) -> exactly zero.
    a, b = np.divmod(np.arange(6 * 5), 5)
    assert mutual_information(a, b) == pytest.approx(0.0, abs=1e-12)


def test_mi_of_a_deterministic_coarsening_is_the_coarser_entropy():
    a = np.arange(1000)
    b = a % 4  # a determines b; b is uniform over 4 values
    assert mutual_information(a, b) == pytest.approx(2.0)


def test_mi_is_symmetric_and_nonnegative():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 2000)
    b = (a + rng.integers(0, 2, 2000)) % 5
    assert mutual_information(a, b) == pytest.approx(mutual_information(b, a))
    assert mutual_information(a, b) > 0


def test_mi_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mutual_information(np.arange(3), np.arange(4))


def test_binning_uses_the_fixed_support_and_clips():
    acts = np.array([[-1.0, -0.999, 0.0, 0.999, 1.0, 5.0, -5.0]])
    idx = bin_activations(acts, bins=30)
    assert idx[0, 0] == 0  # left edge
    assert idx[0, 2] == 15  # zero sits at the midpoint bin
    assert idx[0, 4] == 29  # right edge folds into the last bin, not a 31st
    assert idx[0, 5] == 29 and idx[0, 6] == 0  # out-of-range activations clip


def test_binning_resolution_matches_bin_count():
    # Two activations one bin width apart must land in different bins.
    acts = np.array([[0.0], [2.0 / 30 + 1e-9]])
    assert len(np.unique(bin_activations(acts, bins=30))) == 2


def test_joint_symbols_distinguishes_only_whole_bin_tuples():
    binned = np.array([[1, 2], [1, 2], [1, 3]])
    sym = joint_symbols(binned)
    assert sym[0] == sym[1] != sym[2]

    with pytest.raises(ValueError):
        joint_symbols(np.zeros(5))


def test_layer_information_bounds_and_limits():
    rng = np.random.default_rng(2)
    n = 512
    x = np.arange(n)
    y = (x % 2).astype(np.int64)

    # A layer that is a copy of a distinct-valued input: I(X;T) = H(X), and it
    # retains all of the label information.
    acts = np.linspace(-1, 1, n).reshape(-1, 1)
    ixt, ity = layer_information(acts, x, y, bins=n)
    assert ixt == pytest.approx(entropy(x))
    assert ity == pytest.approx(entropy(y))

    # A constant layer carries nothing about either.
    ixt, ity = layer_information(np.zeros((n, 3)), x, y)
    assert ixt == pytest.approx(0.0) and ity == pytest.approx(0.0)

    # Random activations are bounded by H(X) and H(Y).
    ixt, ity = layer_information(rng.uniform(-1, 1, (n, 4)), x, y)
    assert 0 <= ixt <= entropy(x) + 1e-12
    assert 0 <= ity <= entropy(y) + 1e-12
