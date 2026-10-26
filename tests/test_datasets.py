import numpy as np
import pytest

from infoplane.datasets import build_dataset, make_toy
from infoplane.mi import entropy, mutual_information


@pytest.fixture(scope="module")
def toy():
    return make_toy()


def test_every_input_pattern_appears_exactly_once(toy):
    assert toy.n_samples == 4096
    assert toy.n_features == 12
    assert len(np.unique(toy.X, axis=0)) == toy.n_samples


def test_entropy_of_the_input_is_exactly_12_bits(toy):
    x_sym = np.unique(toy.X, axis=0, return_inverse=True)[1].reshape(-1)
    assert entropy(x_sym) == pytest.approx(12.0)


def test_inputs_are_spins_matching_the_tanh_range(toy):
    assert set(np.unique(toy.X).tolist()) == {-1.0, 1.0}


def test_labels_are_near_balanced_and_deterministic(toy):
    assert toy.Y.mean() == pytest.approx(0.5, abs=0.02)
    assert entropy(toy.Y) == pytest.approx(1.0, abs=0.01)
    # Y is a function of X, so I(X;Y) = H(Y) -- the network has 1 bit to recover.
    x_sym = np.unique(toy.X, axis=0, return_inverse=True)[1].reshape(-1)
    assert mutual_information(x_sym, toy.Y) == pytest.approx(entropy(toy.Y))


def test_the_label_is_a_function_of_the_latent_radius(toy):
    # The threshold must cut between distinct radii, never through a tie group,
    # or the boundary would be unlearnable at that radius.
    radius, theta = toy.extras["radius"], float(toy.extras["threshold"])
    assert np.array_equal(toy.Y, (radius >= theta).astype(np.int64))
    for r in np.unique(radius):
        labels = np.unique(toy.Y[radius == r])
        assert len(labels) == 1


def test_the_boundary_is_not_linearly_separable(toy):
    # A least-squares linear probe on the spins should be at chance: the rule is a
    # radius in the latent plane and has no linear component.
    A = np.concatenate([toy.X, np.ones((toy.n_samples, 1), np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(A, 2.0 * toy.Y - 1.0, rcond=None)
    acc = ((A @ w > 0).astype(np.int64) == toy.Y).mean()
    assert acc == pytest.approx(0.5, abs=0.05)


def test_sphere_geometry_gives_a_higher_rank_rule():
    circle, sphere = make_toy(geometry="circle"), make_toy(geometry="sphere")
    # The circle rule is degenerate (antipodal directions cancel); the rank-3
    # variant separates every pattern.
    assert len(np.unique(circle.extras["radius"])) < len(np.unique(sphere.extras["radius"]))
    assert entropy(sphere.Y) == pytest.approx(1.0, abs=0.01)


def test_build_dataset_dispatch_and_errors():
    assert build_dataset("toy", n_bits=8).n_samples == 256
    with pytest.raises(ValueError):
        build_dataset("matfile")
    with pytest.raises(ValueError):
        build_dataset("nope")
