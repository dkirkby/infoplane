from pathlib import Path

import numpy as np
import pytest

from infoplane.datasets import (
    DEFAULT_VAR_U_PATH,
    VAR_U_SHA256,
    build_dataset,
    load_matfile,
    make_quadric,
)
from infoplane.mi import entropy, mutual_information

# The authors' file is fetched, not vendored (`infoplane fetch`), so the tests that
# compare against it skip when it is absent.
var_u_only = pytest.mark.skipif(
    not DEFAULT_VAR_U_PATH.exists(), reason=f"{DEFAULT_VAR_U_PATH} not fetched"
)


@pytest.fixture(scope="module")
def var_u():
    """The authors' file with its raw 0/1 inputs, for tests about its structure.

    The pipeline defaults to `spin` (-+1) inputs instead; see
    test_var_u_default_encoding_is_spin and the README for why.
    """
    if not DEFAULT_VAR_U_PATH.exists():
        pytest.skip(f"{DEFAULT_VAR_U_PATH} not fetched")
    return load_matfile(DEFAULT_VAR_U_PATH, encoding="binary")


def _x_symbols(d):
    return np.unique(d.X, axis=0, return_inverse=True)[1].reshape(-1)


@var_u_only
def test_var_u_file_is_the_expected_one():
    import hashlib

    assert hashlib.sha256(Path(DEFAULT_VAR_U_PATH).read_bytes()).hexdigest() == VAR_U_SHA256


@var_u_only
def test_var_u_has_the_properties_the_estimator_needs(var_u):
    assert var_u.n_samples == 4096 and var_u.n_features == 12
    assert len(np.unique(var_u.X, axis=0)) == 4096
    assert entropy(_x_symbols(var_u)) == pytest.approx(12.0)
    assert var_u.Y.mean() == pytest.approx(0.5171, abs=1e-3)
    assert entropy(var_u.Y) == pytest.approx(0.9992, abs=1e-3)


@var_u_only
def test_var_u_shell_profile_is_monotone_in_the_bit_count(var_u):
    # Shells 0-4 are all zero, 8-12 all one, and only 5/6/7 are mixed. The *profile*
    # is monotone in the bit count even though the labels are not a function of it
    # (see the quadratic-form test below) -- which is what makes a linear probe on
    # the count reach 87.5% while still being far from the rule.
    k = var_u.extras["count"]
    profile = {int(kk): var_u.Y[k == kk].mean() for kk in np.unique(k)}
    assert all(profile[kk] == 0.0 for kk in range(5))
    assert all(profile[kk] == 1.0 for kk in range(8, 13))
    assert profile[5] == pytest.approx(0.0758, abs=1e-3)
    assert profile[6] == pytest.approx(0.5758, abs=1e-3)
    assert profile[7] == pytest.approx(0.9242, abs=1e-3)


@var_u_only
def test_var_u_labels_are_exactly_a_thresholded_quadratic_form(var_u):
    """The rule is deterministic, and quadratic -- not count-only, not linear.

    This pins down what the paper's prose leaves vague, and it is the reason no
    generator in this package claims to reproduce these labels. If this ever fails,
    the file changed.
    """
    import itertools

    from scipy.optimize import linprog

    F = var_u.X.astype(np.float64)
    pairs = list(itertools.combinations(range(12), 2))
    quad = np.stack([F[:, i] * F[:, j] for i, j in pairs], axis=1)
    A = np.concatenate([F, quad, np.ones((len(F), 1))], axis=1)  # 12 + 66 + 1
    s = 2.0 * var_u.Y - 1.0

    def separable(features):
        res = linprog(
            c=np.zeros(features.shape[1]),
            A_ub=-(s[:, None] * features),
            b_ub=-np.ones(len(features)),
            bounds=[(None, None)] * features.shape[1],
            method="highs",
        )
        return res.status == 0, res.x

    ok, w = separable(A)
    assert ok, "labels are not exactly separable by a quadratic threshold"
    assert np.all((A @ w > 0) == var_u.Y)

    linear_only = np.concatenate([F, np.ones((len(F), 1))], axis=1)
    assert not separable(linear_only)[0], "a linear threshold should NOT suffice"


@var_u_only
def test_var_u_default_encoding_is_spin():
    # The labels and patterns are identical either way -- the encoding is a network
    # input scaling, not part of the dataset -- but `spin` keeps tanh off its rails
    # so the binning estimator stays well conditioned.
    d = load_matfile(DEFAULT_VAR_U_PATH)
    assert set(np.unique(d.X).tolist()) == {-1.0, 1.0}
    assert np.array_equal(d.Y, load_matfile(DEFAULT_VAR_U_PATH, encoding="binary").Y)


@var_u_only
def test_var_u_rows_are_ordered_msb_first(var_u):
    # `_all_bit_patterns` claims to match this order, so `quadric` row i lines up
    # with var_u row i.
    powers = 1 << np.arange(11, -1, -1)
    assert np.array_equal(var_u.X.astype(np.int64) @ powers, np.arange(4096))


@var_u_only
def test_var_u_is_deterministic_so_the_label_carries_a_full_bit(var_u):
    # Y is a function of X, so I(X;Y) = H(Y): there is no label noise to memorise.
    assert mutual_information(_x_symbols(var_u), var_u.Y) == pytest.approx(entropy(var_u.Y))


def test_loader_rejects_malformed_files(tmp_path):
    from scipy.io import savemat

    bad = tmp_path / "bad.mat"
    savemat(bad, {"F": np.zeros((4, 12)), "nope": np.zeros(4)})
    with pytest.raises(ValueError, match="expected 'F' and 'y'"):
        load_matfile(bad)

    savemat(bad, {"F": np.full((4, 12), 5.0), "y": np.zeros(4)})
    with pytest.raises(ValueError, match="expected 0/1 inputs"):
        load_matfile(bad)

    savemat(bad, {"F": np.zeros((4, 12)), "y": np.zeros(3)})
    with pytest.raises(ValueError, match="labels"):
        load_matfile(bad)


def test_quadric_encodings_and_errors():
    assert set(np.unique(make_quadric(encoding="binary").X).tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError):
        make_quadric(encoding="nope")


def test_quadric_is_a_different_harder_problem():
    q = make_quadric()
    assert len(np.unique(q.X, axis=0)) == q.n_samples
    assert entropy(q.Y) == pytest.approx(1.0, abs=0.01)
    # Deterministic in the latent radius, and at chance for a linear probe -- the
    # opposite of var_u on both counts.
    radius, theta = q.extras["radius"], float(q.extras["threshold"])
    assert np.array_equal(q.Y, (radius >= theta).astype(np.int64))
    for r in np.unique(radius):
        assert len(np.unique(q.Y[radius == r])) == 1
    A = np.concatenate([q.X, np.ones((q.n_samples, 1), np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(A, 2.0 * q.Y - 1.0, rcond=None)
    assert ((A @ w > 0).astype(np.int64) == q.Y).mean() == pytest.approx(0.5, abs=0.05)
    assert len(np.unique(make_quadric(geometry="sphere").extras["radius"])) > len(
        np.unique(radius)
    )
    with pytest.raises(ValueError):
        make_quadric(geometry="nope")


def test_build_dataset_dispatch_and_errors():
    assert build_dataset("quadric", n_bits=8).n_samples == 256
    with pytest.raises(ValueError):
        build_dataset("matfile")
    with pytest.raises(ValueError):
        build_dataset("nope")
