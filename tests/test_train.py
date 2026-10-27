import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from infoplane.datasets import DEFAULT_VAR_U_PATH, load_matfile, make_quadric
from infoplane.export import build_payload, write_json
from infoplane.mi import entropy
from infoplane.model import PAPER_HIDDEN_WIDTHS, InfoPlaneNet
from infoplane.runner import run_ensemble

var_u_only = pytest.mark.skipif(
    not DEFAULT_VAR_U_PATH.exists(), reason=f"{DEFAULT_VAR_U_PATH} not fetched"
)
from infoplane.train import (
    TrainConfig,
    _split,
    dpi_stats,
    dpi_violation,
    log_epoch_schedule,
    run_seed,
)


def test_log_schedule_is_sorted_unique_and_bounded():
    pts = log_epoch_schedule(9000, 256)
    assert pts[0] == 0  # initialisation checkpoint
    assert pts[1] == 1 and pts[-1] == 9000
    assert np.all(np.diff(pts) > 0)
    # --log-points is the number of checkpoints delivered, not requested.
    assert len(pts) == 257
    assert len(log_epoch_schedule(9000, 256, include_init=False)) == 256
    # Cannot ask for more distinct epochs than there are epochs.
    assert np.array_equal(log_epoch_schedule(5, 99, include_init=False), [1, 2, 3, 4, 5])
    # Log spacing: half the checkpoints land in the first few hundred epochs,
    # where fitting happens.
    assert np.median(pts) < 400
    assert log_epoch_schedule(100, 10, include_init=False)[0] == 1
    with pytest.raises(ValueError):
        log_epoch_schedule(0, 10)


def test_model_shape_and_activation_range():
    net = InfoPlaneNet(12, PAPER_HIDDEN_WIDTHS, n_out=1, seed=3)
    n = len(PAPER_HIDDEN_WIDTHS)
    assert n == 5  # the paper's 12-10-7-5-4-3-2 read as input + 5 hidden + output
    assert len(net.hidden) == n
    assert net.layer_names() == [f"Layer {i}" for i in range(1, n + 1)]
    acts = net.activations(torch.rand(64, 12) * 2 - 1)
    assert [a.shape[1] for a in acts] == list(PAPER_HIDDEN_WIDTHS)
    assert all(a.abs().max() <= 1.0 for a in acts)  # the binning support
    assert net(torch.rand(5, 12)).shape == (5, 1)


def test_construction_is_seeded_and_leaves_global_rng_untouched():
    torch.manual_seed(1234)
    before = torch.rand(3)
    torch.manual_seed(1234)
    InfoPlaneNet(12, seed=7)
    assert torch.equal(torch.rand(3), before)

    a = InfoPlaneNet(12, seed=7).hidden[0].weight
    b = InfoPlaneNet(12, seed=7).hidden[0].weight
    c = InfoPlaneNet(12, seed=8).hidden[0].weight
    assert torch.equal(a, b) and not torch.equal(a, c)


# The training tests use `quadric` rather than `var_u`: it needs no download, and
# being deterministic and noiseless it converges predictably in a few hundred epochs.
# They also override the optimiser: the defaults are the authors' (Adam at 4e-4),
# chosen to reproduce the paper's *epoch axis*, which deliberately takes thousands
# of epochs.  These tests only need a converged network, so they use the fast SGD
# setting instead of paying for that.
@pytest.fixture(scope="module")
def short_run():
    torch.set_num_threads(1)
    dataset = make_quadric()
    # split="random": these tests are about the training mechanics, and the
    # authors' head slice takes the first 80% of the bit patterns, which for
    # `quadric` is biased against the large-radius class and converges slower.
    cfg = TrainConfig(epochs=400, log_points=25, optimizer="sgd", lr=0.4, split="random")
    return dataset, cfg, run_seed(dataset, cfg, seed=0)


def test_short_run_learns_the_rule(short_run):
    _, _, r = short_run
    assert r.train_acc[0] < 0.75  # untrained
    assert r.train_acc[-1] > 0.95
    assert r.train_loss[-1] < 0.1
    assert r.n_train == int(round(TrainConfig.train_frac * 4096))


def test_information_is_within_its_bounds(short_run):
    dataset, _, r = short_run
    h_x = entropy(np.unique(dataset.X, axis=0, return_inverse=True)[1].reshape(-1))
    h_y = entropy(dataset.Y)
    assert r.ixt.min() >= 0.0 and r.ixt.max() <= h_x + 1e-9
    assert r.ity.min() >= 0.0 and r.ity.max() <= h_y + 1e-9


def test_data_processing_inequality_holds_at_every_checkpoint(short_run):
    _, _, r = short_run
    assert dpi_violation(r.ixt) <= 1e-9
    # I(T;Y) is ordered the same way in principle, but only loosely under the
    # binning estimator: binning layer i+1 is not a coarsening of binning layer i,
    # so a deeper layer can resolve label structure that a wider, blurrier layer
    # lost.  Expect agreement to a few thousandths of a bit, not exactness.
    assert float(np.max(np.diff(r.ity, axis=0))) <= 0.01


def test_label_information_increases_during_fitting(short_run):
    _, _, r = short_run
    # The deepest layer starts knowing nothing about Y and ends up at ~H(Y).
    assert r.ity[-1, 0] < 0.1 < r.ity[-1, -1]


def test_dpi_stats_counts_as_well_as_measures():
    # Layers down the rows, checkpoints across the columns; strictly decreasing
    # down a column is what the inequality requires.
    ok = np.array([[3.0, 2.0], [2.0, 1.0], [1.0, 0.5]])
    assert dpi_stats(ok) == (-0.5, 0, 4)  # worst gap is the smallest decrease

    bad = ok.copy()
    bad[2, 1] = 1.25  # one layer carries more than the one above it
    worst, n_viol, n_total = dpi_stats(bad)
    assert worst == pytest.approx(0.25) and (n_viol, n_total) == (1, 4)
    assert dpi_violation(bad) == pytest.approx(0.25)

    assert dpi_stats(np.zeros((1, 5))) == (0.0, 0, 0)  # a single layer cannot violate


def test_train_size_overrides_the_fraction():
    dataset = make_quadric()
    cfg = TrainConfig(epochs=2, log_points=3, train_size=100)
    assert run_seed(dataset, cfg, seed=0).n_train == 100


def test_ensemble_export_round_trips(tmp_path):
    dataset = make_quadric()
    cfg = TrainConfig(epochs=3, log_points=4)
    results = run_ensemble(dataset, cfg, [0, 1], num_workers=1, progress=False)
    payload = build_payload(dataset, cfg, results)

    assert payload["bounds"]["H_X"] == pytest.approx(12.0)
    # The readout is measured as the paper's highest layer (see model.activations).
    assert payload["layers"] == [f"Layer {i}" for i in range(1, len(cfg.widths) + 1)] + ["Output"]
    assert [run["seed"] for run in payload["runs"]] == [0, 1]
    n_ck = payload["metadata"]["checkpoints"]
    assert len(payload["epochs"]) == n_ck
    for run in payload["runs"]:
        for name in payload["layers"]:
            assert len(run["trajectories"][name]["IXT"]) == n_ck
            assert len(run["trajectories"][name]["ITY"]) == n_ck
    agg = payload["aggregated"]["Layer 1"]
    assert len(agg["IXT_mean"]) == n_ck and len(agg["IXT_std"]) == n_ck
    # Seeds that differ must leave a non-zero spread somewhere for Figure 4's band.
    # (Not in Layer 1, which sits pinned at H(X) = 12 bits -- 10 units at 30 bins
    # resolve all 4096 patterns -- but in the narrow layers that actually compress.)
    deep = payload["aggregated"][payload["layers"][-1]]
    assert max(v for v in deep["IXT_std"] if v is not None) > 0

    out = tmp_path / "e.json"
    size = write_json(payload, out)
    assert size > 0
    assert json.loads(out.read_text())["metadata"]["seeds"] == 2


def test_export_rejects_inconsistent_results():
    dataset = make_quadric()
    a = run_seed(dataset, TrainConfig(epochs=3, log_points=4), seed=0)
    b = run_seed(dataset, TrainConfig(epochs=5, log_points=4), seed=1)
    with pytest.raises(ValueError):
        build_payload(dataset, TrainConfig(), [a, b])
    with pytest.raises(ValueError):
        build_payload(dataset, TrainConfig(), [])


# --- the three places where this pipeline follows IDNNs rather than convention ---
# Each of these was a silent difference from the authors' code until 2026-10-26;
# they are pinned here because every one of them is a tempting thing to "clean up".


def test_default_split_is_the_authors_head_slice_not_a_random_subset():
    """IDNNs' `data_shuffle` defaults to `shuffle_data=False` and slices the front.

    Consequences worth knowing before changing this: the slice is identical for
    every seed (so the ensemble varies only in its weights, though the paper
    claims "different random selections of the training samples"), and because
    `var_u.mat` is ordered by input pattern it is not class-balanced.
    """
    cfg = TrainConfig()
    assert (cfg.split, cfg.shuffle_batches, cfg.measure_output) == ("head", False, True)

    tr_a, te_a = _split(4096, cfg, seed=0)
    tr_b, _ = _split(4096, cfg, seed=17)
    assert np.array_equal(tr_a, np.arange(3277)) and np.array_equal(te_a, np.arange(3277, 4096))
    assert np.array_equal(tr_a, tr_b)  # the seed does not move it

    rnd_a, _ = _split(4096, replace(cfg, split="random"), seed=0)
    rnd_b, _ = _split(4096, replace(cfg, split="random"), seed=17)
    assert not np.array_equal(np.sort(rnd_a), np.sort(rnd_b))
    with pytest.raises(ValueError):
        _split(4096, replace(cfg, split="nope"), seed=0)


@var_u_only
def test_the_head_slice_is_class_imbalanced():
    # Not a defect in our code -- a property of the authors' split, and the reason
    # test accuracy here is not comparable to a random-split number.
    d = load_matfile(DEFAULT_VAR_U_PATH)
    tr, te = _split(d.n_samples, TrainConfig(), seed=0)
    assert d.Y[tr].mean() == pytest.approx(0.4373, abs=1e-3)
    assert d.Y[te].mean() == pytest.approx(0.8364, abs=1e-3)


def test_the_readout_is_measured_as_the_paper_s_highest_layer():
    """§4.1: "the highest is the output layer"; IDNNs measures it as hidden_layers[-1]."""
    dataset = make_quadric()
    r = run_seed(dataset, TrainConfig(epochs=2, log_points=3), seed=0)
    assert r.layer_names == ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Output"]
    assert r.ixt.shape[0] == len(PAPER_HIDDEN_WIDTHS) + 1
    # It is the most compressed point of the chain, and still obeys the DPI.
    assert r.ixt[-1].max() < r.ixt[-2].min()
    assert dpi_violation(r.ixt) <= 1e-9

    bare = run_seed(dataset, TrainConfig(epochs=2, log_points=3, measure_output=False), seed=0)
    assert bare.layer_names == r.layer_names[:-1]
    assert np.allclose(bare.ixt, r.ixt[:-1])  # measuring it changes nothing else


def test_output_probabilities_are_the_softmax_pair_whatever_the_readout():
    # The authors bin both columns of a 2-way softmax.  A single sigmoid unit is
    # the same model, so we bin (p, 1-p) to keep the measurement identical.
    net = InfoPlaneNet(12, seed=0, n_out=1)
    p = net.output_probs(torch.rand(32, PAPER_HIDDEN_WIDTHS[-1]) * 2 - 1)
    assert p.shape == (32, 2)
    assert torch.allclose(p.sum(dim=1), torch.ones(32), atol=1e-6)


# --- the gradient statistics behind Figure 4 ---


def test_gradient_stats_are_recorded_per_weight_matrix():
    """One curve per connection, which is one more than the number of hidden layers.

    The paper's Figure 4 plots (input,L1), (L1,L2) ... (L5,output); the authors
    take the even entries of `tf.trainable_variables()`, which is the same set
    because weights are appended before biases for each layer.
    """
    dataset = make_quadric()
    cfg = TrainConfig(epochs=6, log_points=4)
    r = run_seed(dataset, cfg, seed=0)

    assert r.weight_names == ["input->L1", "L1->L2", "L2->L3", "L3->L4", "L4->L5", "L5->Output"]
    assert len(r.weight_names) == len(cfg.widths) + 1
    assert r.grad_mean.shape == r.grad_std.shape == (len(r.weight_names), len(r.epochs))

    assert np.all(np.isfinite(r.grad_mean)) and np.all(r.grad_mean > 0)
    assert np.all(np.isfinite(r.grad_std)) and np.all(r.grad_std > 0)


def test_the_init_checkpoint_is_measured_before_any_step():
    """Figure 4's left edge is the untrained network, and it has to stay untrained.

    Taking it after the first epoch instead loses the largest part of the rise the
    figure is about: Adam steps by ~lr whatever the gradient is, so by the end of
    epoch 1 the weights have left the symmetric initialisation and the
    batch-to-batch spread is already an order of magnitude up.
    """
    dataset = make_quadric()
    cfg = TrainConfig(epochs=4, log_points=3)
    r = run_seed(dataset, cfg, seed=0)

    fresh = InfoPlaneNet(dataset.n_features, cfg.widths, n_out=1, seed=0)
    init_norms = np.array([float(w.detach().norm()) for w in fresh.weight_matrices()])
    assert np.allclose(r.weight_norm[:, 0], init_norms, rtol=1e-6)
    assert np.all(np.isfinite(r.grad_std[:, 0])) and np.all(r.grad_std[:, 0] > 0)
    # The next checkpoint is after training, so its norms must have moved.
    assert not np.allclose(r.weight_norm[:, 1], init_norms, rtol=1e-6)


def test_gradient_spread_is_zero_without_minibatches():
    # The std is the batch-to-batch spread, so one batch per epoch must give
    # exactly zero -- if it does not, the quantity is measuring something else.
    dataset = make_quadric()
    r = run_seed(dataset, TrainConfig(epochs=3, log_points=3, batch_size=0), seed=0)
    assert np.allclose(r.grad_std[:, 1:], 0.0)
    assert np.all(r.grad_mean[:, 1:] > 0)


def test_weight_norms_are_the_frobenius_norms_at_the_checkpoint():
    # Figure 4's caption normalises by these, so they have to be the norms of the
    # weights as they stand at the checkpoint, not some running average.
    dataset = make_quadric()
    cfg = TrainConfig(epochs=4, log_points=3)
    torch.manual_seed(0)
    r = run_seed(dataset, cfg, seed=0)
    net = InfoPlaneNet(dataset.n_features, cfg.widths, n_out=1, seed=0)
    # At the first checkpoint the network has trained, so the norms must have
    # moved off their initial values but stay the same order of magnitude.
    init = np.array([float(w.detach().norm()) for w in net.weight_matrices()])
    final = r.weight_norm[:, -1]
    assert np.all(np.isfinite(final)) and np.all(final > 0)
    assert not np.allclose(final, init)
    assert np.all(final < 5 * init)


def test_grad_stats_can_be_switched_off():
    dataset = make_quadric()
    r = run_seed(dataset, TrainConfig(epochs=3, log_points=3, grad_stats=False), seed=0)
    assert np.all(np.isnan(r.grad_mean)) and np.all(np.isnan(r.grad_std))


def test_the_normalised_gradient_curves_divide_per_seed_not_per_aggregate():
    """Figure 4's caption divides by ||W||; the division has to happen per seed.

    ||g|| and ||W|| are correlated across initialisations, so a ratio of medians
    is not the median of the ratios -- and the frontend only ever sees the
    aggregate, which is why both forms are exported rather than one plus the
    weight norms to divide by.
    """
    dataset = make_quadric()
    cfg = TrainConfig(epochs=4, log_points=3)
    results = [run_seed(dataset, cfg, seed=s) for s in range(3)]
    payload = build_payload(dataset, cfg, results)

    g = np.stack([r.grad_mean for r in results])
    w = np.stack([r.weight_norm for r in results])
    for wi, name in enumerate(results[0].weight_names):
        got = np.asarray(payload["gradients"][name]["mean_norm"])
        want = np.nanmedian(g[:, wi] / w[:, wi], axis=0)
        assert got == pytest.approx(want, rel=1e-5)
        # the raw norms stay alongside, undivided
        assert payload["gradients"][name]["mean"] == pytest.approx(
            np.nanmedian(g[:, wi], axis=0), rel=1e-5
        )
