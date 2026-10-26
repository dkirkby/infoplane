import json

import numpy as np
import pytest
import torch

from infoplane.datasets import make_quadric
from infoplane.export import build_payload, write_json
from infoplane.mi import entropy
from infoplane.model import PAPER_HIDDEN_WIDTHS, InfoPlaneNet
from infoplane.runner import run_ensemble
from infoplane.train import (
    TrainConfig,
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
    assert len(net.hidden) == 7
    assert net.layer_names() == [f"Layer {i}" for i in range(1, 8)]
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
@pytest.fixture(scope="module")
def short_run():
    torch.set_num_threads(1)
    dataset = make_quadric()
    cfg = TrainConfig(epochs=400, log_points=25, lr=0.4)
    return dataset, cfg, run_seed(dataset, cfg, seed=0)


def test_short_run_learns_the_rule(short_run):
    _, _, r = short_run
    assert r.train_acc[0] < 0.75  # untrained
    assert r.train_acc[-1] > 0.95
    assert r.train_loss[-1] < 0.1
    assert r.n_train == int(round(0.85 * 4096))


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
    assert payload["layers"] == [f"Layer {i}" for i in range(1, 8)]
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
    # (Not in Layer 1, which sits pinned at H(X) = 12 bits -- 12 units at 30 bins
    # resolve all 4096 patterns -- but in the narrow layers that actually compress.)
    deep = payload["aggregated"]["Layer 7"]
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
