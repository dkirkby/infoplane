"""Training a single seed and measuring its information-plane trajectory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch
from torch import nn

from .datasets import Dataset
from .mi import DEFAULT_BINS, layer_information
from .model import PAPER_HIDDEN_WIDTHS, InfoPlaneNet


def log_epoch_schedule(total_epochs: int, n_points: int, include_init: bool = True) -> np.ndarray:
    """``n_points`` distinct log-spaced epoch checkpoints in ``1 .. total_epochs``.

    Log spacing is essential rather than cosmetic: the fitting phase is over within
    the first few tens of epochs while compression takes thousands, so uniform
    sampling would spend its whole budget on the part where nothing moves.

    Two refinements over the ``np.unique(np.logspace(...).astype(int))`` recipe in
    ``PLAN.md``, which does not actually deliver the stated number of checkpoints:

    * ``astype(int)`` truncates, so the final point comes out as 8999 rather than
      9000 and every checkpoint is biased low; we round instead.
    * de-duplicating 256 log-spaced reals over 1..9000 leaves only 189 distinct
      epochs, because the early ones collide.  ``n_points`` is treated as the
      number of checkpoints actually wanted, so the request is grown until that
      many survive and then thinned back -- evenly in *index* space, which keeps
      the spacing logarithmic -- to exactly ``n_points`` (capped at
      ``total_epochs``, beyond which distinct epochs do not exist).

    ``include_init`` prepends epoch 0 (the network at initialisation), where every
    trajectory in Figure 2 starts.
    """
    if total_epochs < 1 or n_points < 1:
        raise ValueError("total_epochs and n_points must be >= 1")
    target = min(n_points, total_epochs)
    request = target
    for _ in range(64):
        pts = np.unique(np.round(np.logspace(0, np.log10(total_epochs), request)).astype(np.int64))
        if len(pts) >= target:
            break
        request = int(request * 1.3) + 1
    if len(pts) > target:
        keep = np.unique(np.round(np.linspace(0, len(pts) - 1, target)).astype(np.int64))
        pts = pts[keep]
    if include_init:
        pts = np.concatenate([[0], pts])
    return pts


@dataclass
class TrainConfig:
    """Everything that defines one run of the pipeline (recorded in the export).

    The optimiser defaults are the authors' own, taken from IDNNs
    (``idnns/networks/network_paramters.py`` and ``model.py``): Adam at 4e-4 over
    batches of 512 with an 80/20 split.  They are not interchangeable with any
    other fast-converging choice, because the *epoch axis of Figures 2-5 is a
    hyperparameter*.  Under SGD at lr 0.4 this network fits in ~2 epochs, so the
    whole fitting phase falls between the first two log-spaced checkpoints and
    the figures show only compression; under these settings it takes tens of
    epochs.  Note the paper's text says SGD while the released code uses Adam --
    the code is what produced the figures, so that is what we follow, and the
    mismatch that remains is in README: our ensemble turns at epoch ~43 where the
    paper's Figure 4 marks the transition at ~350.

    ``split``, ``shuffle_batches`` and ``measure_output`` likewise default to what
    IDNNs does rather than to the conventional choice.  See :func:`_split`,
    :func:`run_seed` and :meth:`InfoPlaneNet.activations` for the lines of their
    code each one comes from and what it changes.
    """

    dataset: str = "var_u"
    n_bits: int = 12
    geometry: str = "circle"
    encoding: str = "binary"
    data_path: str | None = None
    widths: tuple[int, ...] = PAPER_HIDDEN_WIDTHS
    epochs: int = 9000
    log_points: int = 256
    include_init: bool = True
    bins: int = DEFAULT_BINS
    batch_size: int = 512
    optimizer: str = "adam"
    lr: float = 4e-4
    momentum: float = 0.0
    train_frac: float = 0.8
    train_size: int | None = None
    split: str = "head"
    shuffle_batches: bool = False
    measure_output: bool = True
    grad_stats: bool = True
    mi_on: str = "all"
    device: str = "cpu"

    def as_metadata(self) -> dict[str, Any]:
        d = asdict(self)
        d["widths"] = list(self.widths)
        return d


@dataclass
class SeedResult:
    seed: int
    epochs: np.ndarray
    layer_names: list[str]
    ixt: np.ndarray  # (n_layers, n_checkpoints)
    ity: np.ndarray  # (n_layers, n_checkpoints)
    train_loss: np.ndarray
    test_loss: np.ndarray
    train_acc: np.ndarray
    test_acc: np.ndarray
    n_train: int
    seconds: float
    weight_names: list[str] = field(default_factory=list)
    grad_mean: np.ndarray | None = None   # (n_weight_matrices, n_checkpoints)
    grad_std: np.ndarray | None = None
    step_mean: np.ndarray | None = None   # the same two, reduced over steps
    step_std: np.ndarray | None = None
    weight_norm: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _split(n: int, cfg: TrainConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split.

    ``split="head"`` is the authors': ``data_shuffle`` takes the *first* ``train_frac``
    of the file and defaults ``shuffle_data=False``, so every one of their 50
    "randomized" networks trains on the same deterministic slice and only the
    weights differ.  That slice is not a random sample -- ``var_u.mat`` is ordered
    by input pattern read as an integer, so the first 80% is every pattern below
    3277 (p(y=1) = 0.44) and the held-out 20% is the high end (p(y=1) = 0.84).
    This contradicts the paper's §3.2, which claims "different random selections of
    the training samples", and it is the single largest behavioural difference
    between the paper's description and the code that produced its figures.

    ``split="random"`` is a per-seed random subset, which is what the paper
    describes and what the training-size sweep has to use: the first 50 patterns
    of the file are all label 0, so a small head slice is single-class.
    """
    n_train = cfg.train_size if cfg.train_size is not None else int(round(cfg.train_frac * n))
    n_train = int(np.clip(n_train, 1, n))
    if cfg.split == "head":
        return np.arange(n_train), np.arange(n_train, n)
    if cfg.split != "random":
        raise ValueError(f"unknown split {cfg.split!r}")
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:n_train], perm[n_train:]


def _make_loss(n_classes: int) -> tuple[nn.Module, int]:
    """Loss and readout width: a single sigmoid unit for the paper's binary task."""
    if n_classes == 2:
        return nn.BCEWithLogitsLoss(), 1
    return nn.CrossEntropyLoss(), n_classes


@torch.no_grad()
def _evaluate(
    model: InfoPlaneNet, X: torch.Tensor, Y: torch.Tensor, loss_fn: nn.Module, binary: bool
) -> tuple[float, float]:
    if X.shape[0] == 0:
        return float("nan"), float("nan")
    logits = model(X)
    if binary:
        loss = loss_fn(logits.squeeze(-1), Y.to(logits.dtype))
        pred = (logits.squeeze(-1) > 0).long()
    else:
        loss = loss_fn(logits, Y)
        pred = logits.argmax(dim=1)
    return float(loss), float((pred == Y).double().mean())


def run_seed(dataset: Dataset, cfg: TrainConfig, seed: int) -> SeedResult:
    """Train one network and record its information plane at every checkpoint."""
    started = time.perf_counter()
    device = torch.device(cfg.device)
    torch.manual_seed(seed)  # governs minibatch shuffling; weights use fork_rng

    X = torch.from_numpy(dataset.X).to(device)
    Y = torch.from_numpy(dataset.Y).to(device)
    train_idx, test_idx = _split(dataset.n_samples, cfg, seed)
    tr = torch.from_numpy(train_idx).to(device)
    te = torch.from_numpy(test_idx).to(device)
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    loss_fn, n_out = _make_loss(dataset.n_classes)
    binary = dataset.n_classes == 2
    model = InfoPlaneNet(dataset.n_features, cfg.widths, n_out=n_out, seed=seed).to(device)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    # The set the information plane is measured on.  "all" keeps I(X;T) bounded by
    # the full H(X) = log2(N) and is what makes trajectories comparable across the
    # training-set sizes swept for Figure 5.
    if cfg.mi_on == "all":
        mi_idx = np.arange(dataset.n_samples)
    elif cfg.mi_on == "train":
        mi_idx = train_idx
    elif cfg.mi_on == "test":
        mi_idx = test_idx
    else:
        raise ValueError(f"unknown mi_on {cfg.mi_on!r}")
    Xmi = X[torch.from_numpy(mi_idx).to(device)]
    # X symbols: one per distinct input pattern, uniformly weighted.
    _, x_symbols = np.unique(dataset.X[mi_idx], axis=0, return_inverse=True)
    x_symbols = x_symbols.reshape(-1)
    y_symbols = dataset.Y[mi_idx]

    checkpoints = log_epoch_schedule(cfg.epochs, cfg.log_points, cfg.include_init)
    n_layers = len(cfg.widths) + int(cfg.measure_output)
    n_ck = len(checkpoints)
    ixt = np.zeros((n_layers, n_ck))
    ity = np.zeros((n_layers, n_ck))

    # Figure 4: the minibatch-gradient statistics, per weight matrix.  The
    # authors collect the gradient of the loss on every minibatch of a checkpoint
    # epoch, at the weights that minibatch was evaluated at, and reduce them to
    # two numbers per layer (IDNNs `plots/plot_gradients.py`):
    #
    #   mean = || E_b[g_b] ||           the norm of the mean gradient
    #   std  = sqrt( E_b || g_b - E_b[g_b] ||^2 )
    #
    # The second is their `sqrt(trace(mean_cov_mat))`, written without building
    # the covariance matrix: the trace of the mean outer product *is* the mean
    # squared deviation, and the matrix is (n_weights x n_weights), which for the
    # first layer would be 120x120 per checkpoint for no gain.
    #
    # The same two numbers are also reduced over a second collection -- the
    # gradient of the WHOLE training set, re-evaluated at each of the epoch's
    # weight states.  It holds the data fixed and lets only the weights move, so
    # it contains no minibatch noise at all; it is what the published figure's
    # dotted curve looks like, which the minibatch spread does not.  See the
    # note on `fig4.html`.
    weights = model.weight_matrices()
    n_wl = len(weights)
    grad_mean = np.full((n_wl, n_ck), np.nan)
    grad_std = np.full((n_wl, n_ck), np.nan)
    step_mean = np.full((n_wl, n_ck), np.nan)
    step_std = np.full((n_wl, n_ck), np.nan)
    weight_norm = np.full((n_wl, n_ck), np.nan)
    curves = {k: np.zeros(n_ck) for k in ("train_loss", "test_loss", "train_acc", "test_acc")}
    due = {int(e): i for i, e in enumerate(checkpoints)}

    def measure(slot: int) -> None:
        for li, acts in enumerate(model.activations(Xmi, include_output=cfg.measure_output)):
            a = acts.detach().cpu().numpy()
            ixt[li, slot], ity[li, slot] = layer_information(a, x_symbols, y_symbols, bins=cfg.bins)
        curves["train_loss"][slot], curves["train_acc"][slot] = _evaluate(model, Xtr, Ytr, loss_fn, binary)
        curves["test_loss"][slot], curves["test_acc"][slot] = _evaluate(model, Xte, Yte, loss_fn, binary)

    n_train = len(train_idx)
    batch = n_train if cfg.batch_size in (0, None) else min(cfg.batch_size, n_train)
    # The authors' minibatches are fixed contiguous slices, cut once and reused
    # every epoch (IDNNs `network.train_network` builds `batch_points` outside the
    # epoch loop and never reshuffles).  Reshuffling each epoch is the textbook
    # choice but it is a different noise process, and the compression phase is
    # supposed to be driven by exactly that noise.
    fixed_order = torch.arange(n_train, device=device)

    def batch_loss(sel: torch.Tensor) -> torch.Tensor:
        logits = model(Xtr[sel])
        target = Ytr[sel]
        return loss_fn(logits.squeeze(-1), target.to(logits.dtype)) if binary else loss_fn(logits, target)

    def reduce_grads(
        per_layer: list[list[torch.Tensor]],
        slot: int,
        mean_out: np.ndarray,
        std_out: np.ndarray,
    ) -> None:
        for i, grads in enumerate(per_layer):
            g = torch.stack(grads)                      # (n_collected, n_weights)
            mean_vec = g.mean(dim=0)
            mean_out[i, slot] = float(mean_vec.norm())
            std_out[i, slot] = float((g - mean_vec).pow(2).sum(dim=1).mean().sqrt())

    def full_grads() -> list[torch.Tensor]:
        """The gradient of the whole training set, leaving ``.grad`` as it was.

        The optimiser is mid-epoch and about to step on the gradients currently
        in ``.grad`` -- biases included -- so they are put back afterwards.
        """
        saved = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
        opt.zero_grad(set_to_none=True)
        batch_loss(fixed_order).backward()
        out = [w.grad.detach().reshape(-1).clone() for w in weights]
        for p, g in zip(model.parameters(), saved):
            p.grad = g
        return out

    if 0 in due:
        measure(due[0])
        if cfg.grad_stats:
            # The gradient statistics at the *untrained* network, which is where
            # the paper's Figure 4 starts and where its two curves are furthest
            # apart.  They have to be taken without stepping: by the end of the
            # first epoch Adam has already moved the weights off the symmetric
            # initialisation -- seven steps of ~lr each, regardless of how small
            # the gradient is -- and the batch-to-batch spread is an order of
            # magnitude larger than it was at epoch 0.  Measuring from epoch 1
            # hides the first and largest part of the rise the figure is about.
            # Partitioned the same way the first epoch will be, or the epoch-0
            # point is measuring a different noise process from every point after
            # it: contiguous batches of this file are sorted by input pattern and
            # differ systematically in label balance, shuffled ones do not.
            init_order = (
                torch.randperm(n_train, device=device) if cfg.shuffle_batches else fixed_order
            )
            init_grads: list[list[torch.Tensor]] = [[] for _ in weights]
            for start in range(0, n_train, batch):
                sel = init_order[start : start + batch]
                opt.zero_grad(set_to_none=True)
                batch_loss(sel).backward()
                for i, w in enumerate(weights):
                    init_grads[i].append(w.grad.detach().reshape(-1).clone())
            opt.zero_grad(set_to_none=True)
            reduce_grads(init_grads, due[0], grad_mean, grad_std)
            for i, w in enumerate(weights):
                weight_norm[i, due[0]] = float(w.detach().norm())
            # `step_mean`/`step_std` stay NaN here: no step has been taken, so
            # there is only one weight state and no spread to measure.  The
            # paper's own Figure 4 starts at epoch 1 for the same reason.
    for epoch in range(1, cfg.epochs + 1):
        order = torch.randperm(n_train, device=device) if cfg.shuffle_batches else fixed_order
        # Only on checkpoint epochs: keeping every batch's gradient for all 9000
        # epochs would cost more memory than the whole run.
        collecting = cfg.grad_stats and epoch in due
        batch_grads: list[list[torch.Tensor]] | None = [[] for _ in weights] if collecting else None
        step_grads: list[list[torch.Tensor]] | None = [[] for _ in weights] if collecting else None
        for start in range(0, n_train, batch):
            sel = order[start : start + batch]
            opt.zero_grad(set_to_none=True)
            batch_loss(sel).backward()
            if batch_grads is not None:
                # Read the gradient the optimiser is about to use, before it steps:
                # this is the gradient at the weights that minibatch saw, which is
                # what makes the batch-to-batch spread the SGD noise of §5.2.
                for i, w in enumerate(weights):
                    batch_grads[i].append(w.grad.detach().reshape(-1).clone())
            if step_grads is not None:
                # The same weights, the whole training set: this one moves only
                # because the optimiser has stepped.
                for i, g in enumerate(full_grads()):
                    step_grads[i].append(g)
            opt.step()
        slot = due.get(epoch)
        if slot is not None:
            if batch_grads is not None:
                reduce_grads(batch_grads, slot, grad_mean, grad_std)
                reduce_grads(step_grads, slot, step_mean, step_std)
                for i, w in enumerate(weights):
                    weight_norm[i, slot] = float(w.detach().norm())
            measure(slot)

    return SeedResult(
        seed=seed,
        epochs=checkpoints,
        layer_names=model.layer_names(include_output=cfg.measure_output),
        ixt=ixt,
        ity=ity,
        train_loss=curves["train_loss"],
        test_loss=curves["test_loss"],
        train_acc=curves["train_acc"],
        test_acc=curves["test_acc"],
        n_train=n_train,
        seconds=time.perf_counter() - started,
        weight_names=model.weight_names(),
        grad_mean=grad_mean,
        grad_std=grad_std,
        step_mean=step_mean,
        step_std=step_std,
        weight_norm=weight_norm,
    )


def dpi_stats(ixt: np.ndarray, tol: float = 1e-9) -> tuple[float, int, int]:
    """``(largest violation in bits, number of violations, measurements checked)``.

    The count matters as much as the maximum: the plug-in estimator produces rare
    hairline inversions (see :func:`dpi_violation`), so a handful of small
    violations out of tens of thousands of measurements is an estimator artifact,
    whereas a large *fraction* of them means the measurement is broken.
    """
    if ixt.shape[0] < 2:
        return 0.0, 0, 0
    diffs = np.diff(ixt, axis=0)
    return float(diffs.max()), int((diffs > tol).sum()), int(diffs.size)


def dpi_violation(ixt: np.ndarray) -> float:
    """Largest violation of the data processing inequality, in bits.

    ``I(X;T_1) >= I(X;T_2) >= ... >= I(X;T_k)`` must hold at every checkpoint: each
    layer is a function of the one below it, so it cannot carry more information
    about the input.  Anything above a few thousandths of a bit means the
    measurement is wrong -- activations captured from the wrong layer, or a binning
    range that does not match the activation support -- not a discovery.  (Strictly,
    the *binned* estimator is not itself a Markov chain, since binning layer i+1 is
    not a coarsening of binning layer i, so hairline violations are possible even
    when the underlying quantities are ordered.  In practice this estimate comes out
    exactly monotone for the paper's architecture.)
    """
    return float(np.max(np.diff(ixt, axis=0))) if ixt.shape[0] > 1 else 0.0
