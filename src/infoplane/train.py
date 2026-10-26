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
    """Everything that defines one run of the pipeline (recorded in the export)."""

    dataset: str = "toy"
    n_bits: int = 12
    geometry: str = "circle"
    data_path: str | None = None
    widths: tuple[int, ...] = PAPER_HIDDEN_WIDTHS
    epochs: int = 9000
    log_points: int = 256
    include_init: bool = True
    bins: int = DEFAULT_BINS
    batch_size: int = 256
    optimizer: str = "sgd"
    lr: float = 0.4
    momentum: float = 0.0
    train_frac: float = 0.85
    train_size: int | None = None
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
    extras: dict[str, Any] = field(default_factory=dict)


def _split(n: int, cfg: TrainConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed train/test split; the subset is part of the ensemble's variation."""
    n_train = cfg.train_size if cfg.train_size is not None else int(round(cfg.train_frac * n))
    n_train = int(np.clip(n_train, 1, n))
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
    n_layers = len(cfg.widths)
    n_ck = len(checkpoints)
    ixt = np.zeros((n_layers, n_ck))
    ity = np.zeros((n_layers, n_ck))
    curves = {k: np.zeros(n_ck) for k in ("train_loss", "test_loss", "train_acc", "test_acc")}
    due = {int(e): i for i, e in enumerate(checkpoints)}

    def measure(slot: int) -> None:
        for li, acts in enumerate(model.activations(Xmi)):
            a = acts.detach().cpu().numpy()
            ixt[li, slot], ity[li, slot] = layer_information(a, x_symbols, y_symbols, bins=cfg.bins)
        curves["train_loss"][slot], curves["train_acc"][slot] = _evaluate(model, Xtr, Ytr, loss_fn, binary)
        curves["test_loss"][slot], curves["test_acc"][slot] = _evaluate(model, Xte, Yte, loss_fn, binary)

    if 0 in due:
        measure(due[0])

    n_train = len(train_idx)
    batch = n_train if cfg.batch_size in (0, None) else min(cfg.batch_size, n_train)
    for epoch in range(1, cfg.epochs + 1):
        order = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch):
            sel = order[start : start + batch]
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr[sel])
            target = Ytr[sel]
            loss = loss_fn(logits.squeeze(-1), target.to(logits.dtype)) if binary else loss_fn(logits, target)
            loss.backward()
            opt.step()
        slot = due.get(epoch)
        if slot is not None:
            measure(slot)

    return SeedResult(
        seed=seed,
        epochs=checkpoints,
        layer_names=model.layer_names(),
        ixt=ixt,
        ity=ity,
        train_loss=curves["train_loss"],
        test_loss=curves["test_loss"],
        train_acc=curves["train_acc"],
        test_acc=curves["test_acc"],
        n_train=n_train,
        seconds=time.perf_counter() - started,
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
