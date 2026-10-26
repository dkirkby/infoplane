"""Mutual information between discrete variables, and the binning of activations.

Everything here works on *integer symbol arrays*: each sample is assigned one
symbol per variable and the joint distribution is the empirical histogram over
pairs of symbols, with every sample weighted equally.  That is the estimator used
in the paper -- it is a lower bound on the true MI of the continuous
representation and is only meaningful because ``X`` is enumerated (see
:mod:`infoplane.datasets`).
"""

from __future__ import annotations

import numpy as np

DEFAULT_BINS = 30
ACTIVATION_RANGE = (-1.0, 1.0)


def bin_activations(
    acts: np.ndarray, bins: int = DEFAULT_BINS, value_range: tuple[float, float] = ACTIVATION_RANGE
) -> np.ndarray:
    """Quantise ``(N, width)`` activations into ``bins`` uniform intervals per unit.

    The support is the *fixed* ``value_range`` (``[-1, 1]`` for ``tanh``), never the
    observed min/max: an adaptive range would rescale the estimate as the layer's
    dynamic range changes during training and manufacture compression that is not
    there.
    """
    lo, hi = value_range
    idx = np.floor((np.asarray(acts, dtype=np.float64) - lo) / (hi - lo) * bins)
    return np.clip(idx, 0, bins - 1).astype(np.int32, copy=False)


def joint_symbols(binned: np.ndarray) -> np.ndarray:
    """Collapse a ``(N, width)`` bin-index array to ``(N,)`` layer-state symbols.

    The layer's state is the *joint* bin tuple across its units, so two samples
    share a symbol only if every unit falls in the same bin.  ``np.unique`` is used
    rather than a mixed-radix integer encoding because ``bins**width`` overflows for
    wide layers.
    """
    binned = np.ascontiguousarray(binned)
    if binned.ndim != 2:
        raise ValueError(f"expected a 2-D bin-index array, got shape {binned.shape}")
    _, inverse = np.unique(binned, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64, copy=False)


def _relabel(a: np.ndarray) -> tuple[np.ndarray, int]:
    """Map arbitrary integer symbols onto ``0 .. k-1``."""
    a = np.asarray(a).reshape(-1)
    uniq, inverse = np.unique(a, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64, copy=False), int(uniq.size)


def entropy(a: np.ndarray) -> float:
    """Shannon entropy in bits of the empirical distribution of symbols ``a``."""
    labels, k = _relabel(a)
    counts = np.bincount(labels, minlength=k)
    p = counts[counts > 0] / labels.size
    return float(-(p * np.log2(p)).sum())


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information in bits between two discrete variables.

    ``a`` and ``b`` are paired symbol arrays of equal length; samples are weighted
    uniformly.  Returns ``sum p(a,b) log2[ p(a,b) / (p(a) p(b)) ]``.
    """
    la, ka = _relabel(a)
    lb, kb = _relabel(b)
    if la.size != lb.size:
        raise ValueError(f"length mismatch: {la.size} vs {lb.size}")
    n = la.size
    # Pair key; ka * kb stays well inside int64 for the sizes used here.
    keys, counts = np.unique(la * kb + lb, return_counts=True)
    pj = counts / n
    pa = np.bincount(la, minlength=ka) / n
    pb = np.bincount(lb, minlength=kb) / n
    return float(np.sum(pj * np.log2(pj / (pa[keys // kb] * pb[keys % kb]))))


def layer_information(
    acts: np.ndarray,
    x_symbols: np.ndarray,
    y_symbols: np.ndarray,
    bins: int = DEFAULT_BINS,
    value_range: tuple[float, float] = ACTIVATION_RANGE,
) -> tuple[float, float]:
    """``(I(X;T), I(T;Y))`` in bits for one layer's activations.

    ``T`` is the binned joint state of the layer.  Note ``I(X;T) = H(T)`` whenever
    the inputs are distinct (``T`` is then a deterministic function of ``X``), which
    is the regime this project runs in; the general estimator is used anyway so the
    identity acts as a check rather than an assumption.
    """
    t = joint_symbols(bin_activations(acts, bins=bins, value_range=value_range))
    return mutual_information(x_symbols, t), mutual_information(t, y_symbols)
