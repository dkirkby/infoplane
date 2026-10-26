"""Dataset generators for the information-plane experiments.

The quantity that makes this whole replication tractable is that ``X`` is a *small
discrete* variable with a known distribution: every input pattern is enumerated
exactly once, so ``p(x) = 1/N`` and ``H(X) = log2(N)`` exactly.  Any dataset added
here must preserve that property (distinct rows, uniform weight) or the binning
estimator in :mod:`infoplane.mi` loses its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_TOY_BITS = 12


@dataclass(frozen=True)
class Dataset:
    """An enumerated classification problem.

    Attributes
    ----------
    X:
        ``(N, d)`` float32 inputs, already scaled to the network's input range.
    Y:
        ``(N,)`` int64 class labels.
    name:
        Identifier recorded in the export metadata.
    n_classes:
        Number of distinct labels.
    extras:
        Diagnostic arrays (e.g. the 2D latent coordinates of the toy rule) that are
        useful for plotting but are not fed to the network.
    """

    X: np.ndarray
    Y: np.ndarray
    name: str
    n_classes: int
    extras: dict[str, np.ndarray]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _circle_directions(n: int) -> np.ndarray:
    """``n`` uniformly spaced unit vectors on the circle (a "2D sphere")."""
    angles = 2.0 * np.pi * np.arange(n) / n
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniformly spaced unit vectors on the 2-sphere in R^3."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    phi = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _all_bit_patterns(n_bits: int) -> np.ndarray:
    """All ``2**n_bits`` patterns as a ``(2**n_bits, n_bits)`` array of 0/1."""
    codes = np.arange(2**n_bits, dtype=np.int64)
    shifts = np.arange(n_bits, dtype=np.int64)
    return ((codes[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def _balanced_threshold(f: np.ndarray) -> float:
    """Threshold on ``f`` whose induced labelling is closest to balanced.

    ``f`` is massively degenerate for the toy rule, so a median split would cut
    through a tie group and hand identical values opposite labels -- i.e. ``Y``
    would no longer be a function of ``f`` and the decision boundary would be
    unlearnable at that radius.  Instead we only ever cut *between* distinct
    values of ``f``, accepting ``p(y=1)`` slightly off 0.5.
    """
    values, counts = np.unique(f, return_counts=True)
    # Fraction labelled 1 if we cut just below values[k], i.e. y = f >= values[k].
    above = counts.sum() - np.concatenate([[0], np.cumsum(counts)[:-1]])
    frac = above / counts.sum()
    k = int(np.argmin(np.abs(frac - 0.5)))
    return float(values[k])


def make_toy(n_bits: int = N_TOY_BITS, geometry: str = "circle") -> Dataset:
    """The 12-bit binary problem of Shwartz-Ziv & Tishby (2017), section 3.

    The paper specifies 12 binary inputs standing for 12 uniformly distributed
    points on a sphere, labelled by thresholding a *spherically symmetric* real
    function of the pattern.  The exact function is not given in the text, so this
    is a faithful reconstruction rather than a byte-for-byte copy of the authors'
    ``var_u_data.mat`` (use :func:`load_matfile` for that):

    * bit ``i`` is mapped to a spin ``s_i = 2 x_i - 1`` in ``{-1, +1}``;
    * the spins are projected onto the ``n_bits`` unit directions ``u_i``, giving
      ``z(x) = sum_i s_i u_i`` -- a point in the plane for ``geometry="circle"``;
    * the rule is the rotation-invariant ``f(x) = ||z(x)||``, thresholded so that
      ``p(y=1)`` is as close to 0.5 as the (degenerate) values of ``f`` allow.

    Centring the bits to spins removes the linear term from ``||z||^2``, so the
    boundary is a genuine quadric -- a radius in the latent plane -- and is not
    linearly separable in ``x``.  The ``circle`` geometry makes that latent space
    two-dimensional, which is what keeps the problem learnable by the paper's very
    narrow network; ``sphere`` gives the rank-3 variant.

    Inputs are returned as ``+-1`` spins, matching the ``[-1, 1]`` range of the
    ``tanh`` hidden units.
    """
    if geometry == "circle":
        directions = _circle_directions(n_bits)
    elif geometry == "sphere":
        directions = _fibonacci_sphere(n_bits)
    else:  # pragma: no cover - guarded by the CLI choices
        raise ValueError(f"unknown toy geometry {geometry!r}")

    bits = _all_bit_patterns(n_bits)
    spins = (2 * bits.astype(np.float64) - 1.0)
    z = spins @ directions
    f = np.linalg.norm(z, axis=1)
    # Round away float noise so that genuinely equal radii land in one tie group.
    f = np.round(f, 9)

    theta = _balanced_threshold(f)
    Y = (f >= theta).astype(np.int64)

    return Dataset(
        X=spins.astype(np.float32),
        Y=Y,
        name=f"toy{n_bits}-{geometry}",
        n_classes=2,
        extras={"latent": z.astype(np.float32), "radius": f, "threshold": np.asarray(theta)},
    )


def load_matfile(path: str | Path) -> Dataset:
    """Load the original ``var_u_data.mat`` from the authors' IDNNs repository.

    Expects the MATLAB struct used by that code: an ``F`` field of 4096x12 inputs
    and a ``y`` field of labels.  Provided so results can be cross-checked against
    the published data when the file is available locally.
    """
    from scipy.io import loadmat

    raw = loadmat(str(path))
    if "F" not in raw or "y" not in raw:
        raise ValueError(f"{path} has keys {sorted(raw)}; expected 'F' and 'y'")
    X = np.asarray(raw["F"], dtype=np.float32)
    y = np.asarray(raw["y"]).squeeze()
    if y.ndim == 2:  # one-hot / 2xN layouts used by some revisions of that file
        y = y.argmax(axis=0) if y.shape[0] < y.shape[1] else y.argmax(axis=1)
    Y = np.asarray(y, dtype=np.int64).reshape(-1)
    # The published inputs are 0/1; centre them to match make_toy's spin range.
    if X.min() >= 0.0:
        X = 2.0 * X - 1.0
    return Dataset(X=X, Y=Y, name=Path(path).stem, n_classes=int(Y.max()) + 1, extras={})


def make_mnist(root: str | Path = "data/mnist", n_samples: int = 4096, seed: int = 0) -> Dataset:
    """A subsample of MNIST, for sanity-checking that the pipeline is dataset agnostic.

    ``n_samples`` is deliberately small: I(X;T) is bounded by ``log2(n_samples)``
    and the binning estimator costs O(n_samples log n_samples) per layer per
    checkpoint, so the full 60k training set is both slow and bound by a much
    looser ceiling.  Pixels are scaled to ``[-1, 1]``.
    """
    from torchvision import datasets  # imported lazily: optional dependency

    mnist = datasets.MNIST(str(root), train=True, download=True)
    data = mnist.data.numpy().reshape(len(mnist.data), -1).astype(np.float32) / 255.0
    targets = mnist.targets.numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=min(n_samples, len(data)), replace=False)
    return Dataset(
        X=(2.0 * data[idx] - 1.0),
        Y=targets[idx],
        name=f"mnist-{len(idx)}",
        n_classes=10,
        extras={},
    )


def build_dataset(
    kind: str,
    *,
    n_bits: int = N_TOY_BITS,
    geometry: str = "circle",
    path: str | Path | None = None,
    n_samples: int = 4096,
    seed: int = 0,
) -> Dataset:
    """Dispatch used by the CLI's ``--dataset`` flag."""
    if kind == "toy":
        return make_toy(n_bits=n_bits, geometry=geometry)
    if kind == "matfile":
        if path is None:
            raise ValueError("--dataset matfile requires --data-path")
        return load_matfile(path)
    if kind == "mnist":
        return make_mnist(root=path or "data/mnist", n_samples=n_samples, seed=seed)
    raise ValueError(f"unknown dataset {kind!r}")
