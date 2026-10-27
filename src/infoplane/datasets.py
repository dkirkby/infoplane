"""Dataset generators for the information-plane experiments.

The quantity that makes this whole replication tractable is that ``X`` is a *small
discrete* variable with a known distribution: every input pattern is enumerated
exactly once, so ``p(x) = 1/N`` and ``H(X) = log2(N)`` exactly.  Any dataset added
here must preserve that property (distinct rows, uniform weight) or the binning
estimator in :mod:`infoplane.mi` loses its meaning.

The paper's own dataset is ``var_u`` -- the authors' ``var_u.mat``, fetched and
checksummed by :func:`fetch_var_u`.  Use it for anything meant to replicate the
paper.  Two attempts to *synthesise* their labels from the paper's prose were both
refuted by the file itself (see the README), so no generator here claims to
reproduce it; :func:`make_quadric` is an explicitly different problem.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_TOY_BITS = 12

# The authors' 12-bit dataset, from https://github.com/ravidziv/IDNNs.  Note the
# file is `data/var_u.mat`, not the `var_u_data.mat` the paper's text implies.
VAR_U_URL = "https://raw.githubusercontent.com/ravidziv/IDNNs/master/data/var_u.mat"
VAR_U_SHA256 = "0ba9551878a855396a8de0cbaae620788ede1e7b1eb2b8373bbe017d5ea02036"
DEFAULT_VAR_U_PATH = Path("data/var_u.mat")

DEFAULT_VAR_U_PATH = Path("data/var_u.mat")

# Maximum-likelihood fit of p(y=1|x) = sigmoid(gamma * (popcount(x) - theta)) to the
# 4096 labels in var_u.mat.  Per-shell binomial tests are consistent with it (the
# worst, k=7, sits at p=0.016 across 13 shells).
VAR_U_GAMMA = 2.6246
VAR_U_THETA = 5.9206


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
        Diagnostic arrays (e.g. the bit counts the toy rule is built on) that are
        useful for plotting and checking but are not fed to the network.
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


def _encode(bits: np.ndarray, encoding: str) -> np.ndarray:
    """Map 0/1 patterns to the network's input range.

    ``binary`` keeps the authors' raw 0/1 inputs and is the default; ``spin``
    centres them to ``-+1``, which matches the ``tanh`` range.  The two are
    affinely equivalent to a layer with a bias, so they change neither the labels
    nor what is representable -- this is a training detail, not part of anyone's
    dataset -- but they do change the dynamics and with them the measurement.

    This default was ``spin`` for a while, on the evidence that ``binary`` drove
    the narrow layers into ``tanh`` saturation and made the estimator violate the
    data processing inequality (126 of 77,100 measurements, up to +0.381 bits).
    That turned out to be an artifact of a *different* bug: the network then had
    seven hidden layers, from reading the paper's "12-10-7-5-4-3-2" literally
    instead of as input + five hidden + output.  With the authors' five-layer
    network both encodings give 0 violations in 51,400 measurements, so the reason
    to deviate is gone and the default follows the authors again.  ``spin`` still
    trains slightly better (test 0.9779 vs 0.9682) and compresses further, but
    "our variant shows more of the phenomenon" is not a reason to prefer it in a
    replication.
    """
    bits = bits.astype(np.float32, copy=False)
    if encoding == "binary":
        return bits
    if encoding == "spin":
        return 2.0 * bits - 1.0
    raise ValueError(f"unknown input encoding {encoding!r}")


def _all_bit_patterns(n_bits: int) -> np.ndarray:
    """All ``2**n_bits`` patterns as a ``(2**n_bits, n_bits)`` array of 0/1.

    Ordered most-significant bit first, matching the row order of ``var_u.mat``.
    """
    codes = np.arange(2**n_bits, dtype=np.int64)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((codes[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def make_quadric(
    n_bits: int = N_TOY_BITS, geometry: str = "circle", encoding: str = "binary"
) -> Dataset:
    """A synthetic deterministic rule -- **not** the paper's problem; use ``var_u``.

    Bits become spins ``s_i = 2 x_i - 1``, which are projected onto ``n_bits`` unit
    directions (equally spaced on the circle, or a Fibonacci sphere in R^3), and the
    label thresholds the rotation-invariant radius ``f(x) = ||sum_i s_i u_i||`` at
    the most balanced cut *between* distinct radii.

    Centring to spins removes the linear term from ``||z||^2``, so a linear probe
    sits at chance here, whereas the authors' rule is 87.5% linearly separable and
    strongly monotone in the number of set bits.  That is why this is kept only as a
    contrast case -- a hard, fully non-linear rule to compare the information plane
    against -- and as the offline dataset for the tests.  It was this package's
    original stand-in for the paper's data, until ``var_u.mat`` showed it was wrong.
    """
    if geometry == "circle":
        angles = 2.0 * np.pi * np.arange(n_bits) / n_bits
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    elif geometry == "sphere":
        i = np.arange(n_bits) + 0.5
        z = 1.0 - 2.0 * i / n_bits
        r = np.sqrt(np.maximum(0.0, 1.0 - z**2))
        phi = np.pi * (1.0 + 5.0**0.5) * i
        directions = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    bits = _all_bit_patterns(n_bits)
    spins = 2.0 * bits.astype(np.float64) - 1.0
    z = spins @ directions
    # Round away float noise so genuinely equal radii land in one tie group.
    f = np.round(np.linalg.norm(z, axis=1), 9)

    # Cut only between distinct radii: f is massively degenerate (24 distinct values
    # for the circle), so a median split would hand identical radii opposite labels
    # and make the boundary unlearnable at that radius.
    values, counts = np.unique(f, return_counts=True)
    above = counts.sum() - np.concatenate([[0], np.cumsum(counts)[:-1]])
    theta = float(values[int(np.argmin(np.abs(above / counts.sum() - 0.5)))])

    return Dataset(
        X=_encode(bits, encoding),
        Y=(f >= theta).astype(np.int64),
        name=f"quadric{n_bits}-{geometry}",
        n_classes=2,
        extras={"latent": z.astype(np.float32), "radius": f, "threshold": np.asarray(theta)},
    )


def fetch_var_u(dest: str | Path = DEFAULT_VAR_U_PATH, force: bool = False) -> Path:
    """Download the authors' ``var_u.mat`` and verify its SHA-256.

    Returns the local path.  Already-present files are checksummed, not re-fetched.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == VAR_U_SHA256:
            return dest
        raise ValueError(
            f"{dest} exists but its SHA-256 is {digest}, expected {VAR_U_SHA256}; "
            "pass force=True to re-download"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(VAR_U_URL) as response:  # noqa: S310 - fixed https URL
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != VAR_U_SHA256:
        raise ValueError(f"downloaded {VAR_U_URL} but SHA-256 is {digest}, expected {VAR_U_SHA256}")
    dest.write_bytes(payload)
    return dest


def load_matfile(path: str | Path, encoding: str = "binary") -> Dataset:
    """Load the authors' ``var_u.mat`` (4096x12 ``F`` of 0/1, ``y`` of labels)."""
    from scipy.io import loadmat

    raw = loadmat(str(path))
    if "F" not in raw or "y" not in raw:
        raise ValueError(f"{path} has keys {sorted(raw)}; expected 'F' and 'y'")
    bits = np.asarray(raw["F"])
    if bits.min() < 0 or bits.max() > 1:
        raise ValueError(f"{path}: expected 0/1 inputs, got range [{bits.min()}, {bits.max()}]")
    y = np.asarray(raw["y"]).squeeze()
    if y.ndim == 2:  # one-hot layouts used by some revisions of that file
        y = y.argmax(axis=0) if y.shape[0] < y.shape[1] else y.argmax(axis=1)
    Y = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(Y) != len(bits):
        raise ValueError(f"{path}: {len(bits)} inputs but {len(Y)} labels")
    return Dataset(
        X=_encode(bits, encoding),
        Y=Y,
        name=Path(path).stem,
        n_classes=int(Y.max()) + 1,
        extras={"count": bits.sum(1).astype(np.float64)},
    )


def build_dataset(
    kind: str,
    *,
    n_bits: int = N_TOY_BITS,
    geometry: str = "circle",
    path: str | Path | None = None,
    encoding: str = "binary",
) -> Dataset:
    """Dispatch used by the CLI's ``--dataset`` flag."""
    if kind == "quadric":
        return make_quadric(n_bits=n_bits, geometry=geometry, encoding=encoding)
    if kind == "var_u":
        return load_matfile(fetch_var_u(path or DEFAULT_VAR_U_PATH), encoding=encoding)
    if kind == "matfile":
        if path is None:
            raise ValueError("--dataset matfile requires --data-path")
        return load_matfile(path, encoding=encoding)
    raise ValueError(f"unknown dataset {kind!r}")
