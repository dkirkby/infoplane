# infoplane

Interactive replication of the information-plane dynamics in *Opening the Black Box of
Deep Neural Networks via Information* (Shwartz-Ziv & Tishby, 2017). See `PLAN.md` for the
full two-phase design.

**Phase 1 (this package) is implemented**: it trains ensembles of the paper's network,
estimates I(X;T) and I(T;Y) for every hidden layer at log-spaced epochs, and exports the
trajectories as JSON for the Phase 2 frontend. Phase 2 (the Vite/D3 pages) is not started.

## Quick start

```bash
uv sync                  # create .venv from uv.lock
uv run infoplane fetch    # the authors' var_u.mat (1 KB, SHA-256 verified) -> data/
uv run pytest            # 38 unit tests, ~10 s
uv run infoplane verify   # short end-to-end run checking the paper's invariants

# the real thing: 50 seeds x 9000 epochs, ~10 min on 8 cores
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json

# Figure 5: one ensemble per training-set size
uv run infoplane sweep --train-sizes 50,100,500,4000 --outdir data/trainsize
```

`infoplane run --help` lists every knob. Useful non-defaults:

| Flag | Why |
| --- | --- |
| `--lr 0.1` | **Recommended for Figure 2.** Moves Layer 7's I(X;T) peak from epoch 1 to epoch 26, so the fitting arc spans many more checkpoints; the rise is larger (1.13 vs 0.88 bits) and test accuracy slightly better (0.9774), with the DPI still clean. |
| `--bins 10 / 100` | The binning estimator is sensitive to this; sweeping it shows how much of the measured compression is bin-dependent. |
| `--mi-on train` | Measure the information plane on the training subset instead of all 4096 patterns. |
| `--input-encoding binary` | The authors' raw 0/1 inputs instead of the default ±1. Reproduces their setup, at a real cost to the measurement — see below. |
| `--dataset quadric` | A synthetic deterministic rule, offline and deliberately harder — see below. Not the paper's problem. |
| `--format arrow` | Long-format Feather instead of JSON, for large ensembles (needs `uv sync --extra arrow`). |

## The dataset

The paper's own data is **`var_u.mat` from the authors' repository**
([ravidziv/IDNNs](https://github.com/ravidziv/IDNNs)) — note the path is `data/var_u.mat`,
not the `var_u_data.mat` the paper's text implies. `infoplane fetch` downloads it (1,003
bytes) and verifies its SHA-256; `--dataset var_u` is the default.

It holds all 4096 twelve-bit patterns as 0/1 rows ordered most-significant-bit first, with
one label each: `p(y=1) = 0.5171`, `H(Y) = 0.9992` bits, `H(X) = 12` bits exactly.

### What the labelling rule actually is

The paper says only that the 12 bits stand for 12 uniformly distributed points on a sphere
and that a *spherically symmetric* `f(x)` is pushed through
`p(y=1|x) = σ(γ(f(x) − θ))`. That is too vague to reimplement, so the rule was
characterised from the file itself:

- **The labels are exactly a thresholded quadratic form in the 12 bits.** A linear program
  over 12 linear + 66 pairwise + bias features finds a separator that reproduces all 4096
  labels, and no purely linear threshold is feasible. So the rule is **deterministic** —
  there is no label noise, and `I(X;Y) = H(Y)`.
- **It is 87.5% linearly separable**, because the label profile is strongly monotone in the
  number of set bits: shells k≤4 are all 0, k≥8 all 1, and only k=5,6,7 are mixed
  (p(y=1) = 0.076, 0.576, 0.924). A threshold on the bit count alone gets 87.5%.
- **But it is not a function of the bit count.** Within shell k=6, a quadratic probe
  reaches 0.972 cross-validated accuracy against a 0.576 majority baseline.

### Two reconstructions the file refuted

Both are recorded because each looked right until measured, and the second is the subtler
trap:

1. **Radius of projected spins** (now `--dataset quadric`): bits to spins
   `sᵢ = 2xᵢ − 1`, projected onto 12 directions on the circle, thresholding
   `f(x) = ‖Σ sᵢuᵢ‖`. Deterministic and genuinely non-linear — but centring to spins kills
   the linear term, so its label profile is *independent* of the bit count where the real
   one is strongly monotone, and a linear probe sits at chance (0.502) rather than 0.875.
   A different, harder problem. Kept as the offline dataset for tests and as a contrast
   case.
2. **Bit count plus sigmoid noise**: `f(x) = popcount(x)` with γ = 2.62, θ = 5.92 fitted by
   maximum likelihood. This reproduced the shell profile, `p(y=1)`, and the 87.5% linear
   probe — it looked like a match. It is wrong: it replaces real quadratic structure with
   12.5% label noise. Removed from the package.

The mistake that made (2) look correct was testing *within-shell* structure with a linear
probe, which is blind to a pure quadratic form — it returned exactly the majority rate, and
that was read as "no structure". Random bit permutations agreed with the real labels at
0.8358 ± 0.0121 against 0.8356 predicted by a count-only model, which seemed to confirm it;
but a permutation test cannot distinguish "stochastic given the count" from "deterministic
in a geometric `f` with within-shell spread", because a permutation randomises which side
of the threshold a pattern lands on either way. **A probe has to be matched to the
hypothesis it is meant to reject.**

The practical difference is large: at 500 epochs the noisy stand-in reached 0.93 train /
0.86 test accuracy, against 0.995 / 0.95 on the real file, because its noise scatters
mislabelled patterns into shells 4 and 8 that the real data never contains, and isolated
exceptions deep inside a uniform region are far harder to memorise.

## How the measurement works

`X` is *enumerated*, not sampled: all 2^12 = 4096 input patterns appear exactly once, so
`p(x) = 1/4096` and `H(X) = 12` bits exactly. That is the only reason the plug-in estimator
is meaningful, and any new dataset must preserve it.

At each checkpoint, every hidden layer's `tanh` activations are quantised into 30 uniform
bins over the *fixed* support `[-1, 1]` (never the observed range — an adaptive range would
manufacture compression as the layer's dynamic range shrinks). A layer's state `T` is the
joint bin tuple across its units; two samples share a state only if every unit agrees.
`I(X;T)` and `I(T;Y)` are then the plug-in mutual informations of those discrete symbols,
in bits.

Because inputs are distinct and `T` is deterministic given `X`, `I(X;T) = H(T)`. The code
computes the general two-variable estimator anyway, so that identity acts as a check
rather than an assumption.

### Invariants worth re-checking after any change

`infoplane verify` asserts all of these on a short run; the full run reports the last two
in `diagnostics`:

- `H(X) = 12` bits exactly, `H(Y) ≈ 1` bit, and `I(X;Y) = H(Y)` (the rule is deterministic).
- `I(X;T) ≤ H(X)` and `I(T;Y) ≤ H(Y)`.
- Data processing inequality: `I(X;T₁) ≥ I(X;T₂) ≥ … ≥ I(X;T₇)` at every checkpoint.
- Training loss converges.

## Caveats that matter for Phase 2

**Input encoding changes the measurement, not just the dynamics.** The authors fed their
0/1 patterns in raw. Those are all non-negative, which pushes the narrow layers into `tanh`
saturation, collapses bins, and breaks the estimator where it matters most. On the
ensemble's worst seed (18; the bin table below uses seed 0, so its absolute values differ):

| `--input-encoding` | `--bins` | DPI violations | max violation | Layer 5 final I(X;T) | test acc |
| --- | --- | --- | --- | --- | --- |
| binary | 30 | 7 / 1542 | +0.381 bits | 2.47 bits | 0.9691 |
| binary | 60 | 2 / 1542 | +0.124 | 3.15 | 0.9691 |
| binary | 120 | 2 / 1542 | +0.117 | 3.61 | 0.9691 |
| **spin** (default) | 30 | **0 / 1542** | **0.000** | **4.70** | 0.9772 |

Layer 5 reads 2.47 bits under `binary` and 4.70 under `spin` — nearly a factor of two on
the same network and the same data. The encoding is a network *input scaling*, not part of
anyone's dataset (labels and patterns are identical either way), so `spin` is the default
here despite the authors having used 0/1: it is the only setting that satisfies the data
processing inequality exactly at the paper's 30 bins. Use `--input-encoding binary` to
reproduce their exact setup, and expect the narrow layers to be under-measured if you do.

**The fitting phase is brief, and much clearer per-seed than in the mean.** With the
default settings it is over within the first couple of epochs: 45 of 50 seeds show Layer 7
rising before it compresses, by a median of 0.94 bits and up to 2.49, with the peak at
epoch 1-2 (Layer 6: 40/50, median 0.78; Layer 5: 32/50, median 0.39; layers 1-3 are flat
at the ceiling). The ensemble mean does rise, but only by +0.49 bits in Layer 7 — a fifth
of the largest single-seed rise — because seeds peak at slightly different epochs and
cancel.

So Figure 2 should plot `runs`, not `aggregated`, and its epoch scrubber needs fine
resolution over the first ~10 epochs, where almost all of the rightward motion happens.
`--lr 0.1` slows fitting and spreads that arc over many more checkpoints — Layer 7's peak
moves from epoch 1 to epoch 26, and the median rise grows from 0.88 to 1.13 bits — at no
cost to accuracy (test 0.9774) or to the DPI check (0/12,336). Worth doing for the figure. Figure 4's
mean ± 1σ bands are the right place for `aggregated`.

**The widest layers cannot show a fitting phase at all.** Layers 1-2 sit pinned at
`I(X;T) = H(X) = 12` bits from initialisation onward: 30 bins across 10 or more `tanh`
units resolve all 4096 patterns, so the measurement is saturated at its ceiling and has
nowhere to rise.

**The absolute bit values are bin-dependent; only the shape is robust.** Re-measuring one
`var_u` seed (seed 0) under the defaults at several bin counts moves the mid layers a long way,
while leaving the deepest layer and the *shape* of every trajectory intact:

| `--bins` | L4 final | L5 final | L6 final | L7 final | DPI violations |
| --- | --- | --- | --- | --- | --- |
| 10 | 6.23 bits | 3.08 | 1.52 | 1.03 | 0 / 1542 |
| 15 | 7.17 | 4.03 | 1.81 | 1.04 | 0 / 1542 |
| 30 (default, the paper's) | 8.50 | 5.38 | 2.89 | 1.06 | 0 / 1542 |
| 60 | 9.59 | 6.38 | 3.30 | 1.15 | 0 / 1542 |
| 120 | 10.50 | 7.52 | 3.95 | 1.32 | 0 / 1542 |

Layer 5 more than doubles between 10 and 120 bins; Layer 7 barely moves. So "Layer 5
compresses to 5.4 bits" is a property of the network *measured with 30 bins*, not of the
network — the standard objection to the binning estimator (Saxe et al., 2018). The
frontend should state the bin count wherever it puts bits on an axis; `metadata.bins` is in
every export for that. Never compare runs made with different `--bins`.

Note that the inequality stays clean at every bin count here. Bin count moves the
magnitudes; it is the *input encoding* that determines whether the estimator is well
conditioned at all.

**Current numbers under the defaults** (50 seeds x 9000 epochs, `spin` inputs, 30 bins):
train accuracy 1.000, test 0.9734, **0 DPI violations in 77,100 measurements**; mean I(X;T)
compression by layer 0.00 / 0.07 / 1.10 / 3.86 / 5.06 / 4.74 / 4.03 bits. Figure 5 sweep
test accuracy: 0.8057 (N=50), 0.8369 (N=100), 0.8725 (N=500), 0.9769 (N=4000).

**The data processing inequality is the measurement's own smoke alarm.** `I(X;T₁) ≥ … ≥
I(X;T₇)` must hold at every checkpoint. The plug-in binned estimator can invert it even
when the true quantities are ordered, because binning layer i+1 is not a coarsening of
binning layer i: when a layer saturates and its bin grid merges patterns that the next
layer's affine map pulls back apart, the deeper layer's entropy comes out higher. But the
size of that effect tracks how badly conditioned the measurement is, so it is worth
watching rather than excusing:

- `spin` inputs at 30 bins (the default): 0 of 77,100 on the full dataset; worst across the
  Figure 5 sweep is 2 of 77,100 at +0.086 bits, at N=50.
- `binary` inputs at 30 bins: 126 of 77,100 measurements (0.16%) up to **+0.381 bits**,
  clustered at the L4→L5 and L5→L6 transitions — e.g. one seed reading Layer 5 at 2.757
  bits and Layer 6 at 3.138. That is a 14% error on the layer, and it is what prompted the
  default to change.
- small training sizes are worse still (up to +1.44 bits at N=100 under `binary`), because
  layers saturate hardest there.

Finer binning also reduced the violations under `binary` (7 -> 2 -> 2 for 30/60/120 bins on
the worst seed), which is what first identified them as a saturation artifact rather than a
bug; under `spin` there is nothing left to reduce.

`verify` allows 0.05 bits in under 0.1% of measurements, and every export carries
`dpi_violations` / `dpi_measurements`. **Check that rate before trusting a run**: a short
`verify` run is not representative, since the violations appear well into training.

## Export schema (the Phase 1 → Phase 2 contract)

```
{
  "metadata":    { every TrainConfig field, plus dataset_name, n_samples, seeds,
                   checkpoints, generated_utc, infoplane_version },
  "bounds":      { "H_X": 12.0, "H_Y": 0.9992 },     # axis ceilings for the frontend
  "epochs":      [0, 1, 2, ...],                      # the checkpoint epochs
  "layers":      ["Layer 1", ..., "Layer 7"],
  "runs":        [ { "seed", "n_train", "seconds",
                     "trajectories": { "Layer 1": { "IXT": [...], "ITY": [...] }, ... },
                     "curves":      { "train_loss", "test_loss", "train_acc", "test_acc" } } ],
  "aggregated":  { "Layer 1": { "IXT_mean", "IXT_std", "ITY_mean", "ITY_std" }, ... },
  "curves":      { "train_loss": { "mean", "std" }, ... },
  "diagnostics": { "max_dpi_violation_bits", "dpi_violations", "dpi_measurements",
                   "final_train_acc_mean", "final_test_acc_mean", "seconds_total" }
}
```

Every array is indexed by checkpoint and has length `metadata.checkpoints`. `runs` feeds
Figure 2's per-seed highlighting, `aggregated` feeds Figure 4's ±1σ bands, and `sweep`
writes one such file per training size plus an `index.json`. Prefer additive changes.

## Deliberate deviations from `PLAN.md`

1. **The dataset is the authors' file, not a generator.** `PLAN.md` called for a synthetic
   12-bit generator cross-verified against the original; the original turned out to be both
   available and not reproducible from the paper's prose, so `var_u.mat` is the default and
   the synthetic rule is demoted to a contrast case.
2. **Inputs are centred to ±1 by default**, where the authors used raw 0/1. Same labels,
   same patterns — but 0/1 saturates the narrow layers and makes the binning estimator
   violate the data processing inequality by up to 0.38 bits. `--input-encoding binary`
   restores their setup.
3. **`epochs` holds the checkpoint epochs, not all 9000.** The schema sketch listed every
   epoch, but trajectories have one entry per checkpoint, so a 9000-long axis could not be
   zipped against them. `metadata.epochs` keeps the total training length.
4. **`--log-points` is the number of checkpoints delivered.** The sketched recipe
   `np.unique(np.logspace(0, log10(9000), 256).astype(int))` yields only 189 distinct
   epochs (early ones collide) and truncates its last point to 8999. The schedule rounds
   instead of truncating, and grows the request until 256 distinct epochs survive.
5. **Epoch 0 is a checkpoint** (`--no-init-checkpoint` to drop it): it is where every
   trajectory in Figure 2 starts.
6. **CPU by default, parallel across seeds.** `--device mps` works, but these networks are
   far too small to benefit — the win is 8 independent seeds on 8 cores, with one torch
   thread each. MPS plus `multiprocessing` is also a poor combination.
7. **Figure 5's training sizes are separate files** with an identical schema, rather than a
   new axis inside one document, so the frontend can lazy-load only the sizes on screen.
8. **MNIST is wired up but subsampled** (`--dataset mnist --n-samples 4096`): `I(X;T)` is
   bounded by `log2(n_samples)`, so the full 60k set is both slow and measured against a
   much looser ceiling. It exists to prove the pipeline is dataset-agnostic, and is not
   part of the replication.
