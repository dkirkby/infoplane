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
uv run pytest            # unit tests (~5 s)
uv run infoplane verify   # short end-to-end run checking the paper's invariants

# the real thing: 50 seeds x 9000 epochs, ~10 min on 8 cores
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json

# Figure 5: one ensemble per training-set size
uv run infoplane sweep --train-sizes 50,100,500,4000 --outdir data/trainsize
```

`infoplane run --help` lists every knob (architecture, optimizer, bin count, dataset).
Useful non-defaults:

| Flag | Why |
| --- | --- |
| `--lr 0.1` | Slower fitting phase, so the rise in I(X;T) spans more checkpoints before compression starts. |
| `--bins 10 / 100` | The binning estimator is sensitive to this; sweeping it shows how much of the measured compression is bin-dependent. |
| `--mi-on train` | Measure the information plane on the training subset instead of all 4096 patterns. |
| `--format arrow` | Long-format Feather instead of JSON, for large ensembles (needs `uv sync --extra arrow`). |
| `--dataset matfile --data-path var_u_data.mat` | Cross-check against the authors' published data (see below). |

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

- `H(X) = 12` bits exactly, `H(Y) ≈ 1` bit.
- `I(X;T) ≤ H(X)` and `I(T;Y) ≤ H(Y)`.
- Data processing inequality: `I(X;T₁) ≥ I(X;T₂) ≥ … ≥ I(X;T₇)` at every checkpoint. A
  violation means the measurement is broken, not an interesting result.
- Training loss converges (mean final train accuracy 1.000, test ≈ 0.998).

## Caveats that matter for Phase 2

**Figure 2 must plot `runs`, not `aggregated`.** The fitting phase is real but
seed-dependent, and ensemble averaging destroys it. In Layer 7, 45 of 50 seeds show
I(X;T) rising before it falls (median rise +0.95 bits, median peak at epoch 32, max
+2.53) -- yet the *mean* trajectory peaks at epoch 0, because seeds peak at different
epochs and the ones already compressing cancel the ones still fitting. So the per-seed
scatter shows the characteristic right-then-left arc while the mean shows only
compression. This is presumably why the paper plots individual runs in Figure 2; Figure
4's mean +- 1 sigma bands are the right place for `aggregated`.

**Layers 1-3 cannot show a fitting phase at all.** They sit pinned at I(X;T) = H(X) = 12
bits from initialisation onward: 30 bins across 7 or more `tanh` units resolve all 4096
patterns, so the measurement is saturated at its ceiling and has nowhere to rise. Only
layers 4-7 (widths 5, 4, 3, 2) carry the visible dynamics. The starting value is set by
the initialisation scale -- these networks start with order-1 pre-activations, so deep
layers begin at moderate I(X;T) and near-zero I(T;Y), i.e. trajectories enter the plane
from the bottom *right* of their eventual range rather than the bottom left.

**The absolute bit values are bin-dependent; only the shape is robust.** Re-measuring one
seed at several bin counts moves the numbers a long way:

| `--bins` | Layer 5 final I(X;T) | Layer 7 final I(X;T) | max DPI violation | violating measurements |
| --- | --- | --- | --- | --- |
| 15 | 2.61 bits | 1.10 bits | +0.083 | 5 |
| 30 (default) | 2.88 | 1.59 | +0.042 | 5 |
| 60 | 3.57 | 1.62 | +0.013 | 2 |
| 120 | 4.50 | 2.02 | +0.033 | 1 |

The fitting-then-compression *shape* survives all of these, but a reader should not take
"Layer 7 compresses to 1.6 bits" as a property of the network -- it is a property of the
network measured with 30 bins. This is the standard objection to the binning estimator
(Saxe et al., 2018). The frontend should state the bin count wherever it puts bits on an
axis, and `metadata.bins` is in every export for that purpose.

**The data processing inequality is violated by a hairline, rarely.** Over 50 seeds x 7
layers x 257 checkpoints, 5 of 77,100 measurements (0.006%) invert, by at most 0.05 bits,
all at the L5->L6 and L6->L7 transitions of a single seed. This is an artifact of the
estimator, not a bug: binning layer i+1 is not a coarsening of binning layer i, so when a
wide layer saturates and its bin grid merges patterns that the next layer's affine map
pulls back apart, the deeper layer's entropy can come out marginally higher. The table
above is the evidence -- violations shrink as the binning gets finer. `verify` therefore
allows 0.05 bits in under 0.1% of measurements; a large *fraction* of violations still
means something is broken.

The effect is much worse at small training sizes (0.21 bits for `--train-size 50`), where
layers saturate hardest. **Prefer `--bins 60` for the Figure 5 small-N panels.**

## The toy dataset

The paper specifies 12 binary inputs standing for 12 uniformly distributed points on a
sphere, labelled by thresholding a *spherically symmetric* function of the pattern — but
not what that function is. `infoplane.datasets.make_toy` is therefore a reconstruction,
not a copy of the authors' `var_u_data.mat`:

- bit `i` becomes a spin `sᵢ = 2xᵢ − 1 ∈ {−1, +1}` (matching the `tanh` range);
- the spins are projected onto the 12 unit directions `uᵢ`, giving `z(x) = Σ sᵢuᵢ`, a point
  in the plane (`--geometry circle`, the default) or in 3-space (`--geometry sphere`);
- the rule is the rotation-invariant radius `f(x) = ‖z(x)‖`, thresholded for balance.

Centring the bits to spins kills the linear term in `‖z‖²`, so the boundary is a genuine
quadric and a least-squares linear probe sits at chance (0.502). The threshold only ever
cuts *between* distinct radii — `f` has just 24 distinct values under the circle geometry,
so a median split would hand identical radii opposite labels and make the boundary
unlearnable at that radius. The price is `p(y=1) = 0.498` rather than exactly 0.5.

If you have the authors' `var_u_data.mat`, `--dataset matfile` loads it and the whole
pipeline runs unchanged — that is the intended cross-verification path.

## Export schema (the Phase 1 → Phase 2 contract)

```
{
  "metadata":    { every TrainConfig field, plus dataset_name, n_samples, seeds,
                   checkpoints, generated_utc, infoplane_version },
  "bounds":      { "H_X": 12.0, "H_Y": 1.0 },        # axis ceilings for the frontend
  "epochs":      [0, 1, 2, ...],                      # the checkpoint epochs
  "layers":      ["Layer 1", ..., "Layer 7"],
  "runs":        [ { "seed", "n_train", "seconds",
                     "trajectories": { "Layer 1": { "IXT": [...], "ITY": [...] }, ... },
                     "curves":      { "train_loss", "test_loss", "train_acc", "test_acc" } } ],
  "aggregated":  { "Layer 1": { "IXT_mean", "IXT_std", "ITY_mean", "ITY_std" }, ... },
  "curves":      { "train_loss": { "mean", "std" }, ... },
  "diagnostics": { "max_dpi_violation_bits", "final_train_acc_mean",
                   "final_test_acc_mean", "seconds_total" }
}
```

Every array is indexed by checkpoint and has length `metadata.checkpoints`. `runs` feeds
Figure 2's per-seed highlighting, `aggregated` feeds Figure 4's ±1σ bands, and `sweep`
writes one such file per training size plus an `index.json`. Prefer additive changes.

## Deliberate deviations from `PLAN.md`

1. **`epochs` holds the checkpoint epochs, not all 9000.** The schema sketch listed every
   epoch, but trajectories have one entry per checkpoint, so a 9000-long axis could not be
   zipped against them. `metadata.epochs` keeps the total training length.
2. **`--log-points` is the number of checkpoints delivered.** The sketched recipe
   `np.unique(np.logspace(0, log10(9000), 256).astype(int))` yields only 189 distinct
   epochs (early ones collide) and truncates its last point to 8999. The schedule rounds
   instead of truncating, and grows the request until 256 distinct epochs survive.
3. **Epoch 0 is a checkpoint** (`--no-init-checkpoint` to drop it): it is where every
   trajectory in Figure 2 starts.
4. **CPU by default, parallel across seeds.** `--device mps` works, but these networks are
   far too small to benefit — the win is 8 independent seeds on 8 cores, with one torch
   thread each. MPS plus `multiprocessing` is also a poor combination.
5. **Figure 5's training sizes are separate files** with an identical schema, rather than a
   new axis inside one document, so the frontend can lazy-load only the sizes on screen.
6. **MNIST is wired up but subsampled** (`--dataset mnist --n-samples 4096`): `I(X;T)` is
   bounded by `log2(n_samples)`, so the full 60k set is both slow and measured against a
   much looser ceiling. It exists to prove the pipeline is dataset-agnostic, and is not
   part of the replication.
