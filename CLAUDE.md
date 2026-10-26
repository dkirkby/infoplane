# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Replicate and interactively visualize Figures 2–5 of *Opening the Black Box of Deep Neural
Networks via Information* (Shwartz-Ziv & Tishby, 2017). Part of UCI's AI4SCI Deep Learning
course (`../ntk`, `../polyfit` are sibling course projects).

`PLAN.md` is the design spec. **Phase 1 (the Python pipeline) is implemented; Phase 2 (the
Vite + D3 frontend) is not started** — there is no `frontend/` yet. `README.md` documents
the export schema that joins them, and the deliberate deviations from `PLAN.md`; keep both
files current when that contract changes.

## Commands

```bash
uv sync                 # create .venv from uv.lock (Python 3.12)
uv run pytest           # 28 unit tests, ~5 s
uv run pytest tests/test_mi.py::test_mi_of_independent_variables_is_zero_for_a_product_distribution
uv run infoplane verify # ~10 s end-to-end run asserting the paper's invariants
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json   # ~10 min
uv run infoplane sweep --train-sizes 50,100,500,4000 --outdir data/trainsize    # Figure 5
```

No linter or formatter is configured. `data/` is gitignored — it is all regenerable.

## Architecture

The pipeline is a straight line, with each module independently testable:

`datasets.py` (enumerate X, Y) → `model.py` (the MLP) → `train.py` (train one seed, measure
at checkpoints) → `runner.py` (ensemble over seeds) → `export.py` (JSON contract) →
`cli.py` (argparse surface).

- **`datasets.py`** — `Dataset` is a frozen dataclass of `(X, Y)` numpy arrays. The toy rule
  is a *reconstruction* of the paper's unpublished labelling function (see README); the
  authors' `var_u_data.mat` loads via `--dataset matfile` for cross-checking.
- **`model.py`** — `InfoPlaneNet`, `d -> widths -> n_out`. `activations()` is what the
  measurement consumes. Seeded via `torch.random.fork_rng`, matching `../ntk`'s convention,
  so constructing a model never perturbs the caller's RNG.
- **`train.py`** — `TrainConfig` is the single source of truth for a run and is serialized
  verbatim into `metadata`; adding a knob means adding a field here and a flag in `cli.py`.
  `run_seed` interleaves training with measurement at the log-spaced checkpoints.
- **`mi.py`** — the estimator, operating only on integer symbol arrays.

## Things that are easy to break

- **The architecture is 7 hidden layers of widths 12, 10, 7, 5, 4, 3, 2**, with a separate
  12-dim input and a single sigmoid output unit. The figures' "Layer 1"…"Layer 7" are those
  hidden layers. The widths and `tanh` are load-bearing, not incidental.
- **`tanh` is required, not a preference.** `mi.py` bins activations over a hard-coded
  `[-1, 1]` support. ReLU would break the estimator *and* the phenomenon.
- **The binning range must stay fixed.** Binning over the observed min/max would rescale
  the estimate as a layer's dynamic range shrinks and manufacture compression that is not
  there.
- **X must stay enumerated** — distinct rows, uniform weight — or `H(X) = log2(N)` and the
  whole estimator lose their meaning.
- **`joint_symbols` deliberately avoids mixed-radix integer encoding**: `bins**width`
  overflows int64 for wide layers, so it uses `np.unique(..., axis=0)`.
- **MI magnitudes are bin-dependent.** Layer 5's final I(X;T) reads 2.61 bits at
  `--bins 15` and 4.50 at `--bins 120` (README has the table). Only the
  fitting-then-compression shape is robust, so never quote an absolute bit value without
  the bin count, and never compare runs made with different `--bins`.
- **DPI is a correctness check, not a result.** `I(X;T₁) ≥ … ≥ I(X;T₇)` must hold at every
  checkpoint (`train.dpi_violation`, reported in `diagnostics.max_dpi_violation_bits`). It
  comes out exactly monotone on a single seed, but 0.006% of measurements across 50 seeds
  invert by up to 0.05 bits, and more at small `--train-size`. That is an estimator
  artifact (binning layer i+1 is not a coarsening of binning layer i) and it shrinks as
  `--bins` grows — so `verify` checks a tolerance plus a violation *rate*, not exactness.
  The same applies more loosely to `I(T;Y)`. If the violation rate jumps, suspect the
  measurement.
- **Don't reach for MPS/CUDA.** The networks are tiny; all the speedup is from running
  seeds in separate processes with one torch thread each (`runner._init_worker`).

## Expected results (toy dataset, defaults)

**The fitting phase lives in `runs`, not `aggregated`.** 45/50 seeds show Layer 7's
I(X;T) rising before it compresses (median +0.95 bits, peak near epoch 32), but the
ensemble mean peaks at epoch 0 because seeds peak at different times and cancel. Use
per-seed trajectories for any plot that is supposed to show fitting-then-compression.
Layers 1-3 are pinned at H(X) = 12 bits throughout and cannot show it at all.


50 seeds, 9000 epochs: final train accuracy 1.000, test 0.9969, ~60 min of CPU across 8
workers. `I(T;Y)` reaches ~1 bit = `H(Y)` in every layer. Ensemble-mean I(X;T) compression
from peak to epoch 9000, by layer: 0.02, 0.20, 1.22, 3.42, 3.74, 3.37, 3.51 bits — i.e.
the signature is in layers 4–7, and Layer 7 ends near 1.1 bits. Training accuracy crosses
0.95 at epoch 84, so fitting is over in the first ~100 epochs while compression runs for
the remaining 8900.
