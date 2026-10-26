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
uv run infoplane fetch  # the authors' var_u.mat -> data/ (needed by --dataset var_u)
uv run pytest           # 38 unit tests, ~10 s
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

- **`datasets.py`** — `Dataset` is a frozen dataclass of `(X, Y)` numpy arrays. The default
  `--dataset var_u` is the authors' own `var_u.mat`, fetched and checksummed by
  `fetch_var_u` (their file is `data/var_u.mat`, *not* the `var_u_data.mat` the paper
  implies). `--dataset quadric` is a synthetic rule that is **not** the paper's problem.
- **`model.py`** — `InfoPlaneNet`, `d -> widths -> n_out`. `activations()` is what the
  measurement consumes. Seeded via `torch.random.fork_rng`, matching `../ntk`'s convention,
  so constructing a model never perturbs the caller's RNG.
- **`train.py`** — `TrainConfig` is the single source of truth for a run and is serialized
  verbatim into `metadata`; adding a knob means adding a field here and a flag in `cli.py`.
  `run_seed` interleaves training with measurement at the log-spaced checkpoints.
- **`mi.py`** — the estimator, operating only on integer symbol arrays.

## Don't try to synthesise the paper's labels

Two reconstructions were tried and both were refuted by `var_u.mat` (README has the
detail). What the file measurably contains: the labels are **exactly a thresholded
quadratic form** in the 12 bits (LP-verified over 12 linear + 66 pairwise + bias features;
no linear threshold is feasible), hence deterministic with no label noise. The profile is
monotone in the bit count — shells ≤4 all 0, ≥8 all 1, only 5/6/7 mixed — so a count
threshold reaches 87.5%, but the labels are *not* a function of the count: a quadratic
probe inside shell 6 gets 0.972 CV against a 0.576 majority.

The methodological trap, worth remembering: a **linear** probe inside a shell returns
exactly the majority rate for a pure quadratic rule, which reads as "no structure". A
permutation test cannot separate "stochastic given the count" from "deterministic in a
geometric f" either. Match the probe to the hypothesis being rejected.

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
- **`--input-encoding` changes the measurement.** Default is `spin` (±1). The authors' raw
  `binary` 0/1 inputs saturate the narrow layers, collapsing bins: Layer 5 reads 2.47 bits
  under `binary` versus 4.70 under `spin`, and `binary` violates the DPI by up to 0.38
  bits at 30 bins. Don't switch to `binary` without re-reading the DPI rate, and never
  compare runs across encodings.
- **MI magnitudes are bin-dependent.** Layer 5's final I(X;T) reads 2.61 bits at
  `--bins 15` and 4.50 at `--bins 120` (README has the table). Only the
  fitting-then-compression shape is robust, so never quote an absolute bit value without
  the bin count, and never compare runs made with different `--bins`.
- **DPI is a correctness check, not a result.** `I(X;T₁) ≥ … ≥ I(X;T₇)` must hold at every
  checkpoint (`train.dpi_violation`, reported in `diagnostics.max_dpi_violation_bits`). It
  is near-exact with the default `spin` inputs, but the estimator can invert it where a
  layer saturates (binning layer i+1 is not a coarsening of binning layer i), and the
  magnitude tracks how badly conditioned the measurement is — 0.38 bits under `binary`
  inputs, up to 1.44 bits at small `--train-size`. `verify` checks a tolerance plus a
  violation *rate*, not exactness, and the rate is in every export's `diagnostics`. A
  short `verify` run is NOT representative: violations appear well into training, so
  check `dpi_violations` on the full run before trusting it.
- **Don't reach for MPS/CUDA.** The networks are tiny; all the speedup is from running
  seeds in separate processes with one torch thread each (`runner._init_worker`).

## Expected results (`var_u`, defaults)

50 seeds x 9000 epochs, ~50 min of CPU across 8 workers: final train accuracy 1.000, test
0.9734, and **0 DPI violations in 77,100 measurements**. Training accuracy crosses 0.95 at
epoch 35. Ensemble-mean I(X;T) compression from peak to epoch 9000, by layer: 0.00, 0.07,
1.10, 3.86, 5.06, 4.74, 4.03 bits — the signature is in layers 4-7. Layers 1-3 sit pinned
at H(X) = 12 bits (30 bins across 7+ `tanh` units resolve all 4096 patterns) and can
neither compress nor show a fitting phase.

Figure 5 sweep, mean final test accuracy: 0.8057 (N=50), 0.8369 (N=100), 0.8725 (N=500),
0.9769 (N=4000).

`--dataset quadric` for contrast reaches test 0.9981 with compression 0.02, 0.16, 1.11,
3.44, 3.88, 3.44, 3.54 — less in the mid layers than `var_u`, being a deterministic rule
with no near-linear component.

**The fitting phase is brief and per-seed.** It is over within ~2 epochs: 45/50 seeds show
Layer 7 rising first, by a median 0.94 bits (max 2.49), but the mean rises only +0.49
because seeds peak at different epochs and cancel. Use per-seed trajectories for any plot
meant to show fitting-then-compression, and `--lr 0.1` to spread the arc over more
checkpoints.
