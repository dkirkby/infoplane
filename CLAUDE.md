# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Replicate and interactively visualize Figures 2–5 of *Opening the Black Box of Deep Neural
Networks via Information* (Shwartz-Ziv & Tishby, 2017). Part of UCI's AI4SCI Deep Learning
course (`../ntk`, `../polyfit` are sibling course projects).

`PLAN.md` is the design spec. **Both phases are implemented**: the `infoplane` Python
package (Phase 1) and the Vite + D3 site in `frontend/` (Phase 2). `README.md` documents
the export schema that joins them and the deliberate deviations from `PLAN.md`; keep both
files current when that contract changes.

**The paper's text, the authors' code and the authors' data file disagree with each other
in several places**, and this project follows the code and the file, because those are what
produced the figures. README's "What the paper says vs what the authors ran" is the full
list; the three that bite are the optimiser (text says SGD, code runs Adam at 4e-4), the
architecture (see below), and the labels (text says stochastic, file is deterministic).
Reading the paper alone will lead you to re-introduce bugs that have already been fixed
here.

## Commands

```bash
uv sync                 # create .venv from uv.lock (Python 3.12)
uv run infoplane fetch  # the authors' var_u.mat -> data/ (needed by --dataset var_u)
uv run pytest           # 38 unit tests, ~10 s
uv run pytest tests/test_mi.py::test_mi_of_independent_variables_is_zero_for_a_product_distribution
uv run infoplane verify # ~11 s end-to-end run asserting the paper's invariants
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json   # ~5 min
uv run infoplane sweep --train-sizes 50,100,500,4000 --outdir data/trainsize    # Figure 5

cd frontend && npm install && npm run dev     # the figure pages (needs the data above)
cd frontend && npm run build                  # static site -> frontend/dist
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

## Frontend (`frontend/`)

Five static pages (`index`, `fig2`..`fig5`), one Vite entry each — not an SPA. Shared
modules in `src/scripts/`: `chart.js` (frames, axes, legend, tooltip, table view),
`data.js` (fetch + reshape the export into typed arrays), `scrubber.js`, `theme.js`.
`public/data` is a symlink to the repo's `data/`, dereferenced into `dist` at build.

Rules that are load-bearing, not taste — a change here needs the same validation:

- **Layer colour is an ORDINAL ramp** (one blue hue, 7 steps, monotone lightness),
  generated in OKLCH and validated per theme. Never substitute a categorical or rainbow
  palette: depth is ordered, and in a scatter only three categorical slots clear the
  colourblind floor. Series that are *not* layers (train vs test) use `.c1`/`.c2`,
  categorical slots 1-2.
- Marks take a `l1`..`l7` class and read `--c` from CSS, so a theme switch repaints
  without a re-render and a layer's colour never follows its rank. The default network has
  five hidden layers, so a default run uses `l1`..`l5`; `l6`/`l7` stay defined for deeper
  `--widths`. Truncating the ramp is safe (every gate is per-adjacent-pair or on the
  surface-side end) and was re-validated as 5 slots, but extending or re-ordering it is
  not — run the dataviz skill's `validate_palette.js --ordinal` in both modes.
- **Never put I(X;T) and I(T;Y) on one pair of axes** (0-12 bits vs 0-1).
- Figure 2 reads `runs`, not `aggregated` — see the expected-results note below.
- Every chart keeps its table view, legend, and keyboard path.

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

- **The architecture is 5 hidden layers of widths 10, 7, 5, 4, 3**, with a separate 12-dim
  input and a single sigmoid output unit. The figures' "Layer 1"…"Layer 5" are those hidden
  layers. The paper's §4.1 writes "12-10-7-5-4-3-2" and calls it "7 fully connected hidden
  layers" — that sentence is a trap. Its own Figure 4 caption writes a network as
  `input=12-10-8-6-4-2-1=output`, and §4.1 adds "layer 1 is the hidden layer closest to the
  input and the highest is the output layer", so the first entry is the input and the last
  is the output. The authors' code agrees (`select_network_arch('1') -> [10,7,5,4,3]`).
  **Do not "fix" this back to seven layers.** That reading was implemented and measured: it
  pins three layers at the 12-bit ceiling instead of one, traps 8 of 50 seeds near 0.92
  train accuracy (unchanged by training 30,000 epochs instead of 9,000), and produces every
  DPI violation this project has ever recorded.
- **`tanh` is required, not a preference.** `mi.py` bins activations over a hard-coded
  `[-1, 1]` support. ReLU would break the estimator *and* the phenomenon.
- **The binning range must stay fixed.** Binning over the observed min/max would rescale
  the estimate as a layer's dynamic range shrinks and manufacture compression that is not
  there.
- **X must stay enumerated** — distinct rows, uniform weight — or `H(X) = log2(N)` and the
  whole estimator lose their meaning.
- **`joint_symbols` deliberately avoids mixed-radix integer encoding**: `bins**width`
  overflows int64 for wide layers, so it uses `np.unique(..., axis=0)`.
- **`--input-encoding` changes the measurement.** Default is `binary`, the authors' raw
  0/1 inputs. `spin` (±1) trains to slightly better test accuracy (0.9779 vs 0.9682) and
  compresses much further (Layer 5 to 1.09 bits vs 2.16), which is exactly why it was the
  default for a while and exactly why it should not be: showing *more* of the phenomenon is
  not a reason to prefer a setting the paper did not use. The fitting phase is also clearer
  under `binary` (50/50 seeds vs 44/50). Never compare runs across encodings.
- **MI magnitudes are bin-dependent.** Layer 5's final I(X;T) reads 2.61 bits at
  `--bins 15` and 4.50 at `--bins 120` (README has the table). Only the
  fitting-then-compression shape is robust, so never quote an absolute bit value without
  the bin count, and never compare runs made with different `--bins`.
- **DPI is a correctness check, not a result.** `I(X;T₁) ≥ … ≥ I(X;T₅)` must hold at every
  checkpoint (`train.dpi_violation`, reported in `diagnostics.max_dpi_violation_bits`). On
  the current defaults it is exact: 0 violations in 51,400, under both encodings and at
  every `--train-size`. The estimator *can* invert it where a layer saturates (binning
  layer i+1 is not a coarsening of binning layer i), and the magnitude tracks how badly
  conditioned the measurement is. Treat any nonzero rate as a signal that something about
  the run is wrong — but diagnose it rather than tuning knobs until it goes away: these
  violations were once blamed on the input encoding (switching encodings did silence them)
  when the actual cause was two surplus layers. `verify` checks a tolerance plus a
  violation *rate*, not exactness. A short `verify` run is NOT representative: violations
  appear well into training, so check `dpi_violations` on the full run before trusting it.
- **Don't reach for MPS/CUDA.** The networks are tiny; all the speedup is from running
  seeds in separate processes with one torch thread each (`runner._init_worker`).

## Expected results (`var_u`, defaults)

50 seeds x 9000 epochs, ~4 min wall on 8 workers (~35 min of CPU): final train accuracy
0.9988, test 0.9682, and **0 DPI violations in 51,400 measurements**. Train accuracy
crosses 0.95 at epoch 557 and 0.99 at epoch 2884. Ensemble-mean I(X;T) compression from
peak to epoch 9000, by layer: 0.00, 0.02, 0.11, 2.74, 5.53 bits — the signature is in
layers 4-5. Layer 1 sits pinned at H(X) = 12 bits (30 bins across 10 `tanh` units resolve
all 4096 patterns) and Layer 2 nearly so, moving 0.02 bits over the whole run.

The paper's Figure 2 snapshots, as ensemble means of (I(X;T), I(T;Y)):

| epoch | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
| --- | --- | --- | --- | --- | --- |
| 400 | 12.00, 1.00 | 12.00, 1.00 | 11.58, 0.98 | 8.81, 0.92 | 5.12, 0.83 |
| 9000 | 12.00, 1.00 | 11.98, 1.00 | 11.62, 1.00 | 7.26, 0.99 | 2.16, 0.97 |

**If a change moves these, check the epoch axis before anything else.** It is the most
fragile thing here: it is set entirely by the optimiser settings, and getting it wrong
makes the figures look fine while showing the wrong phenomenon. SGD at lr 0.4 fits this
network in ~2 epochs, which puts the entire ERM phase between the first two log-spaced
checkpoints and leaves nothing but compression on screen.

Figure 5 sweep, mean final test accuracy: 0.7681 (N=50), 0.8339 (N=100), 0.8901 (N=500),
0.9787 (N=4000). All four are DPI-clean (0 / 51,400 each).

`--dataset quadric` for contrast reaches test 0.9950 with compression 0.00, 0.22, 0.87,
4.53, 5.98 — more in the deep layers than `var_u`, being a deterministic rule the network
fits essentially perfectly (train 0.99998).

**The fitting phase is visible in the ensemble mean, and spans tens of epochs.** All 50
seeds show Layer 5's I(X;T) rising before it compresses, by a median 1.51 bits, peaking
around epoch 42; Layer 4 rises in 47/50 by a median 1.16 bits, peaking around epoch 46.
The mean rise survives averaging: +1.39 bits in Layer 5, +0.79 in Layer 4. Figure 2 still
plots `runs` rather than `aggregated` — the per-seed scatter is the paper's own figure and
the spread is informative — but averaging no longer destroys the subject, which it did
under the old SGD settings (+0.49 bits, peaking at epoch 1).
