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
list; the four that bite are the optimiser (text says SGD, code runs Adam at 4e-4), the
architecture (see below), the labels (text says stochastic, file is deterministic), and the
train/test split (text says randomized, code takes the first 80% of the file unshuffled).
Reading the paper alone will lead you to re-introduce bugs that have already been fixed
here. The code was re-checked line by line on 2026-10-26 against IDNNs and against an
independent reproduction; README's "Checked line by line against the authors' code" has
what came out of it.

## Commands

```bash
uv sync                 # create .venv from uv.lock (Python 3.12)
uv run infoplane fetch  # the authors' var_u.mat -> data/ (needed by --dataset var_u)
uv run pytest           # 38 unit tests, ~10 s
uv run pytest tests/test_mi.py::test_mi_of_independent_variables_is_zero_for_a_product_distribution
uv run infoplane verify # ~11 s end-to-end run asserting the paper's invariants
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json   # ~5 min
uv run infoplane sweep --train-sizes 205,614,1843,3482 --outdir data/trainsize  # Figure 3

cd frontend && npm install && npm run dev     # the figure pages (needs the data above)
cd frontend && npm run build                  # static site -> frontend/dist
```

No linter or formatter is configured. `data/` is gitignored — it is all regenerable.

## Architecture

The pipeline is a straight line, with each module independently testable:

`datasets.py` (enumerate X, Y) → `model.py` (the MLP) → `train.py` (train one seed, measure
at checkpoints) → `runner.py` (ensemble over seeds) → `export.py` (JSON contract) →
`cli.py` (argparse surface).

**Two things are measured at each checkpoint**, not one: the information plane, and the
minibatch-gradient statistics behind Figure 4 (mean norm, batch-to-batch spread and weight
norm, per weight *matrix* — six for a five-hidden-layer net). The gradients are read inside
the batch loop, before `opt.step()`, because the quantity is the gradient the optimiser is
about to apply.

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

Four static pages (`index`, `fig2`, `fig3`, `fig4`), one Vite entry each — not an SPA.
The paper's Figure 3 is the training-size comparison, which is what `fig3.html` now shows;
the old "fitting against compression" page was retired when it was relabelled. There is no
Figure 5 page yet — the paper's is a comparison of architectures, which Phase 1 has not
been run for. Shared
modules in `src/scripts/`: `chart.js` (frames, axes, legend, tooltip, table view),
`data.js` (fetch + reshape the export into typed arrays), `scrubber.js`, `theme.js`.
`public/data` is a symlink to the repo's `data/`, dereferenced into `dist` at build.

Rules that are load-bearing, not taste — a change here needs the same validation:

- **Layer colour is an ORDINAL ramp** (one blue hue, 7 steps, monotone lightness),
  generated in OKLCH and validated per theme, on `fig4`. Never substitute a
  rainbow palette: depth is ordered. Series that are *not* layers (train vs test) use
  `.c1`/`.c2`, categorical slots 1-2.
- **`fig4` is categorical too, but on a different gate.** It plots LINES, which are
  validated on the *adjacent* pairlist rather than all-pairs, so it uses catalogue slots
  1-6 straight through (worst adjacent CVD ΔE 9.1 light / 8.4 dark). It needs six because
  it draws twelve lines — a mean and a spread per weight matrix — and six steps of one blue
  cannot separate six series that keep crossing. Mean vs spread is line style, not colour.
- **`fig4` plots the gradient norms UNDIVIDED by default**, although Figure 4's caption
  says "normalized by the L2 norms of the weights for each layer". The authors'
  `calc_weights_norms` returns 1 and its call site is commented out, and dividing is what
  destroys the figure: `L5->Output` starts with the smallest ‖W‖ of the six (3 weights)
  and grows it 14x, which lifts its curve above the others early and crushes it late.
  Undivided, the curves start coincident and end ordered by depth, as published. The
  division is a toggle, done per seed in `export.py` (`mean_norm`/`std_norm`) because a
  ratio of medians is not a median of ratios. Do not "fix" this back to the caption.
- **`fig4` draws a smoothed curve over the measured one**, a geometric mean over a
  ±0.07-decade window of epochs, with the measured curve left visible underneath. The
  checkpoint-to-checkpoint swing in ‖E[g]‖ is real and is *not* removed by the median over
  seeds: every seed steps through the same fixed minibatches, so they swing together (one
  seed jitters 32% on `input->L1`, a 16-seed median still 23%, independent noise would
  leave 8%). More seeds is not the fix; a window of epochs is.
- **The information-plane scatters (`fig2`, `fig3`) are categorical, not ordinal.** They
  are the pages where any two marks have to be told apart rather than just neighbours, and
  they plot four series. `PLANE_SERIES`, `planeMark`, `pinnedLayers` and `pinnedLabel` in
  `chart.js` are the single source for both — add a page that plots the plane and it uses
  those, rather than growing a third variant. The hues are catalogue slots **1/4/5/6** (`.c1 .c4 .c5 .c6`) --
  the subset found by enumerating all 70 four-slot combinations against
  `validate_palette.js --pairs all` in both themes, and one of only two that pass. Its
  dark-mode CVD separation is 6.9, inside the band that is legal *only* with a secondary
  channel, so every mark also carries a **shape** (circle / square / triangle / diamond,
  repeated in the legend key). Light-mode yellow and magenta fall below 3:1 on the
  surface, which obliges the relief rule, discharged by the direct labels on the ensemble
  means and the table view. Changing any of this means re-running the enumeration, not
  picking a nicer hue.
- Marks take a `l1`..`l7` class and read `--c` from CSS, so a theme switch repaints
  without a re-render and a layer's colour never follows its rank. A default run has five
  hidden layers plus the measured readout, so it uses `l1`..`l6` and leaves `l7` for a
  deeper `--widths`. Truncating the ramp is safe (every gate is per-adjacent-pair or on the
  surface-side end) and was re-validated as 6 slots, but extending or re-ordering it is
  not — run the dataviz skill's `validate_palette.js --ordinal` in both modes.
- **Never put I(X;T) and I(T;Y) on one pair of axes** (0-12 bits vs 0-1).
- Figure 2 reads `runs`, not `aggregated` — see the expected-results note below.
- **Figure 2 drops the pinned layers from its plot** (not from its table), and finds them
  from the data — any series whose `I(X;T)` never falls 0.05 bits below `H(X)` — rather
  than naming L1/L2, because the architecture has already changed twice here.
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
  is the output. The authors' code agrees (`select_network_arch('1') -> [10,7,5,4,3]`), and
  so does Figure 3's caption: "On each path there are 6 points for the different hidden
  layers... the layer with the smallest IX is the output layer" — six series, one of them
  the output, is five hidden layers. (v3's Figure 3 caption also says *those* panels used
  `12-10-8-6-4-2-1`, a different five-hidden-layer net. We run the code's.)
  **Do not "fix" this back to seven layers.** That reading was implemented and measured: it
  pins three layers at the 12-bit ceiling instead of one, traps 8 of 50 seeds near 0.92
  train accuracy (unchanged by training 30,000 epochs instead of 9,000), and produces every
  DPI violation this project has ever recorded.
- **Three defaults copy IDNNs rather than convention, and all three look like bugs.**
  `split="head"` takes the first 80% of the file, the same rows for every seed (their
  `data_shuffle` defaults to `shuffle_data=False`); that slice is class-imbalanced, 0.437
  positive against 0.836 in the held-out fifth, so test accuracy here is not comparable to
  a random-split number. `shuffle_batches=False` reuses fixed contiguous minibatches every
  epoch, as their `batch_points` does. `measure_output=True` measures the readout as a
  sixth series, because `Model.hidden_layers` includes the final softmax layer and the
  paper's Figure 3 caption says "6 points" per path. `tests/test_train.py` pins all three.
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
  0/1 inputs. `spin` (±1) trains to slightly better test accuracy (0.9731 vs 0.9665) and
  compresses much further (Layer 5 to 1.12 bits vs 2.49), which is exactly why it was the
  default for a while and exactly why it should not be: showing *more* of the phenomenon is
  not a reason to prefer a setting the paper did not use. The fitting phase is also clearer
  under `binary` (50/50 seeds vs 44/50). Never compare runs across encodings.
- **MI magnitudes are bin-dependent.** Layer 4's final I(X;T) reads 6.83 bits at
  `--bins 15` and 11.23 at `--bins 120` (README has the table). Only the
  fitting-then-compression shape is robust, so never quote an absolute bit value without
  the bin count, and never compare runs made with different `--bins`.
- **DPI is a correctness check, not a result.** `I(X;T₁) ≥ … ≥ I(X;T₅)` must hold at every
  checkpoint (`train.dpi_violation`, reported in `diagnostics.max_dpi_violation_bits`). On
  the current defaults it is 0 violations in 64,250 under both encodings, and clean at
  every `--train-size` but one: N=500 inverts once, by +0.16 bits, on the Layer 5 -> Output
  pair. The estimator *can* invert it where a layer saturates (binning layer i+1 is not a
  coarsening of binning layer i), and that pair is its weakest point, since the readout's
  probabilities live in [0,1] while the grid spans [-1,1]. One such inversion is an
  artifact; a rate of them means the output quantisation needs its own treatment. Do not
  widen the tolerances to absorb it. Treat any nonzero rate as a signal that something about
  the run is wrong — but diagnose it rather than tuning knobs until it goes away: these
  violations were once blamed on the input encoding (switching encodings did silence them)
  when the actual cause was two surplus layers. `verify` checks a tolerance plus a
  violation *rate*, not exactness. A short `verify` run is NOT representative: violations
  appear well into training, so check `dpi_violations` on the full run before trusting it.
- **Don't reach for MPS/CUDA.** The networks are tiny; all the speedup is from running
  seeds in separate processes with one torch thread each (`runner._init_worker`).

## Expected results (`var_u`, defaults)

50 seeds x 9000 epochs, ~5 min wall on 8 workers (~34 min of CPU): final train accuracy
0.9984, test 0.9665, and **0 DPI violations in 64,250 measurements**. Train accuracy
crosses 0.95 at epoch 832 and 0.99 at epoch 3342. Ensemble-mean I(X;T) compression from
peak to epoch 9000, by series: 0.00, 0.02, 0.05, 1.25, 5.09, 2.21 bits — the signature is
in Layer 5 and the readout. Layer 1 sits pinned at H(X) = 12 bits (30 bins across 10
`tanh` units resolve all 4096 patterns) and Layer 2 nearly so, moving 0.02 bits over the
whole run; the paper says the same of its own first layer, which "hardly moved in the
plane" (Fig. 3 caption). Test accuracy is depressed by the authors' head split, whose
held-out fifth is 84% positive — do not compare it with a random-split number.

A default run has **six** series: five hidden layers plus the readout, which the paper
counts as the highest layer and IDNNs measures. The readout is the most compressed point
on every path.

The paper's Figure 2 snapshots, as ensemble means of (I(X;T), I(T;Y)):

| epoch | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Output |
| --- | --- | --- | --- | --- | --- | --- |
| 400 | 12.00, 1.00 | 12.00, 1.00 | 11.66, 0.97 | 9.43, 0.86 | 5.85, 0.77 | 2.70, 0.69 |
| 9000 | 12.00, 1.00 | 11.98, 1.00 | 11.64, 1.00 | 8.66, 0.99 | 2.49, 0.98 | 1.10, 0.95 |

**If a change moves these, check the epoch axis before anything else.** It is the most
fragile thing here: it is set entirely by the optimiser settings, and getting it wrong
makes the figures look fine while showing the wrong phenomenon. SGD at lr 0.4 fits this
network in ~2 epochs, which puts the entire ERM phase between the first two log-spaced
checkpoints and leaves nothing but compression on screen.

**Two gaps to the paper are open, measured, and deliberately not tuned away.** The turn
happens at epoch ~85 where Figure 4 marks it at ~350 (the axis is a pure lr knob: SGD at
0.04 lands on 381, which would be calibrating against the paper's answer rather than
replicating it). And the middle layers compress far less than the paper's: it reports
roughly 12 / 10.5 / 3.5 / 1.6 / 1.0 bits for the 85% panel, we get 11.64 / 8.66 / 2.49.
Neither optimiser, nor split, nor batching, nor running the figure's own 10-8-6-4-2
architecture closes the second one. Do not start turning knobs at it.

Figure 3 sweep, at the paper's training fractions, mean final test accuracy: 0.8610
(5%, N=205), 0.9000 (15%, N=614), 0.9463 (45%, N=1,843), 0.9716 (85%, N=3,482). The deep
layers' final I(T;Y) is what separates them -- 0.45 at 5% against 0.96 at 85% -- which is
the paper's point: compression happens in all four, but what it costs depends on the
sample. Three panels are DPI-clean; 85% has one +0.004-bit L1 -> L2 inversion (two
saturated layers within a bin of each other).

`--dataset quadric` for contrast reaches test 0.9890 with compression 0.01, 0.24, 0.81,
3.72, 5.23, 1.89 — more in the deep layers than `var_u`, being a deterministic rule the
network fits perfectly (train 1.0000).

**The fitting phase is visible in the ensemble mean, and spans tens of epochs.** All 50
seeds show Layer 5's I(X;T) rising before it compresses, by a median 1.51 bits, with the
median seed peaking at epoch 88; Layer 4 rises in 49/50 by a median 1.26 bits. The rise
survives averaging: +1.28 bits in Layer 5 (peak epoch 85), +0.68 in Layer 4. Figure 2 still
plots `runs` rather than `aggregated` — the per-seed scatter is the paper's own figure and
the spread is informative — but averaging no longer destroys the subject, which it did
under the old SGD settings (+0.49 bits, peaking at epoch 1).
