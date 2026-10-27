# infoplane

Interactive replication of the information-plane dynamics in *Opening the Black Box of
Deep Neural Networks via Information* (Shwartz-Ziv & Tishby, 2017). See `PLAN.md` for the
full two-phase design.

**Both phases are implemented.** Phase 1 (the `infoplane` Python package) trains ensembles
of the paper's network, estimates I(X;T) and I(T;Y) for every hidden layer at log-spaced
epochs, and exports the trajectories as JSON. Phase 2 (`frontend/`) is a Vite + D3 site of
four interactive figure pages that read those exports.

## Quick start

```bash
uv sync                  # create .venv from uv.lock
uv run infoplane fetch    # the authors' var_u.mat (1 KB, SHA-256 verified) -> data/
uv run pytest            # 38 unit tests, ~10 s
uv run infoplane verify   # short end-to-end run checking the paper's invariants

# the real thing: 50 seeds x 9000 epochs, ~5 min on 8 cores
uv run infoplane run --num-seeds 50 --num-workers 8 --out data/infoplane.json

# Figure 3: one ensemble per training-set size (the paper's 5/45/85%, plus 15%)
uv run infoplane sweep --train-sizes 205,614,1843,3482 --outdir data/trainsize

# Figure 5: one ensemble per depth, the authors' six architectures (see ARCH_DEFAULT)
uv run infoplane arch --outdir data/arch

# refresh the committed snapshot the site is built from (data/ -> frontend/public/data/)
uv run infoplane publish
```

`infoplane run --help` lists every knob. Useful non-defaults:

| Flag | Why |
| --- | --- |
| `--bins 10 / 120` | The binning estimator is sensitive to this; sweeping it shows how much of the measured compression is bin-dependent. |
| `--mi-on train` | Measure the information plane on the training subset instead of all 4096 patterns. |
| `--split random` | A per-seed random train/test subset — what the paper describes. The default `head` is what the code does: the first 80% of the file, identical for every seed. |
| `--shuffle-batches` | Reshuffle minibatches every epoch. Off by default because the authors cut fixed contiguous batches once and reuse them. |
| `--no-measure-output` | Measure only the hidden layers. By default the readout is measured too, as the paper's highest layer. |
| `--input-encoding spin` | Inputs centred to ±1 instead of the authors' raw 0/1. Trains to slightly better test accuracy but shows a markedly weaker fitting phase — see below. |
| `--optimizer sgd --lr 0.4` | Fits in ~2 epochs. Useful when you only need a converged network (the tests use it); **useless for the figures**, because it collapses the entire epoch axis. |
| `--dataset quadric` | A synthetic deterministic rule, offline and deliberately harder — see below. Not the paper's problem. |

## What the paper says vs what the authors ran

Three sources disagree about this experiment: the paper's text, the released code
([IDNNs](https://github.com/ravidziv/IDNNs)), and the released data (`var_u.mat`).
**Where they conflict, this project follows the code and the file**, on the grounds that
those are what actually produced the figures. Every such choice is listed here. A second
table, ["What the captions say vs what the figures show"](#what-the-captions-say-vs-what-the-figures-show),
covers a different kind of disagreement: the places where a caption describes something
the published panel is not.

| Setting | Paper text | IDNNs code | Here |
| --- | --- | --- | --- |
| Optimizer | "trained using SGD" (§4.1) | `tf.train.AdamOptimizer` (`networks/model.py`) | **Adam** |
| Learning rate | not stated | `0.0004` | **4e-4** |
| Batch size | not stated ("the samples batches") | `512` | **512** |
| Epochs | 9000 (Fig. 2 caption) | `num_epochs` default `8000` | 9000 |
| Training fraction | 85% (§5.1, for Fig. 2) | `inds` default `[80]` | **80%** |
| Training split | "a randomized 85% of the input patterns", "different random selections of the training samples" (§4.2) | `data_shuffle(..., shuffle_data=False)`: the **first** 80% of the file, the same rows for every repeat | **the code's head slice** (`--split random` for the paper's) |
| Minibatches | not stated | `batch_points` cut once outside the epoch loop and reused | **fixed, not reshuffled** |
| Measured layers | "6 points for the different hidden layers... the layer with the smallest IX is the output layer" (Fig. 3 caption) | `Model.hidden_layers` includes the final softmax layer | **5 hidden + the readout** |
| Architecture | "up to 7 fully connected hidden layers, with widths 12-10-7-5-4-3-2" (§4.1) | hidden `[10, 7, 5, 4, 3]`, 12-dim input, 2-way softmax output | **5 hidden layers 10-7-5-4-3** — see below |
| Output layer | tanh "shifted to a sigmoidal function in the final layer" (§4.1) | 2-unit softmax | 1-unit sigmoid, **measured as the softmax pair** `(p, 1-p)` |
| Repeats | 50 randomized initializations (§4.3) | `num_of_repeats` default `1` | 50 |
| Binning | 30 intervals in [-1, 1] (§4.2) | `num_of_bins` default `30` | 30 ✓ |
| Activation | tanh (§4.1) — though §4.2 calls it `arctan` | `tf.nn.tanh` | tanh ✓ |
| Input values | "12 binary inputs" (§4.1) | raw 0/1 | ±1, see the deviations list |
| Labels | stochastic: `p(y=1|x) = σ(γ(f(x) − θ))`, γ "high enough to keep I(X;Y) ≈ 0.99 bits" (§4.1) | — | deterministic, as the file is |

Three of these are worth more than a table row.

**SGD vs Adam decides the epoch axis, which is the figure's entire content.** The paper's
argument is about *when* things happen — a fast ERM phase of "a few hundred epochs" and a
compression phase of thousands — so the horizontal axis of Figures 3-5 and the "400 epochs"
in Figure 2's caption are results, not bookkeeping. They are also pure consequences of the
optimizer settings. Train this network with SGD at lr 0.4 and it fits in ~2 epochs: the
entire ERM phase falls between the first two log-spaced checkpoints and the figures show
nothing but compression. At the authors' Adam/4e-4/512 the phase structure reproduces,
including the 400-epoch snapshot. This is the one setting you cannot substitute a
"reasonable equivalent" for.

There is an irony here worth noticing, since the paper does not: §5.2 explains compression
as a property of *SGD's* gradient noise — a drift phase with high gradient SNR giving way
to a diffusion phase where "the weights change essentially as Weiner processes." Adam
rescales each coordinate by a running estimate of its gradient's second moment, which
changes exactly the noise structure that argument depends on. The phenomenon still appears
under Adam, which is evidence *for* its robustness, but the mechanism offered for it is
not the mechanism that was run.

**The architecture string is ambiguous, and the literal reading is wrong.**
"12-10-7-5-4-3-2" appears in §4.1 as "7 fully connected hidden layers". But Figure 4's
caption writes its network as `input=12-10-8-6-4-2-1=output` — the same seven-number
format, with the first number explicitly the input and the last explicitly the output.
Under that convention "12-10-7-5-4-3-2" means a 12-dim input, **five** hidden layers of
10-7-5-4-3, and a 2-unit output. Four things agree: the code's default
(`select_network_arch('1') -> [10, 7, 5, 4, 3]`), Figure 4's caption, §4.1's own "layer 1
is the hidden layer closest to the input and the highest is **the output layer**" (so the
last entry is not a hidden layer), and Figure 2 itself, which shows about five or six
clusters rather than seven.

This repository originally implemented the literal reading — 7 hidden layers of
12-10-7-5-4-3-2 plus a separate sigmoid output — and it was measurably worse in two
specific ways, both of which are worth knowing about because they are how the mistake
announced itself:

- **Three layers pinned at the ceiling.** Layers 1-3 sat at `I(X;T) = H(X) = 12` bits from
  initialisation onward and never moved, because a 12-wide, a 10-wide and a 7-wide tanh
  layer at 30 bins all resolve the 4096 patterns exactly. The paper's figures have *one*
  such pinned layer. Three immovable trajectories out of seven is a third of the figure
  saying nothing.
- **Seeds got stuck.** The extra 2-unit tanh hidden layer is a hard bottleneck to optimise
  through. Under the literal reading 22 of 50 seeds finished below 0.99 train accuracy and
  8 below 0.95, clustered near 0.92, and training for 30,000 epochs instead of 9,000 barely
  moved it (0.9833 vs 0.9806 mean train accuracy, still 2 of 8 seeds stuck at ~0.93). That
  is a trapped optimisation, not undertraining.

The five-layer reading is now the default, and the paper's own Figure 3 caption settles it
outright: "On each path there are **6 points** for the different hidden layers... The layer
with the smallest IX is the output layer and the one with the highest IX is the first
hidden layer." Six points, one of which is the output, is five hidden layers.

One wrinkle for anyone chasing figure numbers: the arXiv v1 (March 2017, §4/§5 numbering)
and the JMLR-format v3 (May 2017, §3 numbering) differ. Section numbers here follow v1,
which is what the citations above use. Figure 2 (the three snapshots) is the same in both.
v3's Figure 3 caption adds a detail v1 omits — the evolution panels were run with
`input=12-10-8-6-4-2-1=output`, i.e. hidden widths 10-8-6-4-2, not the 10-7-5-4-3 that
§4.1 and the code give as the default. So two different five-hidden-layer networks appear
in the paper's figures. We run the code's.

### What the captions say vs what the figures show

The table above is about the experiment. This is about the *figures*, and it is a
different kind of disagreement: in each case the caption or the surrounding text
describes something the published panel demonstrably is not. All of it was found by
trying to reproduce the panel and failing, then going back to the code that drew it.

| Figure | The caption says | What the figure or its code shows | Evidence |
| --- | --- | --- | --- |
| 4 | "the values are normalized by the L2 norms of the weights for each layer" | nothing is divided | `calc_weights_norms` builds `flatted_list = [1]` and returns 1 for every layer, its real body commented out, and its call site is commented out too |
| 4 | per-layer gradient curves | `plot_gradients.py` draws *cumulative sums over layers*, and takes `sqrt` of values that are already square roots | `y_var = np.sum(cov_traces_all, axis=0)` inside the `for layer` loop, then `axes_log.plot(..., np.sqrt(y_var))` |
| 4 | the dotted curves are `STD(∇W_i)` over minibatches | they cannot be: they open 100–220x *below* the mean, where that ratio has a hard ceiling near 3 on this task | measured, 50 seeds x 6 matrices; see the Figure 4 section |
| 5 | "the width of the hidden layers start with 12, and each additional layer has 2 fewer neurons" | `[10]`, `[10,7]`, `[10,7,5]`, `[10,7,5,4]`, `[10,7,5,4,3]`, `[10,7,5,4,4,3]` | `plot_figures.py` mode 11 — the 2x3 grid titled "One hidden layer" … "Six hidden layers" — is fed `ff3_5 … ff3_1` plus `layerSizes=10,7,5,4,4,3`, and `ff3_N` is the `net_type` that `select_network_arch` maps to the first five |
| 5 | axes in bits, as every other information-plane figure in the paper | axes in **nats** | they stop at 9 and 0.7; H(X) = 12 bits = 8.32 nats and H(Y) = 1 bit = 0.693 nats, which is exactly where its curves top out |
| 5 | "the first hidden layer (of width 12)" (§3.6) | the first hidden layer is 10 wide; 12 is the input | same miscount as §4.1's "12-10-7-5-4-3-2", which the architecture discussion above is about |

Two of these are worth a sentence more.

**Figure 4's dotted curve is the one irreducible discrepancy in this replication.** Every
other gap here is a settings difference that can be stated and moved past. This one is
not: the quantity the paper defines cannot take the value the paper prints. At
initialisation the network answers 0.4908–0.5070 to all 4,096 inputs, so each sample's
gradient is `(p − y)` times nearly the same vector and only the label separates one sample
from another, which bounds `‖mean‖/spread` by `√B·|p̄ − ȳ|/sd(y) = 2.8`. Measured over 50
seeds and 6 matrices the ratio is 0.29 with the authors' contiguous batches and 1.30 with
random ones, against a ceiling of 2.6 median / 10.7 max. The figure needs 100–220.
Reaching that would take 650,000 samples per batch, 158x the whole dataset. What *does*
land on the published curve is a spread taken over the epoch's optimiser **steps** rather
than its minibatches — see the Figure 4 section for the numbers and the caveats.

**Figure 5's caption probably describes a different network than its panels.** "Start with
12, and each additional layer has 2 fewer neurons" is precisely the
`input=12-10-8-6-4-2-1=output` chain of Figure 4's inset and of v3's Figure 3 caption —
the *other* five-hidden-layer network that appears in this paper. The panels were produced
from the code's default family instead. That is a guess about how the caption came to be
written; what is not a guess is which runs the plotting code feeds those six panels.

### Checked line by line against the authors' code

The comparison was redone in full against [IDNNs](https://github.com/ravidziv/IDNNs) and
against an [independent reproduction](https://github.com/cynxchen/deep-learning-mutual-info)
that works from the same code. The estimator matched exactly —
`calc_information_from_mat` computes `H(T) - H(T|X)` and `H(T) - H(T|Y)` over the 4096
patterns with uniform `p(x)`, which is what `mi.py` does — but three behaviours did not,
and all three are now the defaults here:

1. **The readout is a measured layer.** `Model.hidden_layers` appends the final softmax
   layer to the same list `exctract_activity` reads activations from, so every IDNNs
   trajectory includes it, and the paper counts it ("6 points"). It is the most compressed
   point on every path, so leaving it out cut the left end off the figure. We bin it as the
   pair `(p, 1-p)`, matching their 2-way softmax, so the measurement is identical whichever
   readout parameterisation the network was trained with.
2. **The training split is not random.** `data_shuffle` defaults to `shuffle_data=False`
   and takes the first `train_frac` rows. Because `var_u.mat` is ordered by input pattern
   read as an integer, that is not a sample: the training 80% has `p(y=1) = 0.437` and the
   held-out fifth has `0.836`, and the slice is byte-identical for all 50 "randomized"
   networks, which vary only in their weights. This directly contradicts §4.2's "different
   random selections of the training samples". Test accuracy under the authors' split is
   therefore **not** comparable to a random-split number.
3. **Minibatches are cut once and reused.** `batch_points` is built outside the epoch loop,
   so the same contiguous batches recur in the same order every epoch. Reshuffling each
   epoch — what this project did — is the textbook choice but a different noise process,
   and §5.2's whole account of compression rests on the statistics of that noise.

Two further differences are recorded but not adopted:

- **Bin edges.** `np.linspace(-1, 1, 30)` with `np.digitize` gives 29 intervals, not 30;
  `mi.py` uses 30 equal intervals, which is what the paper's text describes. The gap is
  bounded by `log2(30/29) = 0.05` bits and only bites on a saturated layer.
- **Initialisation in the independent reproduction.** It uses
  `stddev = 1/sqrt(fan_out)` where IDNNs uses `1/sqrt(fan_in)`; it also implements the
  literal seven-hidden-layer reading. We follow IDNNs on both.

**The released labels are deterministic; the paper describes them as noisy.** §4.1 defines
the rule as a sigmoid of `f(x) − θ` with gain γ, i.e. genuinely stochastic, and says γ was
set to keep `I(X;Y) ≈ 0.99` bits. The file contains hard 0/1 labels that are exactly a
thresholded quadratic form in the 12 bits (LP-verified — see below), so `I(X;Y) = H(Y) =
0.9992` bits exactly. The paper's 0.99 figure is reproduced, but by class imbalance
(`p(y=1) = 0.5171`) rather than by label noise: γ was evidently taken to the hard-threshold
limit. Anything built on the assumption that these labels are noisy will be wrong.

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

## Phase 2 — the figure pages

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # static site -> frontend/dist (self-contained, ~4.9 MB with data)
npm run preview    # serve the built site
```

**Two data directories, on purpose.** `data/` is the working output of Phase 1 and is
gitignored — it is 21 MB and entirely regenerable. `frontend/public/data/` is a *committed
snapshot* of the subset the pages fetch, written by `uv run infoplane publish`, so a fresh
clone builds and serves the site without running anything. It is 4.5 MB, and the
difference is `runs`: the per-seed trajectories are 85% of the bytes and only Figure 2
plots them, so `publish` keeps them in `infoplane.json` and empties them everywhere else.
`quadric.json` is not published at all, since no page fetches it.

The workflow is `run` / `sweep` / `arch` → `publish` → `npm run build`. `publish` is a pure
file transform over finished exports, so it takes a second and never retrains. Everything
in the snapshot is paid for in repository history permanently (~1.5 MB of git objects per
refresh), which is why it carries the subset rather than the lot.

| Page | What it shows |
| --- | --- |
| `index.html` | Headline numbers, train/test accuracy against epoch, and the caveats below. |
| `fig2.html` | The plane with one mark per (seed, layer) at a scrubbed epoch; hovering traces one seed through all five layers. |
| `fig3.html` | The plane at four training-set fractions as small multiples; all four exports load up front, since the figure is the comparison between them. |
| `fig4.html` | The paper's Figure 4: each weight matrix's mean gradient against its spread, log-log, plus their ratio. The spread is reduced over minibatches (default) or over optimiser steps. |
| `fig5.html` | The paper's Figure 5: the same plane for one to six hidden layers, one panel per depth; the five-layer panel is the main run. |

### Design decisions worth keeping

- **Layer colour is an ordinal ramp, not a categorical palette.** Depth is an ordered
  quantity — the DPI orders I(X;T) by it — so the reader should see the order in the
  colour. One blue hue, 7 steps, monotone in lightness, generated in OKLCH and validated
  in both themes (monotone L, every adjacent gap ≥ 0.06, surface-side end ≥ 2:1 contrast).
  Dark mode is a separately validated set of steps, not an inversion. This holds for
  Figures 3, 4 and 5.
- **Figure 2 is the exception, and it is categorical.** It is the one scatter of layers,
  where any two marks can be neighbours rather than just adjacent steps, and after the
  pinned layers are dropped it carries only four series. The naive four categorical hues
  fail: in a scatter every pair is in play, and the catalogue's first four put yellow next
  to orange. Enumerating all 70 four-slot subsets against `validate_palette.js --pairs all`
  in both themes leaves exactly two that pass; Figure 2 uses slots **1/4/5/6** (blue,
  yellow, magenta, green), worst-pair CVD ΔE 13.0 light / 6.9 dark, normal-vision
  19.6 / 19.3. The dark 6.9 sits in the band that is legal *only* with a secondary channel,
  so each series also has its own **mark shape**, repeated in the legend key; and the two
  light-mode hues below 3:1 on the surface oblige the relief rule, which the direct labels
  on the ensemble means and the table view discharge.
- **Figure 2 omits the layers pinned at the ceiling**, choosing them from the data — any
  series whose `I(X;T)` never falls 0.05 bits below `H(X)` — rather than by name. They
  cannot move, two stacks of 50 immovable marks crowd the layers that do, and the table
  view still lists them. Naming L1 and L2 instead would have silently dropped the wrong
  series the next time the architecture changed, which in this project has happened twice.
- **Figure 4 is two charts, not one chart with two y-axes.** I(X;T) spans 0–12 bits and
  I(T;Y) spans 0–1; aligning two such scales on one plot invents a relationship whose
  slope is an artifact of the alignment.
- **Figure 3 is faceted, not overlaid**, for the same colour reason — the four-hue budget
  is spent on depth, so four overlaid training sizes on top of it could not be coloured
  safely in a scatter. Its marks, hues and heavy mean paths come from `chart.js`, shared
  with Figure 2, so a trajectory means the same thing on both pages.
- **Figure 2 plots `runs`, not `aggregated`**, because Phase 1 measured that averaging
  destroys the fitting phase it exists to show.
- **Direct labels are dropped, never nudged, when they collide.** A label pushed off its
  mark stops pointing at anything; the legend and the table view keep every series
  identifiable.
- Every chart ships a table view, a legend, keyboard access (arrow keys drive the scrubber
  and the crosshair), and tooltips that enhance rather than gate.

### Not built

The plan asked Figure 3 to overlay **theoretical IB bound curves**. Computing one means
solving the information-bottleneck self-consistent equations for this joint distribution,
which the Phase 1 pipeline does not do — so the page draws the limits the data does
establish (I(X;T) ≤ H(X), I(T;Y) ≤ H(Y)) and says the bound is absent, rather than drawing
a guess. Adding it would be a Phase 1 change first.

## Caveats that matter for Phase 2

**Input encoding: the authors' 0/1 is the default, and the deviation was a mistake.** The
encoding is a network *input scaling*, not part of anyone's dataset — labels and patterns
are identical either way, and the two are affinely equivalent to a layer with a bias — but
it changes the dynamics and with them the measurement.

This default was `spin` (±1) for a while, on the evidence that the authors' 0/1 inputs
drove the narrow layers into `tanh` saturation and made the estimator violate the data
processing inequality: 126 of 77,100 measurements, up to +0.381 bits. That evidence was
real but misattributed. It was an artifact of the *architecture* bug described above —
seven hidden layers instead of five, including a 12-wide and a 7-wide layer that saturate.
On the authors' five-layer network both encodings are clean:

| `--input-encoding` | DPI violations | final test acc | L3 / L4 / L5 final I(X;T) | seeds whose L5 rises before compressing |
| --- | --- | --- | --- | --- |
| **binary** (default, the authors') | **0 / 64,250** | 0.9665 | 11.64 / 8.66 / 2.49 bits | **50 / 50**, median +1.51 bits |
| spin | 0 / 64,250 | 0.9731 | 10.77 / 4.40 / 1.12 bits | 44 / 50, median +0.55 bits |

`spin` reaches about a point higher test accuracy and compresses considerably further, and
on that basis it looked like the better setting. It is not: compression magnitude is the
thing being replicated, so preferring our own variant because it shows more of the
phenomenon is exactly backwards. And the fitting phase — the paper's actual subject — is
*clearer* under the authors' encoding, not weaker. Never compare runs made with different
encodings.

**The fitting phase is visible in the ensemble mean.** Under the defaults, every one of the
50 seeds shows Layer 5's `I(X;T)` rising before it compresses, by a median of 1.51 bits,
with the median seed peaking at epoch 88; Layer 4 rises in 49 of 50 by a median 1.26 bits.
Because the seeds peak at similar epochs the rise survives averaging — the ensemble mean
rises +1.28 bits in Layer 5 (peak at epoch 85) and +0.68 in Layer 4 — so the characteristic
right-then-left arc is in `aggregated` as well as in `runs`.

That turn is still about four times too early. The paper's Figure 4 marks the
drift-to-diffusion transition at **~350 epochs** and §4.4 calls the ERM phase "a few
hundred epochs"; ours is at ~85. The epoch axis is a pure function of the optimiser
settings — SGD at lr 0.04 puts the turn at 381 — so matching it would mean calibrating an
unstated hyperparameter against the paper's reported answer. The authors' code says Adam at
4e-4, so that is what runs here, and the gap is recorded rather than tuned away.

This was not true of the earlier configuration, and the difference is worth keeping in
mind when reading older notes: with SGD at lr 0.4 the whole arc was over within two epochs
and largely cancelled in the mean (+0.49 bits). Figure 2 still plots `runs` rather than
`aggregated`, because the per-seed spread is itself informative and the paper's Figure 2
is explicitly a scatter of 50 networks — but it is no longer the case that averaging
destroys the subject.

**The widest layer cannot show a fitting phase at all — and the paper says so.** Its
Figure 3 caption describes "the one with the highest IX" as "the first hidden layer that
hardly moved in the plane during the optimization". Layer 1 sits pinned at
`I(X;T) = H(X) = 12` bits from initialisation onward: 30 bins across 10 `tanh` units
resolve all 4096 patterns, so the measurement is saturated at its ceiling and has nowhere
to rise. Layer 2 (7 units) is nearly as pinned, moving 0.02 bits over the whole run. This
is a property of the estimator, not of the network, and the paper's own figures show the
same thing — one or two trajectories parked at the right-hand edge. It is also why the
seven-layer misreading was so costly: it produced *three* immovable trajectories out of
seven.

Both coordinates are pinned, and the second follows from the first: a layer whose binned
state separates all 4096 patterns is an invertible function of `X`, so `I(T;Y)` must equal
`I(X;Y) = H(Y)` exactly, for any weights at all. Our Layer 1 therefore reads
`(12.00, 0.999)` at epoch 0, before a single gradient step. Any figure that shows the first
layer rising in `I(T;Y)` during training is showing a layer whose binning does *not* fully
separate the inputs — so if a trajectory starts below the ceiling, something in the
measurement differs, not in the dynamics.

**The absolute bit values are bin-dependent; only the shape is robust.** Re-measuring one
`var_u` seed (seed 0) under the defaults at several bin counts moves the mid layers a long
way, while leaving the *shape* of every trajectory intact:

| `--bins` | L1 final | L2 final | L3 final | L4 final | L5 final | Output final | DPI violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 11.93 bits | 11.43 | 8.84 | 5.68 | 2.14 | 1.05 | 0 / 1,285 |
| 15 | 11.98 | 11.86 | 10.43 | 6.83 | 2.41 | 1.18 | 0 / 1,285 |
| **30** (default, the paper's) | 12.00 | 11.99 | 11.80 | 8.87 | 2.97 | 1.18 | 0 / 1,285 |
| 60 | 12.00 | 12.00 | 11.99 | 10.38 | 3.52 | 1.29 | 0 / 1,285 |
| 120 | 12.00 | 12.00 | 12.00 | 11.23 | 4.05 | 1.38 | 1 / 1,285 |

Layer 3 moves 3 bits between 10 and 120 bins (8.84 → 12.00) and Layer 4 doubles
(5.68 → 11.23); Layer 5 moves far less (2.14 → 4.05), and the readout least of all. So "Layer 4 compresses to 7.8 bits" is a property of the network *measured
with 30 bins*, not of the network — the standard objection to the binning estimator
(Saxe et al., 2018). Note also that which layers read as "pinned at the ceiling" is itself
a bin-count artifact: at 10 bins no layer reaches 12, at 120 bins three do. The frontend
states the bin count wherever it puts bits on an axis; `metadata.bins` is in every export
for that. Never compare runs made with different `--bins`.

**Current numbers under the defaults** (`var_u`, 50 seeds × 9000 epochs, binary inputs,
30 bins, the authors' head split, ~5 min on 8 cores / 34 CPU-min): train accuracy 0.9984,
test 0.9665, **0 DPI violations in 64,250 measurements**. Train accuracy crosses 0.95 at
epoch 832 and 0.99 at epoch 3342. Ensemble-mean `I(X;T)` compression from peak to epoch
9000, by series: 0.00, 0.02, 0.05, 1.25, 5.09, 2.21 bits. Figure 3 sweep, mean final test
accuracy at the paper's fractions: 0.8610 (5%, N=205), 0.9000 (15%, N=614), 0.9463
(45%, N=1,843), 0.9716 (85%, N=3,482).

Test accuracy is **not** comparable to a random-split replication: the authors' head slice
holds out the top fifth of the pattern ordering, which is 84% positive.

Against the paper's Figure 2 snapshots, the ensemble means (`I(X;T)`, `I(T;Y)`) land at:

| | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Output |
| --- | --- | --- | --- | --- | --- | --- |
| epoch 400 (paper's center panel) | 12.00, 1.00 | 12.00, 1.00 | 11.66, 0.97 | 9.43, 0.86 | 5.85, 0.77 | 2.70, 0.69 |
| epoch 9000 (paper's right panel) | 12.00, 1.00 | 11.98, 1.00 | 11.64, 1.00 | 8.66, 0.99 | 2.49, 0.98 | 1.10, 0.95 |

The qualitative shape matches the paper's panels — a spread down the `I(X;T)` axis at 400
with the deepest series still short of `I(T;Y) = 1`, then everything at `I(T;Y) ≈ 1` and
pushed left by 9000 — but matching one snapshot is not the same as matching the dynamics,
and ours arrives at that spread having already turned ~250 epochs earlier. Nor are the
endpoints the same: reading the paper's Figure 3 (85% panel, its own 10-8-6-4-2 net) gives
roughly 12 / 10.5 / 3.5 / 1.6 / 1.0 bits, where our middle layers hold 11.6 and 8.7. That
gap survives every setting tried here, including running that architecture — it is the
open discrepancy of this replication, not a resolved one.

**The data processing inequality is the measurement's own smoke alarm.** `I(X;T₁) ≥ … ≥
I(X;T₅)` must hold at every checkpoint. The plug-in binned estimator can invert it even when
the true quantities are ordered, because binning layer i+1 is not a coarsening of binning
layer i: when a layer saturates and its bin grid merges patterns that the next layer's
affine map pulls back apart, the deeper layer's entropy comes out higher. The size of that
effect tracks how badly conditioned the measurement is, which makes it a useful alarm
rather than a nuisance.

On the authors' architecture it is near-silent. **0 violations in 64,250 measurements** on
the full dataset, under both input encodings, and 0 at three of the four training-set sizes
in the Figure 3 sweep. The exceptions are worth stating precisely. At `--train-size 3482`
one seed inverts by +0.0035 bits at epoch 9000 on the **Layer 1 → Layer 2** pair, which is
two saturated layers sitting inside one bin of each other. The larger one arrived with the
readout series: at `--train-size 500` (the earlier sweep), one seed inverted by +0.16 bits
at epoch 2193, on the **Layer 5 → Output** pair. That pair is the one place where the estimator's
assumption is structurally weakest — the readout's probabilities live in `[0, 1]` while the
bin grid spans `[-1, 1]`, so binning the output is not remotely a coarsening of binning a
3-unit tanh layer. One inversion in 64,250 is an artifact; a *rate* of them on that pair
would mean the output quantisation needs its own treatment.

The runs it fired on in bulk were all made with the discarded seven-layer reading, whose
12-wide, 7-wide and 2-wide layers saturate:

| configuration | violations | worst |
| --- | --- | --- |
| 5 hidden layers (the authors'), binary or spin | 0 / 64,250 | 0.000 bits |
| 5 hidden layers, `--train-size 500` | 1 / 64,250 | +0.160 bits (L5 → Output) |
| 5 hidden layers, `--train-size 3482` | 1 / 64,250 | +0.004 bits (L1 → L2) |
| 7 hidden layers, spin, full dataset | 4 / 77,100 | +0.068 bits |
| 7 hidden layers, spin, `--train-size 500` | 43 / 77,100 | **+0.412 bits** |
| 7 hidden layers, binary, full dataset | 126 / 77,100 | +0.381 bits |
| 7 hidden layers, binary, small `--train-size` | — | up to +1.44 bits |

The lesson is about diagnosis rather than about binning. A violation rate says the
measurement is badly conditioned; it does not say *which* knob is responsible. Here it was
blamed on the input encoding, since switching the encoding made it go away, and the real
cause was two surplus layers. Changing a knob until a symptom disappears is not the same as
finding what caused it.

`verify` checks a tolerance (0.1 bits) **and** a violation rate (0.01%), and every export
carries `dpi_violations` / `dpi_measurements`. Note that a single L5 → Output inversion of
the size seen above would trip the *magnitude* half of that check on its own. The
thresholds have deliberately not been widened to accommodate it: if `verify` ever fails
there, the thing to look at is the output-layer quantisation, named above, and not a
loosened tolerance. **Check that rate before trusting a run**: a
short `verify` run is not representative, since violations appear well into training.

**Figure 4 measures the gradients, not the information.** At every checkpoint epoch the
pipeline also records, per weight *matrix* (six of them: `input->L1` … `L5->Output`), the
norm of the mean minibatch gradient, the batch-to-batch spread `√(E‖g − E[g]‖²)`, and the
matrix's own L2 norm. The gradients are read inside the batch loop before `opt.step()`, so
they are the gradients the optimiser actually applied. A **second** reduction of the same
two statistics is recorded alongside — over the epoch's optimiser *steps* rather than its
minibatches (`step_mean`/`step_std`; see "the spread the paper prints" below). This
follows IDNNs `plots/plot_gradients.py` and `networks/network.py`, with four differences,
all checked:

- **They read the gradient *after* the step.** Their batch loop runs
  `sess.run([optimizer, accuracy], feed)` and only then `sess.run([grads, ...], feed)` on
  the same batch, so every gradient they save is taken at the weights that batch has
  already been stepped to. Measured both ways here (scratch experiment, 4 seeds, 2,000
  epochs, Adam 4e-4 and SGD 0.4): it moves the SNR by under 30% at every epoch and changes
  no conclusion. We read before the step, which is the quantity §5.2 argues about.
- Their plotting loop **accumulates**: `y_var = np.sum(cov_traces_all, axis=0)` sits inside
  the `for layer` loop, so each curve they draw is the running total over all layers up to
  that one, and it then plots `sqrt` of values that are already square roots. Partial sums
  are nested by construction, and the published figure's curves are not nested — they lie
  on top of each other for 100 epochs and then separate — so the figure was produced by
  some other version of that file. We store per-layer values.
- **The caption says normalized and the figure is not.** Figure 4's caption reads
  "normalized by the L2 norms of the weights for each layer", but `calc_weights_norms`
  builds `flatted_list = [1]`, returns 1 for every layer, and its call site is commented
  out. Both forms are exported (`mean`/`std` and `mean_norm`/`std_norm`, the latter
  divided **per seed**, since ‖g‖ and ‖W‖ are correlated across seeds and a ratio of
  medians is not a median of ratios), and `fig4.html` plots the undivided ones with a
  toggle. Dividing is what breaks the figure here: ‖W‖ for `L5->Output` starts four times
  smaller than `input->L1`'s — 3 weights against 120 — and grows 14× as that layer
  converges, so the ratio lifts its curve above the others early and crushes it below them
  late. Undivided, the six curves start coincident and end ordered by depth, which is what
  the published figure shows.
- **Their curves are smoothed and ours were not.** The published curves have no
  checkpoint-to-checkpoint noise at all; ‖E[g]‖ under Adam swings by tens of percent
  between adjacent log-spaced checkpoints, and **a median over seeds does not remove it**:
  every seed steps through the same fixed contiguous minibatches in the same order, so
  they swing together. Measured (16 seeds, scatter about a running geometric mean):
  one seed jitters 32% for `input->L1`, the 16-seed median still jitters 23%, where
  seed-independent noise would have left 8% — two thirds of it is common-mode. The deepest
  two matrices are the other way round (13% → 4%, independent prediction 3%). `fig4.js`
  draws a geometric mean over a ±0.07-decade window of epochs on top of the measured
  curve, which is left visible underneath. `plot_gradients.py` imports `statsmodels` and
  never uses it.

**The spread the paper prints is not a spread between minibatches.** This is the one
discrepancy in the whole replication that settings cannot reach. At epoch 1 the published
figure's six mean curves sit at 0.078–0.080 and its six dotted curves at 7.8e-4 down to
3.5e-4 — the mean gradient is 100 to 220 times *larger* than the spread. Running the
figure's own network (`12-10-8-6-4-2` with the authors' two-way softmax) gets the shape of
both families right (medians over 20 seeds at epoch 0): the mean is flat across the six
matrices at 0.024, 0.022, 0.020, 0.016, 0.021, 0.019, as the published ones are flat to
3%; the spread falls with depth 0.077 → 0.033, a factor 2.3 against the published 2.2.
What is wrong is the gap between them — our ratio is 0.33–0.49 against their 100–220.
Compare the ratio, not the two curves separately: the gradient norm spans more than a
decade across initialisations, so the published 0.078 is an ordinary draw from our
ensemble, while the ratio barely moves. Nothing closes the ratio:

- Cutting batches at random rather than contiguously lifts ‖mean‖/spread at init from a
  median 0.29 to 1.30 (50 seeds x 6 matrices).
- The *ceiling* — what the ratio would be if all the batch-to-batch disagreement lay along
  the mean direction and none of it anywhere else — has median 2.6 and maximum 10.7 over
  the same 300 cases.
- The ceiling is a property of the labels, not the network. At init the network answers
  0.4908–0.5070 to all 4,096 inputs, so each sample's gradient is `(p − y)` times nearly
  the same vector and only the label distinguishes one sample from another:
  `√B·|p̄ − ȳ|/sd(y) = √512 × 0.062/0.496 = 2.8`. Reaching 100 would need 650,000 samples
  per batch, 158x the whole dataset. No architecture, initialiser, optimiser or learning
  rate moves it.

What *does* land on the published curve is a different collection: the whole training
set's gradient re-evaluated at each weight state the epoch passes through — same data,
only the weights moving, no minibatch noise in it at all. At epoch 1, median over 50
seeds:

| per weight matrix | published | spread between minibatches | spread between steps |
| --- | --- | --- | --- |
| `input->L1` | 7.8e-4 | 4.5e-2 | 3.6e-4 |
| `L5->Output` | 3.5e-4 | 2.9e-2 | 2.6e-4 |
| ‖mean‖ / spread | 100–220 | 0.29–0.35 | 35–43 |

It keeps matching: the six ratios peak at 77–432 where the published curves open 100–220
apart, and the spread climbs from 1.3e-4 to 7.3e-2, a factor 560 where the published curve
climbs about a thousand. The transition is late rather than missing — crossings at epochs
797 to 1,532 against the paper's single line at ~350, with `L5->Output` never falling
through 1. It also explains the transition without any appeal to SGD noise: Adam's step is
scale-free, so early on seven steps barely disturb a large gradient while late on they
swing a small one further than its own size. This is **not** evidence that it is what the
authors ran — their released loop feeds each batch's own rows to `tf.gradients` — only
that it is the sole reduction tried here that reproduces the published numbers.
`fig4.html` carries both behind a control, defaulting to the minibatch one the paper
defines.

What it shows here (smoothed curves, median over 50 seeds): the SNR peaks at epochs 115,
127 and 130 for the three deepest matrices and falls back through 1 at 142, 240 and 832,
while the three nearest the input never exceed 1 at all — 0.31, 0.32, 0.37 at their peaks.
The paper marks a single transition for the whole network at ~350 epochs. The earlier peak
is the same optimiser-driven epoch-axis shift as everywhere else in this replication; the
*spread* of the transition across layers is not something the paper's single line allows
for. Two further gaps stay open and are stated on the page rather than tuned away: at the
untrained network the SNR is already 0.29–0.33 for every matrix, where the paper opens two
orders of magnitude above 1 (SGD at 0.4 and the authors' after-the-step measurement were
both tried; neither lifts it above 1, and the ceiling above says why), and only the three
shallowest matrices' spread
rises over training (27x, 17x, 7x, peaking near epoch 1,500) while the three deepest peak
inside 70 epochs and then fall one to two orders of magnitude as the network converges.

**Figure 5 asks what the hidden layers buy.** `infoplane arch` trains one ensemble per
depth, 1 to 6 hidden layers, 50 seeds each on the same 80% head split as the main run.

*The architectures* are `[10]`, `[10,7]`, `[10,7,5]`, `[10,7,5,4]`, `[10,7,5,4,3]`,
`[10,7,5,4,4,3]` — the authors' own, read off their plotting code. The caption describes
something else ("the width of the hidden layers start with 12, and each additional layer
has 2 fewer neurons"), and §3.6 adds "the first hidden layer (of width 12)", but
`plots/plot_figures.py` builds a 2x3 grid titled "One hidden layer" … "Six hidden layers"
and feeds it `ff3_5`, `ff3_4`, `ff3_3`, `ff3_2`, `ff3_1` and a run named
`layerSizes=10,7,5,4,4,3`. The `ff3_N` suffix is the `net_type` argument, and
`select_network_arch` maps 5, 4, 3, 2, 1 to the first five of the list above — one to five
hidden layers in panel order, with the sixth spelled out in the path. The widths grow from
the input end, the decrements are 3, 2, 1, 1 rather than 2, and the 12 in the caption is
the *input*, counted as a hidden layer exactly as §4.1 does when it writes the network as
"12-10-7-5-4-3-2". Following the caption instead was tried first and is wrong: it makes
the five-layer panel 12-10-8-6-4, comparable to nothing else here. **The five-layer member
is `PAPER_HIDDEN_WIDTHS`**, so that panel is the main run's architecture on the main run's
split and should reproduce Figure 2's ensemble means — which is what makes it a check on
the pipeline rather than a new measurement.

*The paper's Figure 5 is plotted in nats, not bits.* Its axes stop at 9 and 0.7 where
every other information-plane figure in the paper stops at 12 and 1. Those are the same
two bounds in natural logarithms — H(X) = 12 bits = 8.32 nats, H(Y) = 1 bit = 0.693 nats —
and its curves top out exactly there. Divide the paper's Figure 5 numbers by ln 2 before
comparing them with anything here.

What it shows (50 seeds, 9,000 epochs; the → columns are the first epoch at which the
readout's mean I(T;Y) reaches that share of H(Y)):

| hidden | widths | test acc | → 80% | → 90% | → 95% | readout I(T;Y) | DPI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.9684 | 2,541 | 4,126 | 7,765 | 0.953 | 0/12,850 |
| 2 | 10-7 | 0.9781 | 1,350 | 2,541 | 4,040 | 0.970 | 0/25,700 |
| 3 | 10-7-5 | 0.9733 | 1,049 | 2,102 | 4,126 | 0.961 | 23/38,550 |
| 4 | 10-7-5-4 | 0.9656 | 1,071 | 2,436 | never | 0.947 | 1/51,400 |
| 5 | 10-7-5-4-3 | 0.9665 | 964 | 2,436 | never | 0.947 | 0/64,250 |
| 6 | 10-7-5-4-4-3 | 0.9549 | 1,005 | 3,072 | never | 0.930 | 196/77,100 |

- **The five-layer panel is bit-identical to the main run**, as it must be: same
  architecture, split, seeds and settings. Zero difference at every checkpoint on all six
  series. That is the check that the family above is the right one — and it is why that
  panel is worth comparing with Figure 3's 85% panel, which is the same architecture on a
  random 85% split: the shallow layers agree to 0.2 bits and the deep ones compress
  further under the random split (Layer 4 reaches 6.37 bits there against 8.66 here),
  which is the head slice's class imbalance, not the depth.
- **Depth buys speed, and nearly all of it is the second hidden layer.** At the 80% bar:
  2,541 epochs with one hidden layer, 1,350 with two, and then flat — 1,049, 1,071, 964,
  1,005. The second layer is worth 1.9x; the four after it are worth 1.3x between them,
  and not monotonically.
- **Past two layers, depth costs.** Test accuracy peaks at two hidden layers (0.9781) and
  falls to 0.9549 at six; the readout's final I(T;Y) falls 0.970 → 0.930. At the 95% bar
  the three deepest never arrive at all. The paper's claim does not reproduce in its
  strong form: a single 10-wide hidden layer is not "unable to achieve good I_Y values
  even after 10⁴ epochs" — it reaches 0.953 of 0.999 bits — and no architecture reaches
  the bar "within 400 epochs".
- **The fitting phase does move earlier with depth.** The readout's I(X;T) peak lands at
  epochs 534, 255, 179, 139, 117, 103 for 1 to 6 layers. That 5.2x is the real "deeper is
  faster" effect in these runs.
- **Only the deepest layers move.** Counting the deepest hidden layers that leave the
  I(X;T) ceiling: 0, 1, 1, 2, 3, 4 across the six panels. A width is not pinned or
  unpinned in itself — the 5-wide layer compresses 6.32 bits when it is the last hidden
  layer, 0.34 with one behind it, and 0.02 with three. The 10-wide first layer never moves
  in any panel, which is the paper's own point 4, except that the paper expects it to
  compress eventually and in 9,000 epochs it does not.
- **The six-layer run is the one measurement here with a problem.** Its architecture is
  10-7-5-4-**4**-3 — two consecutive 4-wide layers — and between exactly that pair the
  estimator inverts the DPI in 4 of 50 seeds, 186 times out of 77,100, by up to 0.39 bits.
  I(X;T) cannot rise with depth, so that is estimator error: binning layer *i+1* is not a
  coarsening of binning layer *i*, and when the map between two layers of the same width
  is close to invertible, the 30-bin quantisation swamps the information the layer
  actually loses. It is the only equal-width consecutive pair in the sweep and the only
  violation above a hairline; the rest are 1-23 inversions under 0.01 bits between two
  layers sitting at 11.99 bits, the usual saturated pair. The rate (2.5e-3) is 25x the
  `verify` tolerance, which is working as intended — it is telling the truth about an
  architecture the estimator handles badly, and nothing has been tuned to silence it.

## Export schema (the Phase 1 → Phase 2 contract)

```
{
  "weight_layers": ["input->L1", ... "L5->Output"],   # one per weight MATRIX
  "gradients":   { "<weight layer>": { mean, std, mean_norm, std_norm,   # per seed /||W||
                                       step_mean, step_std,           # over steps, not
                                       step_mean_norm, step_std_norm, #   minibatches
                                       mean_sd, std_sd, weight_norm } },
  "metadata":    { every TrainConfig field, plus dataset_name, n_samples, seeds,
                   checkpoints, generated_utc, infoplane_version },
  "bounds":      { "H_X": 12.0, "H_Y": 0.9992 },     # axis ceilings for the frontend
  "epochs":      [0, 1, 2, ...],                      # the checkpoint epochs
  "layers":      ["Layer 1", ..., "Layer 5"],   # one per hidden layer
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
and `arch` each write one such file per panel plus an `index.json` — `{"sizes": [...]}`
keyed by `train_size` for Figure 3, `{"architectures": [...]}` keyed by `n_hidden` with
the `widths` of each for Figure 5. Prefer additive changes.

## Deliberate deviations from `PLAN.md`

1. **The dataset is the authors' file, not a generator.** `PLAN.md` called for a synthetic
   12-bit generator cross-verified against the original; the original turned out to be both
   available and not reproducible from the paper's prose, so `var_u.mat` is the default and
   the synthetic rule is demoted to a contrast case.
2. **The optimiser settings, architecture and input encoding are the authors' code's, not
   the paper's prose**, wherever the two disagree — see the comparison section above. This
   repository briefly deviated on two of them (seven hidden layers; ±1 inputs) and both
   deviations turned out to be mistakes that were hiding the phenomenon rather than
   protecting it. `--input-encoding spin` restores the second one if you want to compare.
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
7. **The training sizes are separate files** with an identical schema, rather than a
   new axis inside one document, so the frontend can lazy-load only the sizes on screen.
8. **No MNIST.** `PLAN.md` asked for the dataset interface to support alternatives, and a
   subsampled MNIST was wired up to demonstrate it. It has been removed: it was never part
   of the replication, had no test covering it, downloaded 10 MB on first use, and was the
   only reason the project depended on `torchvision` — a second large, torch-version-bound
   dependency bought for a path nobody ran. `--dataset quadric` already demonstrates the
   same thing offline, and `--dataset matfile` takes any `.mat` in the authors' layout.
   The interface itself is unchanged, so adding a dataset is still one function and one
   `build_dataset` branch.
9. **No Arrow/Feather fallback.** `PLAN.md` specified one for exports over ~50 MB. It was
   implemented and has now been removed: the largest file this project writes is 2 MB, two
   orders of magnitude inside the budget, so the path never ran. It was also not a drop-in
   — it carried the per-seed `IXT`/`ITY` rows and the metadata, but not `aggregated`,
   `gradients` or `curves`, so every page except Figure 2 would have broken on a file
   written that way. A fallback that is never exercised and would not work is worse than
   none. If the ensemble ever does outgrow JSON, the honest fix is fewer seeds or fewer
   checkpoints; the writer prints a warning at the budget.

## License

[MIT](LICENSE). The paper this replicates is Shwartz-Ziv & Tishby (2017),
[arXiv:1703.00810](https://arxiv.org/abs/1703.00810); the authors' own code is
[IDNNs](https://github.com/ravidziv/IDNNs), and `data/var_u.mat` is fetched from it at
run time rather than vendored here. Neither is covered by the licence above.
