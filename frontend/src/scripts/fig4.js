/** Figure 4 - the minibatch gradients of each weight matrix, over training.
 *
 * The paper's Figure 4 is not about information at all: it plots, per layer and
 * per epoch, the norm of the MEAN minibatch gradient against the norm of their
 * spread, on log-log axes, and reads a phase transition off the point where the
 * two cross. Before it the gradient has a large mean and small variance - SGD is
 * drifting downhill. After it the mean is swamped by batch-to-batch noise and
 * the weights diffuse under the error constraint, which is the mechanism §5.2
 * offers for the compression phase.
 *
 * Six curves, not five: one per weight MATRIX, so (input,L1), (L1,L2) ... and
 * (L5,output). They take CATEGORICAL hues, catalogue slots 1-6, rather than the
 * ordinal depth ramp: twelve lines are on screen at once here, a mean and a
 * spread for each matrix, and six steps of one blue cannot separate six series
 * that cross each other repeatedly. Lines are gated on the ADJACENT pairlist
 * rather than all-pairs, so the catalogue order is legal this far -- see
 * main.css. Mean against spread is then a line-STYLE distinction, solid against
 * dashed, exactly as the paper draws it.
 *
 * TWO THINGS HERE ARE NOT WHAT THE CAPTION SAYS, both on purpose.
 *
 * 1. The curves are NOT divided by the weight norms by default, though Figure
 *    4's caption says "the values are normalized by the L2 norms of the weights
 *    for each layer". The authors' released code does not divide:
 *    `calc_weights_norms` builds `flatted_list = [1]` and returns 1 for every
 *    layer, and even that is commented out at the call site. Dividing is also
 *    what breaks the figure: ||W|| for (L5,output) starts four times smaller
 *    than ||W|| for (input,L1) -- three weights against a hundred and twenty --
 *    and then grows fourteenfold, so the ratio lifts that one curve above all
 *    the others early and crushes it below them late. Undivided, the six curves
 *    start on top of each other and end ordered by depth, which is what the
 *    published figure shows. The toggle puts the division back.
 * 2. The spread can be reduced over the epoch's optimiser STEPS instead of its
 *    minibatches. That second quantity holds the data fixed -- it is the whole
 *    training set's gradient, re-evaluated at each weight state the epoch passes
 *    through -- so it contains no minibatch noise at all, only the gradient
 *    moving because the weights moved. It is here because the minibatch spread
 *    the paper DEFINES cannot produce the figure the paper PRINTS: that figure
 *    opens with ‖mean‖/spread near 100, and for this task that ratio has a hard
 *    ceiling near 3. Whichever batches you cut, each one carries its own share
 *    of the two labels, and that alone puts a floor under the disagreement
 *    between batches. The step spread has no such floor, starts ~100x under the
 *    mean, climbs three orders of magnitude, and crosses -- which is the
 *    published shape. See the notes on fig4.html.
 * 3. A smoothed curve is drawn over the measured one (faint, behind). The
 *    published figure has no checkpoint-to-checkpoint noise at all, and the
 *    quantity it plots does: with Adam, ||E[g]|| swings by 50% between adjacent
 *    log-spaced checkpoints late in training, and a median over 50 seeds does
 *    not remove it, because every seed is stepping through the same fixed
 *    minibatches and swings with the others. Averaging over a WINDOW OF EPOCHS
 *    does remove it. `plot_gradients.py` imports statsmodels and never uses it,
 *    which is where a lowess smoother would have gone.
 */
import { line } from "d3-shape";
import { scaleLog } from "d3-scale";
import { load, provenance } from "./data.js";
import { initTheme } from "./theme.js";
import { frame, gridY, legend, responsive, tableView, xAxis, yAxis } from "./chart.js";

// Catalogue order, one slot per weight matrix. A seventh connection would need
// the palette re-validated, not a seventh hue invented.
const CLS = (k) => `c${k + 1}`;

// Half-width of the smoothing window, in decades of epoch. The checkpoints are
// log-spaced 0.0155 decades apart, so this averages over nine of them.
const SMOOTH_DECADES = 0.07;

// Only powers of ten get a label on a log axis; anything else is noise at this
// density, and d3's default format prints "2e-3" style labels between them.
const decade = (v) => {
  const e = Math.log10(v);
  if (Math.abs(e - Math.round(e)) > 1e-9) return "";
  return v >= 0.001 && v < 100000 ? String(+v.toPrecision(6)) : v.toExponential(0);
};
const epochTick = (v) => (v < 1 || decade(v) === "" ? "" : v.toLocaleString("en-US"));

initTheme(document.getElementById("theme"));

const state = { visible: null, emphasis: null, normalize: false, reduction: "batch" };
let d = null;
let W = [];
let width = 0;
let redraw = () => {};

load("data/infoplane.json").then((data) => {
  d = data;
  W = d.weightLayers;
  state.visible = W.map(() => true);
  document.getElementById("provenance").textContent = provenance(d);

  const box = document.getElementById("normalize");
  box.checked = state.normalize;
  box.addEventListener("change", () => {
    state.normalize = box.checked;
    rebuild();
  });

  const pick = document.getElementById("reduction");
  // An export made before the step reduction existed still plots: the control
  // disappears rather than offering a curve that is not in the file.
  if (!d.grads[W[0]]?.stepStd) {
    pick.closest("label").hidden = true;
  } else {
    pick.value = state.reduction;
    pick.addEventListener("change", () => {
      state.reduction = pick.value;
      rebuild();
    });
  }

  legend(document.getElementById("legend"), W, {
    classFor: CLS,
    onEmphasis: (i) => {
      state.emphasis = i;
      redraw();
    },
    onToggle: (vis) => {
      state.visible = vis;
      redraw();
    },
  });

  responsive(document.getElementById("norms"), (w) => {
    width = w;
    rebuild();
  });
});

/** Rebuild both charts: a toggle changes the data, not just what is shown. */
function rebuild() {
  document.getElementById("gradnote").textContent =
    (step()
      ? `Mean and spread of the whole training set's gradient across the ` +
        `optimiser steps of each epoch`
      : `Mean and spread of the per-minibatch gradients of each weight matrix`) +
    `${state.normalize ? ", each divided by that matrix's L2 norm" : ""}. ` +
    `Median over ${d.nSeeds} seeds, ${d.meta.batch_size}-sample batches; the bold ` +
    `curve is a geometric mean over a ±${SMOOTH_DECADES}-decade window of epochs.`;
  const a = buildNorms(width);
  const b = buildSnr(Math.floor(width));
  redraw = () => {
    a();
    b();
  };
  redraw();
}

/** The measured curves for one matrix, in whichever form is selected.
 *
 * The mean follows the reduction too: over steps it is the norm of the mean of
 * the full-training-set gradients, which is not quite the mean of the minibatch
 * gradients (the last batch is short), and a ratio has to be taken between two
 * halves of the same collection.
 */
const step = () => state.reduction === "step";
const mean = (name) => {
  const g = d.grads[name];
  if (step()) return state.normalize ? g.stepMeanNorm : g.stepMean;
  return state.normalize ? g.meanNorm : g.mean;
};
const spread = (name) => {
  const g = d.grads[name];
  if (step()) return state.normalize ? g.stepStdNorm : g.stepStd;
  return state.normalize ? g.stdNorm : g.std;
};
/** ‖mean‖ / spread, from whichever pair is on screen. */
const snrOf = (name) => {
  const m = mean(name);
  const sp = spread(name);
  return Float64Array.from(m, (v, i) => v / sp[i]);
};

/** Geometric running mean over a ±SMOOTH_DECADES window of log(epoch).
 *
 * Geometric, not arithmetic, because the figure is log-log and these norms are
 * log-distributed: the mean of the logs is what a reader's eye takes off a
 * smooth curve through a scatter on this axis. The init point has no log
 * position and is left exactly as measured.
 */
function smooth(v) {
  const n = v.length;
  const out = new Float64Array(n).fill(NaN);
  const lg = d.epochs.map((e) => (e > 0 ? Math.log10(e) : NaN));
  for (let i = 0; i < n; i += 1) {
    if (!Number.isFinite(lg[i])) {
      out[i] = v[i];
      continue;
    }
    let s = 0;
    let k = 0;
    for (let j = 0; j < n; j += 1) {
      if (!Number.isFinite(lg[j]) || Math.abs(lg[j] - lg[i]) > SMOOTH_DECADES) continue;
      if (!(v[j] > 0) || !Number.isFinite(v[j])) continue;
      s += Math.log(v[j]);
      k += 1;
    }
    out[i] = k ? Math.exp(s / k) : NaN;
  }
  return out;
}

/** Every checkpoint, epoch 0 included.
 *
 * The untrained network is the left edge of this figure -- it is where the mean
 * and the spread are furthest apart, and leaving it out hides the first and
 * largest part of the rise. A log axis has no position for 0, so it is placed
 * half a decade left of epoch 1 and labelled, with a rule marking the break.
 */
const defined = () => d.epochs.map((_, i) => i);
const INIT_X = 0.5;
const xpos = (x, e) => x(e === 0 ? INIT_X : e);

/** The shared epoch scale, with room for the init point on the left.
 *
 * The step reduction has nothing at init -- one weight state is not a spread --
 * so there the axis starts at epoch 1 and the break marker is left off, rather
 * than opening a gap where a point cannot exist.
 */
function epochScale(w) {
  return scaleLog([step() ? 1 : INIT_X, d.meta.epochs], [0, w]);
}

/** Mark where the log axis actually begins, so the init point is not read as
 * "epoch 0.5". */
function initRule(plot, x, h) {
  if (step()) return;
  const at = x(Math.sqrt(INIT_X * 1));
  plot
    .append("line")
    .attr("class", "crosshair")
    .attr("x1", at)
    .attr("x2", at)
    .attr("y1", 0)
    .attr("y2", h)
    .attr("stroke-dasharray", "2 3");
  plot
    .append("text")
    .attr("class", "bound-label")
    .attr("x", x(INIT_X))
    .attr("y", h + 14)
    .attr("text-anchor", "middle")
    .text("init");
}

function logExtent(pick) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const name of W) {
    const v = pick(name);
    for (const i of defined()) {
      if (Number.isFinite(v[i]) && v[i] > 0) {
        lo = Math.min(lo, v[i]);
        hi = Math.max(hi, v[i]);
      }
    }
  }
  return [lo, hi];
}

function buildNorms(width) {
  const host = document.getElementById("norms");
  const height = Math.max(320, Math.min(520, Math.round(width * 0.5)));
  const { plot, w, h } = frame(host, { width, height, margin: { left: 78, right: 24, top: 16 } });

  const idx = defined();
  const x = epochScale(w);
  // One scale for both quantities: the figure's whole content is where the mean
  // curve crosses the spread curve, which a separate scale per quantity would
  // destroy.
  const [mlo, mhi] = logExtent(mean);
  const [slo, shi] = logExtent(spread);
  const y = scaleLog([Math.min(mlo, slo) * 0.8, Math.max(mhi, shi) * 1.25], [h, 0]);

  gridY(plot, y, w, 5);
  xAxis(plot, x, { w, h, label: "Training epoch (log scale)", ticks: 6, tickFormat: epochTick });
  yAxis(plot, y, {
    h,
    label: state.normalize ? "‖gradient‖ / ‖W‖  (log scale)" : "‖gradient‖  (log scale)",
    ticks: 6,
    tickFormat: decade,
  });
  initRule(plot, x, h);

  const mk = (v) =>
    line()
      .defined((i) => Number.isFinite(v[i]) && v[i] > 0)
      .x((i) => xpos(x, d.epochs[i]))
      .y((i) => y(v[i]));

  // Measured first, faint; the smoothed curve goes on top of it.
  const draw = (v, cls) => plot.append("path").attr("class", cls).attr("d", mk(v)(idx));
  const means = W.map((name, k) => [
    draw(mean(name), `series-line raw ${CLS(k)}`),
    draw(smooth(mean(name)), `series-line ${CLS(k)}`),
  ]);
  const stds = W.map((name, k) => [
    draw(spread(name), `series-line raw dashed ${CLS(k)}`),
    draw(smooth(spread(name)), `series-line dashed ${CLS(k)}`),
  ]);

  return function render() {
    W.forEach((_, k) => {
      const on = state.visible[k];
      const emph = state.emphasis == null || state.emphasis === k;
      for (const p of [...means[k], ...stds[k]]) {
        p.attr("display", on ? null : "none").classed("dim", !emph);
      }
    });
  };
}

function buildSnr(width) {
  const host = document.getElementById("snr");
  const height = Math.max(280, Math.min(440, Math.round(width * 0.42)));
  const { plot, w, h } = frame(host, { width, height, margin: { left: 78, right: 24, top: 16 } });

  const idx = defined();
  const x = epochScale(w);
  const snr = Object.fromEntries(W.map((name) => [name, snrOf(name)]));
  const y = scaleLog(logExtent((name) => snr[name]), [h, 0]).nice();

  gridY(plot, y, w, 5);
  xAxis(plot, x, { w, h, label: "Training epoch (log scale)", ticks: 6, tickFormat: epochTick });
  yAxis(plot, y, {
    h,
    label: step() ? "gradient SNR  ‖mean‖ / step spread" : "gradient SNR  ‖mean‖ / spread",
    ticks: 6,
    tickFormat: decade,
  });
  initRule(plot, x, h);

  // SNR = 1 is where the mean gradient and the batch-to-batch noise are equal:
  // the paper's drift-to-diffusion transition, read off the crossing rather than
  // asserted. Drawn as a bound, not a series.
  if (y.domain()[0] <= 1 && 1 <= y.domain()[1]) {
    plot
      .append("line")
      .attr("class", "bound")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", y(1))
      .attr("y2", y(1));
    plot
      .append("text")
      .attr("class", "bound-label")
      .attr("x", 4)
      .attr("y", y(1) - 5)
      .text(
        step()
          ? "SNR = 1 — the gradient moves as far within an epoch as its own size"
          : "SNR = 1 — mean gradient equals the batch-to-batch noise",
      );
  }

  const mk = (v) =>
    line()
      .defined((i) => Number.isFinite(v[i]) && v[i] > 0)
      .x((i) => xpos(x, d.epochs[i]))
      .y((i) => y(v[i]));
  const draw = (v, cls) => plot.append("path").attr("class", cls).attr("d", mk(v)(idx));

  const lines = W.map((name, k) => [
    draw(snr[name], `series-line raw ${CLS(k)}`),
    draw(smooth(snr[name]), `series-line ${CLS(k)}`),
  ]);

  return function render() {
    W.forEach((_, k) => {
      const on = state.visible[k];
      const emph = state.emphasis == null || state.emphasis === k;
      for (const p of lines[k]) {
        p.attr("display", on ? null : "none").classed("dim", !emph);
      }
    });
    table();
  };
}

/** Where each weight matrix's SNR last crosses 1, which is the transition the
 * paper marks with a single grey line for the whole network.
 *
 * Read off the SMOOTHED SNR, not the measured one: a curve that crosses 1
 * fourteen times while it wanders across the line has no single crossing epoch,
 * and the number the paper reports is the one a reader takes off the figure.
 */
function table() {
  const idx = defined();
  const unit = state.normalize ? "/‖W‖" : "";
  tableView(document.getElementById("grad-table"), {
    summary: "Table view — gradient statistics, and where each matrix crosses SNR = 1",
    columns: [
      "Weights",
      "Peak SNR",
      "at epoch",
      "Last SNR = 1 crossing",
      `Final ‖mean‖${unit}`,
      `Final spread${unit}`,
    ],
    swatchFor: CLS,
    rows: W.map((name, k) => {
      const s = smooth(snrOf(name));
      let best = -Infinity;
      let bestAt = null;
      let crossing = null;
      let prev = null;
      for (const i of idx) {
        const v = s[i];
        if (!Number.isFinite(v)) continue;
        if (v > best) {
          best = v;
          bestAt = d.epochs[i];
        }
        if (prev != null && prev >= 1 && v < 1) crossing = d.epochs[i];
        prev = v;
      }
      const last = idx[idx.length - 1];
      return {
        layer: k,
        cells: [
          name,
          best.toFixed(2),
          bestAt == null ? "—" : bestAt.toLocaleString("en-US"),
          crossing == null ? "never above 1" : crossing.toLocaleString("en-US"),
          mean(name)[last].toExponential(2),
          spread(name)[last].toExponential(2),
        ],
      };
    }),
  });
}
