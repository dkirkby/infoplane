/** Figure 5 - the plane for one to six hidden layers, as small multiples.
 *
 * The paper's Figure 5 is a 2x3 of information planes, one per architecture,
 * asking what the hidden layers buy. The architectures are the authors' own,
 * read off their plotting code rather than the caption (see `cli.cmd_arch`):
 * [10], [10,7], [10,7,5], [10,7,5,4], [10,7,5,4,3], [10,7,5,4,4,3]. They grow
 * from the input end, so the FIFTH panel is the main run's architecture and
 * should reproduce Figure 2's means.
 *
 * Series are keyed from the OUTPUT end of the chain rather than by index: the
 * readout is the 2nd series in the first panel and the 7th in the last but it
 * is the same thing, "the last hidden layer" is the thing the panels are being
 * compared on, and the six-layer net is not a prefix of the others, so counting
 * from the input would not line the panels up anyway.
 *
 * Styled as Figure 3: faceted, one shared legend, one shared scrubber, the
 * ensemble-mean path per layer drawn exactly as Figure 2 draws a mean, and the
 * marks, hues and pinned-layer treatment taken from chart.js so the three plane
 * pages cannot drift apart. Four slots is the whole palette, which covers the
 * readout plus L(n), L(n-1), L(n-2); the pinning takes care of the rest, except
 * in the deepest panel, where one more layer moves than there are colours and
 * the shallowest mover is left out. `pinnedNote` says which, and what it does.
 *
 * The paper's own panels are coloured by epoch instead, with the layers left
 * unlabelled. That reads well as a still and badly as a page: with a scrubber
 * the epoch is already a control, and spending the palette on it would leave
 * nothing to tell six layers apart with.
 */
import { line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { load, loadArchIndex } from "./data.js";
import { initTheme } from "./theme.js";
import { scrubber } from "./scrubber.js";
import {
  PLANE_SERIES,
  bits,
  frame,
  gridX,
  gridY,
  legend,
  pinnedLabel,
  pinnedLayers,
  planeMark,
  responsive,
  short,
  tableView,
  xAxis,
  yAxis,
} from "./chart.js";

initTheme(document.getElementById("theme"));

// Panel titles as the paper words them: "One hidden layer" ... "Six hidden layers".
const WORDS = ["zero", "One", "Two", "Three", "Four", "Five", "Six"];

// "Good generalisation" for the claim that depth reaches it sooner. The paper
// says the six-layer net "reached the full relevant information at the output
// layer within 400 epochs", so the measure is on the OUTPUT layer's I(T;Y) as a
// fraction of H(Y). THREE thresholds, because one would misreport the answer:
// the order of the architectures reverses between the loosest and the strictest.
const BARS = [0.8, 0.9, 0.95];

const state = { epochIdx: 0, emphasis: null, visible: null };
const sets = new Map(); // n_hidden -> prepared dataset
const panels = new Map(); // n_hidden -> { update }
let manifest = [];
let roles = []; // ROLE_NAMES, truncated to what the panels actually plot
let total = 4096;

/** A layer's role, counted from the OUTPUT end of the chain.
 *
 * Panels have different layer counts, so an index from the input is not an
 * identity a reader can carry across them: L3 is the last hidden layer in one
 * panel and the middle of another. Counting from the readout instead gives four
 * roles that mean the same thing everywhere -- the readout, the last hidden
 * layer, the one before it, the one before that -- which is also the comparison
 * the figure is for, since it is the deepest layers that move. The panel titles
 * carry n, so L(n-1) resolves per panel.
 */
const ROLE_NAMES = ["Output", "L(n)", "L(n−1)", "L(n−2)"];
const slotOf = (d, li) => (li < d.meta.widths.length ? d.meta.widths.length - li : 0);

loadArchIndex().then(async (archs) => {
  manifest = archs.slice().sort((a, b) => a.n_hidden - b.n_hidden);
  await loadAll();

  const deepest = sets.get(manifest[manifest.length - 1].n_hidden);
  total = deepest.meta.n_samples ?? total;

  // How many roles any panel actually plots. Pinned layers are left out here as
  // they are on Figures 2 and 3 -- a layer that never leaves the (H(X), H(Y))
  // corner is one hollow marker, not a trajectory -- and that is also what keeps
  // this page inside the four-slot plane palette: nowhere do more than three
  // hidden layers leave the ceiling.
  let maxSlot = 0;
  for (const m of manifest) {
    const d = sets.get(m.n_hidden);
    const skip = new Set(pinnedLayers(d));
    d.layers.forEach((_, li) => {
      if (!skip.has(li)) maxSlot = Math.max(maxSlot, slotOf(d, li));
    });
  }
  roles = ROLE_NAMES.slice(0, Math.min(maxSlot + 1, PLANE_SERIES.length));
  state.visible = roles.map(() => true);

  document.getElementById("provenance").textContent =
    `${deepest.nSeeds} seeds · ${deepest.meta.epochs.toLocaleString("en-US")} epochs · ` +
    `${deepest.meta.bins} bins · ${Math.round(deepest.meta.train_frac * 100)}% of the patterns`;

  pinnedNote();
  claimsNote();

  legend(document.getElementById("legend"), roles, {
    classFor: (i) => PLANE_SERIES[i].cls,
    keyClassFor: (i) => PLANE_SERIES[i].key,
    onEmphasis: (i) => {
      state.emphasis = i;
      paint();
    },
    onToggle: (vis) => {
      state.visible = vis;
      paint();
    },
  });

  scrubber({
    slider: document.getElementById("epoch"),
    readout: document.getElementById("epoch-readout"),
    playButton: document.getElementById("play"),
    epochs: deepest.epochs,
    onChange: (i) => {
      state.epochIdx = i;
      paint();
    },
  });

  depthTable();
  responsive(document.getElementById("panels"), () => rebuild());
});

async function loadAll() {
  const host = document.getElementById("panels");
  host.classList.add("loading");
  await Promise.all(
    manifest.map(async (m) => {
      sets.set(m.n_hidden, await load(`data/arch/${m.file}`));
    })
  );
  host.classList.remove("loading");
}

/** First checkpoint at which the readout carries `bar` of the label information. */
function epochsToBar(d, bar) {
  const want = bar * d.bounds.H_Y;
  const ity = readout(d).ity;
  for (let i = 0; i < ity.length; i += 1) if (ity[i] >= want) return d.epochs[i];
  return null;
}

/** The readout is measured last, after every hidden layer. */
const readout = (d) => d.layerAgg(d.layers.length - 1);

function pinnedNote() {
  // How many of the DEEPEST hidden layers move at all, per panel. Which widths
  // are pinned is not a property of the width: the 8-wide layer compresses 6.1
  // bits when it is the last hidden layer and sits on the ceiling when there
  // are two more behind it, so the note has to be built per panel.
  const moved = manifest.map((m) => {
    const d = sets.get(m.n_hidden);
    const pin = new Set(pinnedLayers(d));
    let n = 0;
    for (let li = d.meta.widths.length - 1; li >= 0 && !pin.has(li); li -= 1) n += 1;
    return n;
  });
  // A panel can have more movers than the palette has slots. Nothing is dropped
  // quietly: name it, and say how far it actually moves.
  const extra = [];
  for (const m of manifest) {
    const d = sets.get(m.n_hidden);
    const pin = new Set(pinnedLayers(d));
    d.layers.forEach((name, li) => {
      if (pin.has(li) || slotOf(d, li) < roles.length) return;
      const a = d.layerAgg(li);
      const moved = Math.max(...a.ixt) - a.ixt[a.ixt.length - 1];
      extra.push(
        `${short(name)} in the ${m.n_hidden}-layer panel, which moves ${bits(moved)} bits`
      );
    });
  }
  const widest = sets.get(manifest[0].n_hidden);
  const first = widest.meta.widths[0];
  document.getElementById("pinned-note").innerHTML =
    `<strong>Only the deepest layers move.</strong> A layer whose binning separates all ` +
    `${total.toLocaleString("en-US")} input patterns is an invertible function of the ` +
    `input, so it sits at (H(X), H(Y)) for any weights at all; those are drawn as the ` +
    `hollow marker in the top right rather than as trajectories. Counting the deepest ` +
    `hidden layers that leave the ceiling, the six panels give ${moved.join(", ")} — the ` +
    `${first}-wide first layer is pinned in every one of them, and a layer compresses most ` +
    `when it is the last hidden one, less the more layers sit behind it. The paper reports ` +
    `the same of its own first hidden layer — which it calls 12 wide, counting the input — ` +
    `"such 1-1 transformations do not help learning, since they do not generate compressed ` +
    `representations" — but adds that in its simulations they eventually compress given ` +
    `enough epochs. In ` +
    `${widest.meta.epochs.toLocaleString("en-US")} epochs here, none of the pinned ones ` +
    `moves at all.` +
    (extra.length
      ? ` One layer moves but has no colour left: ${extra.join(", ")}, against four ` +
        `slots in the palette. It is not drawn.`
      : "");
}

/** The paper's first claim - that depth buys speed - checked against these runs.
 *
 * Written from the numbers rather than around them: which way the comparison
 * goes depends on where the bar is, and the note has to say so whichever way
 * this run comes out.
 */
function claimsNote() {
  const ep = (e) => (e == null ? "never" : e.toLocaleString("en-US"));
  const at = (bar) => manifest.map((m) => epochsToBar(sets.get(m.n_hidden), bar));
  const list = (bar) =>
    manifest.map((m, i) => `${m.n_hidden} → ${ep(at(bar)[i])}`).join(", ");
  const loose = at(BARS[0]);
  const strict = at(BARS[BARS.length - 1]);
  const miss = manifest.filter((_, i) => strict[i] == null);
  const speed =
    loose[0] != null && loose[loose.length - 1] != null
      ? (loose[0] / loose[loose.length - 1]).toFixed(1)
      : null;
  // Say "monotone" only if it is: the whole point of the note is that the
  // ordering is not stable across the bars.
  const mono = loose.every((v, i) => i === 0 || (v != null && loose[i - 1] != null && v <= loose[i - 1]));

  let out =
    `<strong>Does depth buy speed?</strong> The paper says it does — "with 1 hidden ` +
    `layer the network was unable to achieve good I<sub>Y</sub> values even after ` +
    `10<sup>4</sup> epochs, with 6 hidden layers it reached the full relevant ` +
    `information at the output layer within 400 epochs". Measured as the first epoch ` +
    `at which the readout carries a given share of H(Y), the answer depends entirely ` +
    `on the share. At ${BARS[0] * 100}%: ${list(BARS[0])} epochs`;
  out += speed
    ? `, ${mono ? "monotone in depth and " : ""}${speed}x faster end to end. `
    : `. `;
  out += `At ${BARS[BARS.length - 1] * 100}%: ${list(BARS[BARS.length - 1])}`;
  if (miss.length) {
    const plateau = miss
      .map((m) => {
        const r = readout(sets.get(m.n_hidden));
        return `${m.n_hidden} layers plateau at ${bits(Math.max(...r.ity))}`;
      })
      .join(", ");
    out += ` — ${plateau} bits, short of the bar. `;
  } else {
    out += `. `;
  }
  if (loose[0] != null && loose[1] != null && loose[loose.length - 1] != null) {
    out +=
      `Most of that is the second hidden layer alone (${(loose[0] / loose[1]).toFixed(1)}x); ` +
      `the rest together are worth ${(loose[1] / loose[loose.length - 1]).toFixed(1)}x. `;
  }
  out +=
    `Depth brings the rise forward and lowers the ceiling it rises to, so the ` +
    `paper's claim holds for reaching a decent level quickly and fails for reaching ` +
    `the full relevant information at all. Nothing here reproduces a one-layer ` +
    `network that cannot learn: it gets to ${bits(Math.max(...readout(sets.get(manifest[0].n_hidden)).ity))} ` +
    `bits of ${bits(sets.get(manifest[0].n_hidden).bounds.H_Y)}.`;
  document.getElementById("claims-note").innerHTML = out;
}

function rebuild() {
  const host = document.getElementById("panels");
  host.textContent = "";
  panels.clear();

  // Two passes, as on Figure 3: a card measured while it is alone in the grid
  // reports the full grid width, so every card is appended before any is sized.
  const built = manifest.map((m) => {
    const card = document.createElement("section");
    card.className = "card panel";
    const head = document.createElement("div");
    head.className = "card-head";
    const left = document.createElement("div");
    const h2 = document.createElement("h2");
    // n in the title is what makes the legend's L(n), L(n-1) readable here.
    h2.textContent =
      `${WORDS[m.n_hidden]} hidden layer${m.n_hidden > 1 ? "s" : ""} (n = ${m.n_hidden})`;
    const sub = document.createElement("p");
    sub.className = "note";
    // The paper's own notation for an architecture: input, hidden widths, output.
    sub.textContent = `12-${m.widths.join("-")}-1`;
    left.append(h2, sub);
    head.append(left);
    const plot = document.createElement("div");
    plot.className = "plot";
    card.append(head, plot);
    host.append(card);
    return { m, plot };
  });

  for (const { m, plot } of built) {
    const panelW = Math.max(220, Math.floor(plot.getBoundingClientRect().width));
    panels.set(m.n_hidden, { update: buildPanel(plot, sets.get(m.n_hidden), panelW) });
  }
  paint();
}

function paint() {
  for (const p of panels.values()) p.update();
  planeTable();
}

/** One table for the whole grid, as on Figure 3: the panels carry no direct
 * labels, so the numbers need a path that is not colour. */
function planeTable() {
  const rows = [];
  for (const m of manifest) {
    const d = sets.get(m.n_hidden);
    if (!d) continue;
    const e = Math.min(state.epochIdx, d.nEpochs - 1);
    const skip = new Set(pinnedLayers(d));
    d.layers.forEach((name, li) => {
      const slot = skip.has(li) ? -1 : slotOf(d, li);
      const coloured = slot >= 0 && slot < roles.length;
      const a = d.layerAgg(li);
      rows.push({
        layer: coloured ? slot : null,
        cells: [
          skip.has(li) ? `${short(name)} (pinned)` : short(name),
          coloured ? roles[slot] : "—",
          m.n_hidden,
          li < d.meta.widths.length ? d.meta.widths[li] : "—",
          bits(a.ixt[e]),
          bits(a.ity[e]),
        ],
      });
    });
  }
  const any = sets.get(manifest[0].n_hidden);
  const e = Math.min(state.epochIdx, any.nEpochs - 1);
  tableView(document.getElementById("plane-table"), {
    summary: `Table view — every layer in every panel at epoch ${any.epochs[e].toLocaleString("en-US")}`,
    columns: ["Layer", "Role", "Hidden layers", "Width", "I(X;T)", "I(T;Y)"],
    swatchFor: (slot) => PLANE_SERIES[slot].cls,
    rows,
  });
}

function buildPanel(host, d, width) {
  const height = Math.max(200, Math.round(width * 0.8));
  const { plot, w, h } = frame(host, {
    width,
    height,
    margin: { left: 42, right: 14, bottom: 36, top: 14 },
  });

  // The same axes in every panel, including panels whose layers never go near
  // the left of them: the figure is a comparison, and a scale that fits each
  // panel would make six incomparable pictures.
  const x = scaleLinear([0, d.bounds.H_X], [0, w]);
  const y = scaleLinear([0, d.bounds.H_Y], [h, 0]).nice();
  gridY(plot, y, w, 4);
  gridX(plot, x, h, 4);
  xAxis(plot, x, { w, h, label: "I(X;T) — bits", ticks: 4 });
  yAxis(plot, y, { h, label: "I(T;Y) — bits", ticks: 4, tickFormat: bits });

  plot
    .append("line")
    .attr("class", "bound")
    .attr("x1", 0)
    .attr("x2", w)
    .attr("y1", y(d.bounds.H_Y))
    .attr("y2", y(d.bounds.H_Y));

  const pinned = pinnedLayers(d);
  if (pinned.length) {
    const pinX = x(d.bounds.H_X);
    const pinY = y(d.bounds.H_Y);
    plot.append("circle").attr("class", "pinned-mark").attr("cx", pinX).attr("cy", pinY).attr("r", 5);
    plot
      .append("text")
      .attr("class", "point-label pinned-label")
      .attr("x", pinX + 3)
      .attr("y", pinY - 10)
      .attr("text-anchor", "end")
      .text(pinnedLabel(d, pinned));
  }

  const skip = new Set(pinned);
  const shown = d.layers
    .map((_, li) => li)
    .filter((li) => !skip.has(li) && slotOf(d, li) < roles.length);

  const haloLayer = plot.append("g");
  const pathLayer = plot.append("g");
  const traceLayer = plot.append("g");
  const markLayer = plot.append("g");

  const trace = traceLayer.append("path").attr("class", "trace");
  const chainLine = line()
    .x((pt) => x(pt[0]))
    .y((pt) => y(pt[1]));

  const mkPath = (li) =>
    line()
      .x((i) => x(d.layerAgg(li).ixt[i]))
      .y((i) => y(d.layerAgg(li).ity[i]))(d.epochs.map((_, i) => i));

  const halos = shown.map((li) =>
    haloLayer.append("path").attr("class", "epoch-path-halo mean").attr("d", mkPath(li))
  );
  const paths = shown.map((li) =>
    pathLayer
      .append("path")
      .attr("class", `epoch-path mean ${PLANE_SERIES[slotOf(d, li)].cls}`)
      .attr("d", mkPath(li))
  );
  const marks = shown.map((li) =>
    markLayer
      .append("path")
      .attr("class", `dot ${PLANE_SERIES[slotOf(d, li)].cls}`)
      .attr("d", planeMark(slotOf(d, li), 60))
  );

  return function render() {
    const e = Math.min(state.epochIdx, d.nEpochs - 1);
    const on = (li) => state.visible[slotOf(d, li)];
    const chain = shown.filter(on).map((li) => [d.layerAgg(li).ixt[e], d.layerAgg(li).ity[e]]);
    if (pinned.length && chain.length) chain.unshift([d.bounds.H_X, d.bounds.H_Y]);
    trace.attr("d", chainLine(chain));

    shown.forEach((li, i) => {
      const slot = slotOf(d, li);
      const emph = state.emphasis == null || state.emphasis === slot;
      const a = d.layerAgg(li);
      paths[i].attr("display", on(li) ? null : "none").classed("dim", !emph);
      halos[i].attr("display", on(li) ? null : "none");
      marks[i]
        .attr("transform", `translate(${x(a.ixt[e])},${y(a.ity[e])})`)
        .attr("display", on(li) ? null : "none")
        .classed("dim", !emph);
    });
  };
}

function depthTable() {
  const any = sets.get(manifest[0].n_hidden);
  document.getElementById("depth-note").textContent =
    `Final accuracy after ${any.meta.epochs.toLocaleString("en-US")} epochs, and the first ` +
    `checkpoint (the → columns) at which the readout carries each share of H(Y) — the ` +
    `paper's "full relevant information at the output layer" — at three bars, because ` +
    `the order of the architectures reverses between them.`;
  tableView(document.getElementById("depth-table"), {
    summary: "Table view — what each depth reaches",
    columns: [
      "Hidden layers",
      "Widths",
      "Test accuracy",
      ...BARS.map((b) => `→ ${b * 100}% H(Y)`),
      "Readout I(T;Y)",
      "DPI violations",
    ],
    rows: manifest.map((m) => {
      const d = sets.get(m.n_hidden);
      const r = readout(d);
      return {
        cells: [
          m.n_hidden,
          m.widths.join("-"),
          m.final_test_acc.toFixed(4),
          ...BARS.map((b) => {
            const e = epochsToBar(d, b);
            return e == null ? "never" : e.toLocaleString("en-US");
          }),
          bits(r.ity[r.ity.length - 1]),
          `${m.dpi_violations.toLocaleString("en-US")} / ${m.dpi_measurements.toLocaleString("en-US")}`,
        ],
      };
    }),
  });
}
