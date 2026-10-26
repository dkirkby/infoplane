/** Figure 3 - the plane at four training-set fractions, as small multiples.
 *
 * The paper's Figure 3 is three panels at 5%, 45% and 85% of the patterns, each
 * showing the layers' averaged trajectories over the whole run. This is the same
 * comparison in a 2x2, with 15% added so the gap between 5% and 45% is not a
 * jump: at 5% the compression phase takes label information away with it, and at
 * 85% it mostly does not.
 *
 * Faceted rather than overlaid, and the reason is the palette: these are scatter
 * panels, so colour has to separate any two marks rather than just neighbours,
 * and that budget is spent on depth (see PLANE_SERIES). Four overlaid training
 * sizes on top of that could not be coloured safely.
 *
 * Marks, hues and the heavy mean path are shared with Figure 2 through
 * chart.js, so the two pages cannot drift apart: a trajectory means the same
 * thing on both.
 *
 * Each size is a separate Phase 1 export (~1.9 MB); all four load up front, since
 * all four are always on screen -- the point of the figure is the comparison
 * between them, which a reader should not have to assemble by toggling.
 */
import { line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { load, loadSweepIndex } from "./data.js";
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

const state = { epochIdx: 0, emphasis: null, visible: null };
const sets = new Map(); // trainSize -> prepared dataset
const panels = new Map(); // trainSize -> { card, update }
let manifest = [];
let shown = []; // indices into d.layers that are plotted, shared by every panel
let pinned = []; // ...and the ones stuck at the ceiling
let total = 4096; // denominator for the "% of the data" labels

loadSweepIndex().then(async (sizes) => {
  manifest = sizes;
  await loadAll();

  const any = sets.values().next().value;
  total = any.meta.n_samples ?? total;
  pinned = pinnedLayers(any);
  const isPinned = new Set(pinned);
  shown = any.layers.map((_, li) => li).filter((li) => !isPinned.has(li));
  if (shown.length > PLANE_SERIES.length) shown = shown.slice(-PLANE_SERIES.length);
  state.visible = shown.map(() => true);

  document.getElementById("provenance").textContent =
    `${any.nSeeds} seeds · ${any.meta.epochs.toLocaleString("en-US")} epochs · ` +
    `${any.meta.bins} bins · random training subsets`;
  if (pinned.length) {
    document.getElementById("pinned-note").textContent =
      `${pinnedLabel(any, pinned)} sit at the (H(X), H(Y)) corner in every panel and never ` +
      `leave it — ${any.meta.bins} bins across ${pinned.map((li) => any.meta.widths[li]).join(" and ")} ` +
      `tanh units separate all ${total.toLocaleString("en-US")} input patterns, which makes those ` +
      `layers invertible functions of the input whatever the weights are. They are the hollow ` +
      `marker, not a trajectory.`;
  }

  legend(document.getElementById("legend"), shown.map((li) => short(any.layers[li])), {
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
    epochs: any.epochs,
    onChange: (i) => {
      state.epochIdx = i;
      paint();
    },
  });

  accuracyTable();
  responsive(document.getElementById("panels"), () => rebuild());
});

async function loadAll() {
  const host = document.getElementById("panels");
  host.classList.add("loading"); // hold the previous render, no skeleton
  await Promise.all(
    manifest.map(async (m) => {
      sets.set(m.train_size, await load(`data/trainsize/${m.file}`));
    })
  );
  host.classList.remove("loading");
}

/** "45% of the data" - the paper states these panels as fractions, not counts. */
const pct = (n) => `${Math.round((n / total) * 100)}%`;

function rebuild() {
  const host = document.getElementById("panels");
  host.textContent = "";
  panels.clear();

  // Two passes on purpose. The CSS grid decides the column count from how many
  // cards are in it, so a card measured while it is still the only one reports
  // the full grid width and its chart would then be built too large and scaled
  // down. Append every card first, let the grid settle, and only then measure.
  const built = manifest.map((m) => {
    const card = document.createElement("section");
    card.className = "card panel";
    const head = document.createElement("div");
    head.className = "card-head";
    const left = document.createElement("div");
    // One line, no accuracy subtitle: the accuracies are all in the table below
    // the grid, where they can be compared, and a second line in every panel is
    // 4 lines of height spent on something read once.
    const h2 = document.createElement("h2");
    h2.textContent = `${pct(m.train_size)} of the data (N = ${m.train_size.toLocaleString("en-US")})`;
    left.append(h2);
    head.append(left);
    const plot = document.createElement("div");
    plot.className = "plot";
    card.append(head, plot);
    host.append(card);
    return { m, card, plot };
  });

  for (const { m, card, plot } of built) {
    const panelW = Math.max(240, Math.floor(plot.getBoundingClientRect().width));
    panels.set(m.train_size, {
      card,
      update: buildPanel(plot, sets.get(m.train_size), panelW),
    });
  }
  paint();
}

function paint() {
  for (const p of panels.values()) p.update();
  planeTable();
}

/** One table for the whole grid, rather than one under each panel.
 *
 * The panels carry no direct labels -- identity comes from the shared legend --
 * and two of the light-mode hues sit under 3:1 against the surface, so some
 * non-colour path to the numbers has to exist. One table for four panels keeps
 * that path and keeps the grid compact; four disclosure rows inside the grid
 * were height spent on furniture.
 */
function planeTable() {
  const rows = [];
  for (const m of manifest) {
    const d = sets.get(m.train_size);
    if (!d) continue;
    const e = Math.min(state.epochIdx, d.nEpochs - 1);
    d.layers.forEach((name, li) => {
      const si = shown.indexOf(li);
      const a = d.layerAgg(li);
      rows.push({
        layer: si >= 0 ? si : null,
        // Layer first, so the swatch -- which tableView puts on the first cell --
        // lands beside the series it belongs to rather than beside a percentage.
        cells: [
          si >= 0 ? name : `${name} (pinned)`,
          pct(m.train_size),
          bits(a.ixt[e]),
          bits(a.ity[e]),
        ],
      });
    });
  }
  const any = sets.values().next().value;
  const e = Math.min(state.epochIdx, any.nEpochs - 1);
  tableView(document.getElementById("plane-table"), {
    summary: `Table view — every layer in every panel at epoch ${any.epochs[e].toLocaleString("en-US")}`,
    columns: ["Layer", "Share of data", "I(X;T)", "I(T;Y)"],
    swatchFor: (si) => PLANE_SERIES[si].cls,
    rows,
  });
}

function buildPanel(host, d, width) {
  const height = Math.max(230, Math.round(width * 0.78));
  const { plot, w, h } = frame(host, {
    width,
    height,
    margin: { left: 44, right: 16, bottom: 38, top: 26 },
  });

  const x = scaleLinear([0, d.bounds.H_X], [0, w]);
  const y = scaleLinear([0, d.bounds.H_Y], [h, 0]).nice();
  gridY(plot, y, w, 4);
  gridX(plot, x, h, 4);
  xAxis(plot, x, { w, h, label: "I(X;T) — bits", ticks: 4 });
  yAxis(plot, y, { h, label: "I(T;Y) — bits", ticks: 4, tickFormat: bits });

  // The limits the data does establish. The IB bound curve itself is not drawn:
  // Phase 1 does not solve the IB equations, so there is no honest curve to plot.
  plot
    .append("line")
    .attr("class", "bound")
    .attr("x1", 0)
    .attr("x2", w)
    .attr("y1", y(d.bounds.H_Y))
    .attr("y2", y(d.bounds.H_Y));
  plot
    .append("text")
    .attr("class", "bound-label")
    .attr("x", 3)
    .attr("y", y(d.bounds.H_Y) - 4)
    .text("H(Y)");

  // The pinned layers, as one hollow marker where the two bounds meet - the same
  // mark, and the same reasoning, as Figure 2.
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

  const haloLayer = plot.append("g");
  const pathLayer = plot.append("g");
  const traceLayer = plot.append("g");
  const markLayer = plot.append("g");

  // The layer chain at this epoch, drawn exactly as Figure 2 draws it: neutral
  // and dashed, because it is not a series. It starts at the pinned marker -- the
  // data processing chain does not begin at the first plotted layer.
  const trace = traceLayer.append("path").attr("class", "trace");
  const chainLine = line()
    .x((pt) => x(pt[0]))
    .y((pt) => y(pt[1]));

  const mkPath = (li) =>
    line()
      .x((i) => x(d.layerAgg(li).ixt[i]))
      .y((i) => y(d.layerAgg(li).ity[i]))(d.epochs.map((_, i) => i));

  // Heavy coloured path over a surface halo, exactly as Figure 2 draws an
  // ensemble mean: every halo first, so no path is broken by another's.
  const halos = shown.map((li) =>
    haloLayer.append("path").attr("class", "epoch-path-halo mean").attr("d", mkPath(li))
  );
  const paths = shown.map((li, si) =>
    pathLayer.append("path").attr("class", `epoch-path mean ${PLANE_SERIES[si].cls}`).attr("d", mkPath(li))
  );
  const marks = shown.map((_, si) =>
    markLayer.append("path").attr("class", `dot ${PLANE_SERIES[si].cls}`).attr("d", planeMark(si, 70))
  );

  return function render() {
    const e = Math.min(state.epochIdx, d.nEpochs - 1);
    const chain = shown
      .filter((_, si) => state.visible[si])
      .map((li) => [d.layerAgg(li).ixt[e], d.layerAgg(li).ity[e]]);
    if (pinned.length && state.visible[0]) chain.unshift([d.bounds.H_X, d.bounds.H_Y]);
    trace.attr("d", chainLine(chain));

    shown.forEach((li, si) => {
      const on = state.visible[si];
      const emph = state.emphasis == null || state.emphasis === si;
      const a = d.layerAgg(li);
      paths[si].attr("display", on ? null : "none").classed("dim", !emph);
      halos[si].attr("display", on ? null : "none");
      marks[si]
        .attr("transform", `translate(${x(a.ixt[e])},${y(a.ity[e])})`)
        .attr("display", on ? null : "none")
        .classed("dim", !emph);
    });
  };
}

function accuracyTable() {
  tableView(document.getElementById("acc-table"), {
    summary: "Table view — accuracy by training-set size",
    columns: ["Share of data", "Training size", "Final train accuracy", "Final test accuracy", "Generalisation gap"],
    rows: manifest.map((m) => ({
      cells: [
        pct(m.train_size),
        m.train_size.toLocaleString("en-US"),
        m.final_train_acc.toFixed(4),
        m.final_test_acc.toFixed(4),
        (m.final_train_acc - m.final_test_acc).toFixed(4),
      ],
    })),
  });
}
