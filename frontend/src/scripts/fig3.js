/** Figure 3 - fitting against compression, and the ordering the DPI demands.
 *
 * Two charts: the mean path each layer takes through the plane over the whole run
 * with the scrubbed epoch marked, and the same instant as bits-by-depth. The
 * second chart exists because the data processing inequality is the measurement's
 * own check - I(X;T) must fall monotonically with depth, and a reader should be
 * able to see that it does rather than take it on trust.
 */
import { line } from "d3-shape";
import { scaleBand, scaleLinear } from "d3-scale";
import { load, provenance } from "./data.js";
import { initTheme } from "./theme.js";
import { scrubber } from "./scrubber.js";
import {
  Tooltip,
  bits,
  frame,
  gridX,
  gridY,
  lclass,
  legend,
  responsive,
  selectiveLabels,
  short,
  tableView,
  xAxis,
  yAxis,
} from "./chart.js";

const tip = new Tooltip();
initTheme(document.getElementById("theme"));

const state = { epochIdx: 0, emphasis: null, visible: null, trails: true };
let d = null;
let plane = null;
let dpi = null;

load("data/infoplane.json").then((data) => {
  d = data;
  state.visible = d.layers.map(() => true);
  document.getElementById("provenance").textContent = provenance(d);

  legend(document.getElementById("legend"), d.layers.map(short), {
    mark: "dot",
    onEmphasis: (i) => {
      state.emphasis = i;
      paint();
    },
    onToggle: (vis) => {
      state.visible = vis;
      paint();
    },
  });

  const scrub = scrubber({
    slider: document.getElementById("epoch"),
    readout: document.getElementById("epoch-readout"),
    playButton: document.getElementById("play"),
    epochs: d.epochs,
    onChange: (i) => {
      state.epochIdx = i;
      paint();
    },
  });

  const trailBtn = document.getElementById("trail");
  trailBtn.addEventListener("click", () => {
    state.trails = !state.trails;
    trailBtn.setAttribute("aria-pressed", String(state.trails));
    paint();
  });

  responsive(document.getElementById("plane"), (w) => {
    plane = buildPlane(w, scrub);
    paint();
  });
  responsive(document.getElementById("dpi"), (w) => {
    dpi = buildDpi(w);
    paint();
  });
});

function paint() {
  plane?.();
  dpi?.();
}

function buildPlane(width, scrub) {
  const host = document.getElementById("plane");
  const height = Math.max(360, Math.min(520, Math.round(width * 0.58)));
  const { plot, w, h } = frame(host, { width, height, margin: { left: 54, right: 28, top: 14 } });

  const x = scaleLinear([0, d.bounds.H_X], [0, w]);
  const y = scaleLinear([0, d.bounds.H_Y], [h, 0]).nice();
  gridY(plot, y, w, 5);
  gridX(plot, x, h, 6);
  xAxis(plot, x, { w, h, label: "I(X;T)  —  bits about the input", ticks: 6 });
  yAxis(plot, y, { h, label: "I(T;Y)  —  bits about the label", ticks: 5, tickFormat: bits });

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
    .attr("x", w)
    .attr("y", y(d.bounds.H_Y) - 5)
    .attr("text-anchor", "end")
    .text(`H(Y) = ${bits(d.bounds.H_Y)} bits`);

  const trailLayer = plot.append("g");
  const walkLayer = plot.append("g");
  const markLayer = plot.append("g");

  const mkPath = (k, upto) =>
    line()
      .x((i) => x(d.layerAgg(k).ixt[i]))
      .y((i) => y(d.layerAgg(k).ity[i]))(
      d.epochs.map((_, i) => i).slice(0, upto + 1)
    );

  const trails = d.layers.map((_, k) =>
    trailLayer.append("path").attr("class", `series-line ${lclass(k)}`).attr("opacity", 0.22)
  );
  const walked = d.layers.map((_, k) => walkLayer.append("path").attr("class", `series-line ${lclass(k)}`));
  const marks = d.layers.map((_, k) => markLayer.append("circle").attr("class", `dot ${lclass(k)}`).attr("r", 6.5));
  const labels = d.layers.map((name, k) =>
    markLayer.append("text").attr("class", "point-label").text(short(name))
  );

  return function render() {
    const e = state.epochIdx;
    const keep = selectiveLabels(
      d.layers.map((_, k) => (state.visible[k] ? y(d.layerAgg(k).ity[e]) : NaN))
    );
    d.layers.forEach((_, k) => {
      const on = state.visible[k];
      const emph = state.emphasis == null || state.emphasis === k;
      trails[k]
        .attr("d", state.trails ? mkPath(k, d.nEpochs - 1) : null)
        .attr("display", on && state.trails ? null : "none");
      walked[k].attr("d", mkPath(k, e)).attr("display", on ? null : "none").classed("dim", !emph);
      marks[k]
        .attr("cx", x(d.layerAgg(k).ixt[e]))
        .attr("cy", y(d.layerAgg(k).ity[e]))
        .attr("display", on ? null : "none")
        .classed("dim", !emph);
      labels[k]
        .attr("x", x(d.layerAgg(k).ixt[e]) + 10)
        .attr("y", y(d.layerAgg(k).ity[e]) + 4)
        .attr("display", on && keep.has(k) ? null : "none")
        .classed("dim", !emph);
    });

    tableView(document.getElementById("plane-table"), {
      summary: `Table view — mean trajectory at epoch ${d.epochs[e].toLocaleString("en-US")}`,
      columns: ["Layer", "I(X;T)", "I(T;Y)", "I(X;T) at end", "compressed from peak"],
      rows: d.layers.map((name, k) => {
        const a = d.layerAgg(k);
        let peak = 0;
        for (let i = 0; i < d.nEpochs; i++) if (a.ixt[i] > a.ixt[peak]) peak = i;
        return {
          layer: k,
          cells: [
            name,
            bits(a.ixt[e]),
            bits(a.ity[e]),
            bits(a.ixt[d.nEpochs - 1]),
            bits(a.ixt[peak] - a.ixt[d.nEpochs - 1]),
          ],
        };
      }),
    });
  };
}

function buildDpi(width) {
  const host = document.getElementById("dpi");
  const height = 240;
  const { plot, w, h } = frame(host, { width, height, margin: { left: 54, right: 24, bottom: 42, top: 10 } });

  const xb = scaleBand(d.layers.map(short), [0, w]).padding(0.34);
  const y = scaleLinear([0, d.bounds.H_X], [h, 0]).nice();
  gridY(plot, y, w, 5);
  xAxis(plot, xb, { w, h, label: "Layer (input on the left, deepest on the right)" });
  yAxis(plot, y, { h, label: "I(X;T)  —  bits", ticks: 5, tickFormat: bits });

  // Columns capped at 24px with a 4px rounded cap and a square foot on the
  // baseline; the band's leftover width is deliberately air.
  const bw = Math.min(24, xb.bandwidth());
  const bars = d.layers.map((_, k) => plot.append("path").attr("class", `dot ${lclass(k)}`).attr("stroke", "none"));
  const vals = d.layers.map(() => plot.append("text").attr("class", "point-label").attr("text-anchor", "middle"));

  const hit = d.layers.map((name, k) => {
    const cx = xb(short(name)) + xb.bandwidth() / 2;
    return plot
      .append("rect")
      .attr("x", cx - Math.max(12, bw / 2 + 4))
      .attr("y", 0)
      .attr("width", Math.max(24, bw + 8))
      .attr("height", h)
      .attr("fill", "transparent")
      .on("pointerenter", (event) => {
        const e = state.epochIdx;
        tip.show(event.clientX, event.clientY, `${name} at epoch ${d.epochs[e].toLocaleString("en-US")}`, [
          { layer: k, name: "I(X;T)", value: `${bits(d.layerAgg(k).ixt[e])} bits` },
          { layer: k, name: "I(T;Y)", value: `${bits(d.layerAgg(k).ity[e])} bits` },
          { layer: k, name: "sd across seeds", value: `${bits(d.layerAgg(k).ixtSd[e])} bits` },
        ]);
      })
      .on("pointerleave", () => tip.hide());
  });

  return function render() {
    const e = state.epochIdx;
    let monotone = true;
    d.layers.forEach((name, k) => {
      const v = d.layerAgg(k).ixt[e];
      if (k > 0 && v > d.layerAgg(k - 1).ixt[e] + 1e-9) monotone = false;
      const cx = xb(short(name)) + xb.bandwidth() / 2;
      const top = y(v);
      const r = Math.min(4, Math.max(0, (h - top) / 2));
      // Rounded at the data end, square at the baseline.
      bars[k]
        .attr(
          "d",
          `M${cx - bw / 2},${h} L${cx - bw / 2},${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top}
           L${cx + bw / 2 - r},${top} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} L${cx + bw / 2},${h} Z`
        )
        .attr("display", state.visible[k] ? null : "none")
        .classed("dim", !(state.emphasis == null || state.emphasis === k));
      vals[k]
        .attr("x", cx)
        .attr("y", top - 6)
        .text(bits(v))
        .attr("display", state.visible[k] ? null : "none");
    });

    document.getElementById("dpi-note").textContent = monotone
      ? "Monotone decreasing with depth, as the data processing inequality requires: a layer cannot know more about the input than the layer feeding it."
      : "Not monotone at this epoch — the binned estimator can invert the ordering by a hairline where a layer saturates. See the Overview.";

    tableView(document.getElementById("dpi-table"), {
      summary: `Table view — bits by depth at epoch ${d.epochs[e].toLocaleString("en-US")}`,
      columns: ["Layer", "width", "I(X;T)", "drop from previous layer"],
      rows: d.layers.map((name, k) => ({
        layer: k,
        cells: [
          name,
          String(d.meta.widths[k]),
          bits(d.layerAgg(k).ixt[e]),
          k === 0 ? "—" : bits(d.layerAgg(k - 1).ixt[e] - d.layerAgg(k).ixt[e]),
        ],
      })),
    });
  };
}
