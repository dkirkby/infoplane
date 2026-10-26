/** Figure 4 - both information quantities against training epoch.
 *
 * Two charts stacked on a shared x-axis, never one chart with two y-scales:
 * I(X;T) spans 0-12 bits and I(T;Y) spans 0-1, and aligning two such scales on
 * one plot invents a relationship that isn't in the data.
 *
 * The x-axis is logarithmic, which means epoch 0 cannot appear on it (log has no
 * zero). The initialisation checkpoint is therefore omitted here and shown on
 * Figures 2 and 3 instead, rather than being quietly clamped onto epoch 1.
 */
import { area, line } from "d3-shape";
import { scaleLinear, scaleLog } from "d3-scale";
import { load, provenance } from "./data.js";
import { initTheme } from "./theme.js";
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

const TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000, 9000];
const state = { emphasis: null, visible: null, allBands: false, hoverIdx: null };
let d = null;
const charts = [];

load("data/infoplane.json").then((data) => {
  d = data;
  state.visible = d.layers.map(() => true);
  document.getElementById("provenance").textContent = provenance(d);

  legend(document.getElementById("legend"), d.layers.map(short), {
    mark: "line",
    onEmphasis: (i) => {
      state.emphasis = i;
      charts.forEach((c) => c.paint());
    },
    onToggle: (vis) => {
      state.visible = vis;
      charts.forEach((c) => c.paint());
    },
  });

  const bandBtn = document.getElementById("band");
  bandBtn.addEventListener("click", () => {
    state.allBands = !state.allBands;
    bandBtn.setAttribute("aria-pressed", String(state.allBands));
    charts.forEach((c) => c.paint());
  });

  // from[0] is the first epoch >= 1, since epoch 0 has no place on a log axis.
  const first = d.epochs.findIndex((e) => e >= 1);

  responsive(document.getElementById("ixt"), (w) => {
    charts[0] = buildChart({
      host: document.getElementById("ixt"),
      width: w,
      first,
      key: "ixt",
      sdKey: "ixtSd",
      max: d.bounds.H_X,
      yLabel: "I(X;T)  —  bits",
      bound: { v: d.bounds.H_X, text: `H(X) = ${d.bounds.H_X} bits` },
      endLabels: true,
    });
    charts.forEach((c) => c.paint());
  });

  responsive(document.getElementById("ity"), (w) => {
    charts[1] = buildChart({
      host: document.getElementById("ity"),
      width: w,
      first,
      key: "ity",
      sdKey: "itySd",
      max: d.bounds.H_Y,
      yLabel: "I(T;Y)  —  bits",
      bound: { v: d.bounds.H_Y, text: `H(Y) = ${bits(d.bounds.H_Y)} bits` },
      // Every layer converges on H(Y) at the right edge, so end labels would pile
      // up on each other; the legend and the crosshair carry identity here.
      endLabels: false,
      showX: true,
    });
    charts.forEach((c) => c.paint());
  });

  summaryTable(first);
});

function buildChart({ host, width, first, key, sdKey, max, yLabel, bound, endLabels, showX }) {
  const height = Math.max(240, Math.min(340, Math.round(width * 0.38)));
  const { plot, w, h } = frame(host, {
    width,
    height,
    margin: { left: 56, right: endLabels ? 42 : 24, bottom: showX ? 42 : 26, top: 14 },
  });

  const x = scaleLog([Math.max(1, d.epochs[first]), d.epochs[d.epochs.length - 1]], [0, w]);
  const y = scaleLinear([0, max], [h, 0]).nice();

  gridY(plot, y, w, 5);
  gridX(plot, x, h, TICKS);
  xAxis(plot, x, {
    w,
    h,
    label: showX ? "Training epoch (log scale)" : null,
    ticks: TICKS,
    tickFormat: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)),
  });
  yAxis(plot, y, { h, label: yLabel, ticks: 5, tickFormat: bits });

  plot
    .append("line")
    .attr("class", "bound")
    .attr("x1", 0)
    .attr("x2", w)
    .attr("y1", y(bound.v))
    .attr("y2", y(bound.v));
  plot
    .append("text")
    .attr("class", "bound-label")
    .attr("x", w)
    .attr("y", y(bound.v) - 5)
    .attr("text-anchor", "end")
    .text(bound.text);

  const idx = d.epochs.map((_, i) => i).slice(first);
  const bandLayer = plot.append("g");
  const lineLayer = plot.append("g");
  const labelLayer = plot.append("g");

  // One generator per layer, so nothing depends on mutable state while drawing.
  const mkLine = (k) =>
    line()
      .x((i) => x(d.epochs[i]))
      .y((i) => y(d.layerAgg(k)[key][i]));
  const mkArea = (k) =>
    area()
      .x((i) => x(d.epochs[i]))
      .y0((i) => y(Math.max(0, d.layerAgg(k)[key][i] - d.layerAgg(k)[sdKey][i])))
      .y1((i) => y(Math.min(max, d.layerAgg(k)[key][i] + d.layerAgg(k)[sdKey][i])));

  const bandPaths = d.layers.map((_, k) =>
    bandLayer.append("path").attr("class", `band ${lclass(k)}`).attr("d", mkArea(k)(idx))
  );
  const linePaths = d.layers.map((_, k) =>
    lineLayer.append("path").attr("class", `series-line ${lclass(k)}`).attr("d", mkLine(k)(idx))
  );

  const endText = endLabels
    ? d.layers.map((name, k) =>
        labelLayer
          .append("text")
          .attr("class", "point-label")
          .attr("x", w + 5)
          .attr("y", y(d.layerAgg(k)[key][d.nEpochs - 1]) + 4)
          .text(short(name))
      )
    : [];

  const cross = plot
    .append("line")
    .attr("class", "crosshair")
    .attr("y1", 0)
    .attr("y2", h)
    .attr("display", "none");

  const hit = plot.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent");
  const toIdx = (px) => {
    const target = x.invert(px);
    let best = first;
    let bd = Infinity;
    for (const i of idx) {
      const dd = Math.abs(Math.log(d.epochs[i]) - Math.log(target));
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    return best;
  };

  hit.on("pointermove", (event) => {
    const r = hit.node().getBoundingClientRect();
    state.hoverIdx = toIdx(event.clientX - r.left);
    charts.forEach((c) => c.paintCross());
    const i = state.hoverIdx;
    tip.show(
      event.clientX,
      event.clientY,
      `Epoch ${d.epochs[i].toLocaleString("en-US")} — I(X;T) , I(T;Y)`,
      d.layers
        .map((name, k) => ({ k, name }))
        .filter(({ k }) => state.visible[k])
        .map(({ k, name }) => ({
          layer: k,
          name: short(name),
          value: `${bits(d.layerAgg(k).ixt[i])} , ${bits(d.layerAgg(k).ity[i])}`,
        }))
    );
  });
  hit.on("pointerleave", () => {
    state.hoverIdx = null;
    charts.forEach((c) => c.paintCross());
    tip.hide();
  });

  // Keyboard parity: the chart is focusable and the arrow keys walk the crosshair.
  host.tabIndex = 0;
  host.setAttribute("role", "application");
  host.setAttribute("aria-label", `${yLabel} against epoch. Arrow keys move the crosshair.`);
  host.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const cur = state.hoverIdx == null ? first : state.hoverIdx;
    state.hoverIdx = Math.max(first, Math.min(d.nEpochs - 1, cur + (e.key === "ArrowRight" ? 1 : -1)));
    charts.forEach((c) => c.paintCross());
    const r = host.getBoundingClientRect();
    const i = state.hoverIdx;
    tip.show(
      r.left + 56 + x(d.epochs[i]),
      r.top + 40,
      `Epoch ${d.epochs[i].toLocaleString("en-US")} — I(X;T) , I(T;Y)`,
      d.layers
        .map((name, k) => ({ k, name }))
        .filter(({ k }) => state.visible[k])
        .map(({ k, name }) => ({
          layer: k,
          name: short(name),
          value: `${bits(d.layerAgg(k).ixt[i])} , ${bits(d.layerAgg(k).ity[i])}`,
        }))
    );
  });

  return {
    paint() {
      const keepLabel = selectiveLabels(
        d.layers.map((_, k) => (state.visible[k] ? y(d.layerAgg(k)[key][d.nEpochs - 1]) : NaN))
      );
      d.layers.forEach((_, k) => {
        const on = state.visible[k];
        const emph = state.emphasis == null || state.emphasis === k;
        linePaths[k].attr("display", on ? null : "none").classed("dim", !emph);
        // A band per layer by default would be a stack of overlapping washes; show one
        // on legend hover, or all on explicit request.
        const showBand = on && (state.allBands || state.emphasis === k);
        bandPaths[k].attr("display", showBand ? null : "none");
        if (endText[k])
          endText[k].attr("display", on && keepLabel.has(k) ? null : "none").classed("dim", !emph);
      });
    },
    paintCross() {
      const i = state.hoverIdx;
      if (i == null) return cross.attr("display", "none");
      cross.attr("display", null).attr("x1", x(d.epochs[i])).attr("x2", x(d.epochs[i]));
    },
  };
}

function summaryTable(first) {
  tableView(document.getElementById("table"), {
    summary: `Table view — per-layer summary over ${d.nSeeds} seeds`,
    columns: [
      "Layer",
      "I(X;T) at epoch 1",
      "I(X;T) peak",
      "peak epoch",
      "I(X;T) final",
      "compressed",
      "I(T;Y) final",
    ],
    rows: d.layers.map((name, k) => {
      const a = d.layerAgg(k);
      let pi = first;
      for (let i = first; i < d.nEpochs; i++) if (a.ixt[i] > a.ixt[pi]) pi = i;
      return {
        layer: k,
        cells: [
          name,
          bits(a.ixt[first]),
          bits(a.ixt[pi]),
          d.epochs[pi].toLocaleString("en-US"),
          bits(a.ixt[d.nEpochs - 1]),
          bits(a.ixt[pi] - a.ixt[d.nEpochs - 1]),
          bits(a.ity[d.nEpochs - 1]),
        ],
      };
    }),
  });
}
