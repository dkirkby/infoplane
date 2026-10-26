/** Figure 2 - the information plane, one mark per (seed, layer), at one epoch.
 *
 * This figure plots `runs`, not `aggregated`, on purpose. Phase 1 measured that
 * the fitting phase is seed-dependent and largely cancels in the ensemble mean
 * The paper's Figure 2 is itself a scatter of 50 networks, and the per-seed spread
 * is part of its content. (Under an earlier configuration the mean also actively
 * destroyed the subject: the arc was over in two epochs and cancelled to +0.49
 * bits. That is no longer so - it survives averaging at +1.39 - but the scatter
 * is still the faithful figure.)
 */
import { line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { load, nearest, provenance } from "./data.js";
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

const state = { epochIdx: 0, focusSeed: null, visible: null };
let d = null;
let update = () => {};

load("data/infoplane.json").then((data) => {
  d = data;
  state.visible = d.layers.map(() => true);
  document.getElementById("provenance").textContent = provenance(d);
  document.getElementById("binsnote").textContent = String(d.meta.bins);
  // Written from the data, not hardcoded: the layer count is a property of the run
  // (--widths), and a stale "7 layers x 50 seeds" here outlived two architectures.
  document.getElementById("markcount").textContent =
    `${d.layers.length * d.nSeeds} marks: ${d.layers.length} layers \u00d7 ${d.nSeeds} seeds. ` +
    "Large labelled marks are the ensemble mean per layer.";

  legend(document.getElementById("legend"), d.layers.map(short), {
    mark: "dot",
    onEmphasis: (i) => {
      state.emphasis = i;
      update();
    },
    onToggle: (vis) => {
      state.visible = vis;
      update();
    },
  });

  const scrub = scrubber({
    slider: document.getElementById("epoch"),
    readout: document.getElementById("epoch-readout"),
    playButton: document.getElementById("play"),
    epochs: d.epochs,
    onChange: (i) => {
      state.epochIdx = i;
      update();
    },
  });

  responsive(document.getElementById("plot"), (w) => {
    update = build(w, scrub);
    update();
  });
});

function build(width, scrub) {
  const host = document.getElementById("plot");
  const height = Math.max(380, Math.min(560, Math.round(width * 0.62)));
  const { svg, plot, w, h } = frame(host, { width, height, margin: { left: 54, right: 24, top: 14 } });

  const x = scaleLinear([0, d.bounds.H_X], [0, w]);
  const y = scaleLinear([0, d.bounds.H_Y], [h, 0]).nice();

  gridY(plot, y, w, 5);
  gridX(plot, x, h, 6);
  xAxis(plot, x, { w, h, label: "I(X;T)  —  bits the layer keeps about the input", ticks: 6 });
  yAxis(plot, y, { h, label: "I(T;Y)  —  bits about the label", ticks: 5, tickFormat: bits });

  // The ceilings, as solid hairlines: I(T;Y) cannot exceed H(Y), and I(X;T)
  // cannot exceed H(X). Drawing them stops the reader wondering how far there is
  // to go, and makes "Layer 1 is pinned at the ceiling" legible.
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
  // Along the bottom, not the top: up there it would sit on the H(Y) label.
  plot
    .append("text")
    .attr("class", "bound-label")
    .attr("x", x(d.bounds.H_X) - 5)
    .attr("y", h - 7)
    .attr("text-anchor", "end")
    .text(`H(X) = ${d.bounds.H_X} bits`);

  const traceLayer = plot.append("g");
  const dotLayer = plot.append("g");
  const meanLayer = plot.append("g");

  // One circle per (seed, layer), created once and moved on every update: the
  // scrub path does no DOM churn.
  const keys = [];
  for (let s = 0; s < d.nSeeds; s++)
    for (let li = 0; li < d.layers.length; li++) keys.push({ s, li });

  const dots = dotLayer
    .selectAll("circle")
    .data(keys, (k) => `${k.s}-${k.li}`)
    .join("circle")
    .attr("class", (k) => `dot ${lclass(k.li)}`)
    .attr("r", 4);

  const meanDots = meanLayer
    .selectAll("circle")
    .data(d.layers.map((_, li) => li))
    .join("circle")
    .attr("class", (li) => `dot ${lclass(li)}`)
    .attr("r", 6.5);

  // Direct labels on the ensemble means only - selective by design. A value
  // beside every mark would be unreadable; the tooltip and table carry the rest.
  const meanLabels = meanLayer
    .selectAll("text")
    .data(d.layers.map((_, li) => li))
    .join("text")
    .attr("class", "point-label")
    .text((li) => short(d.layers[li]));

  const trace = traceLayer.append("path").attr("class", "trace").attr("display", "none");
  const path = line()
    .x((p) => x(p[0]))
    .y((p) => y(p[1]));

  const marks = []; // pixel positions for nearest-point hover

  // Hit layer covers the plot so the pointer only has to be *closest* to a mark,
  // never dead-centre on a 8px dot.
  const hit = plot
    .append("rect")
    .attr("width", w)
    .attr("height", h)
    .attr("fill", "transparent")
    .style("cursor", "crosshair");

  hit.on("pointermove", (event) => {
    const [px, py] = pointer(event, hit.node());
    const m = nearest(marks, px, py, 30);
    if (!m) return clearFocus();
    focusSeed(m.s, event.clientX, event.clientY);
  });
  hit.on("pointerleave", clearFocus);

  host.tabIndex = 0;
  host.setAttribute("role", "application");
  host.setAttribute("aria-label", "Information plane scatter. Arrow keys change epoch; up and down cycle seeds.");
  host.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") scrub.set(scrub.index + 1), e.preventDefault();
    else if (e.key === "ArrowLeft") scrub.set(scrub.index - 1), e.preventDefault();
    else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const dir = e.key === "ArrowUp" ? -1 : 1;
      const cur = state.focusSeed == null ? -1 : state.focusSeed;
      focusSeedKeyboard((cur + dir + d.nSeeds) % d.nSeeds);
      e.preventDefault();
    } else if (e.key === "Escape") clearFocus();
  });

  function focusSeed(s, clientX, clientY) {
    state.focusSeed = s;
    paintFocus();
    const rows = d.layers.map((name, li) => ({
      layer: li,
      name: short(name),
      value: `${bits(d.seedPts[s][li].ixt[state.epochIdx])} , ${bits(d.seedPts[s][li].ity[state.epochIdx])}`,
    }));
    tip.show(clientX, clientY, `Seed ${d.seeds[s]} — I(X;T) , I(T;Y)`, rows);
  }

  function focusSeedKeyboard(s) {
    const box = host.getBoundingClientRect();
    state.focusSeed = s;
    paintFocus();
    const m = marks.find((k) => k.s === s && k.li === d.layers.length - 1);
    focusSeed(s, box.left + (m ? m.px + 54 : box.width / 2), box.top + (m ? m.py + 14 : box.height / 2));
  }

  function clearFocus() {
    state.focusSeed = null;
    paintFocus();
    tip.hide();
  }

  function paintFocus() {
    const s = state.focusSeed;
    dots.classed("dim", (k) => s != null && k.s !== s);
    if (s == null) return trace.attr("display", "none");
    // The trajectory through this seed's layers at this epoch: the data
    // processing chain, which must march leftward as depth increases.
    const pts = d.layers
      .map((_, li) => [d.seedPts[s][li].ixt[state.epochIdx], d.seedPts[s][li].ity[state.epochIdx]])
      .filter((_, li) => state.visible[li]);
    trace.attr("display", null).attr("d", path(pts));
  }

  return function render() {
    const e = state.epochIdx;
    marks.length = 0;
    dots
      .attr("cx", (k) => {
        const px = x(d.seedPts[k.s][k.li].ixt[e]);
        k._px = px;
        return px;
      })
      .attr("cy", (k) => {
        const py = y(d.seedPts[k.s][k.li].ity[e]);
        k._py = py;
        return py;
      })
      .attr("display", (k) => (state.visible[k.li] ? null : "none"))
      .attr("opacity", (k) =>
        state.emphasis == null || state.emphasis === k.li ? 0.75 : 0.12
      );
    for (const k of keys) if (state.visible[k.li]) marks.push({ s: k.s, li: k.li, px: k._px, py: k._py });

    meanDots
      .attr("cx", (li) => x(d.layerAgg(li).ixt[e]))
      .attr("cy", (li) => y(d.layerAgg(li).ity[e]))
      .attr("display", (li) => (state.visible[li] ? null : "none"));
    // Layers 1-3 sit on top of one another at the ceiling for most of training,
    // so only label the ones that have room.
    const keep = selectiveLabels(
      d.layers.map((_, li) => (state.visible[li] ? y(d.layerAgg(li).ity[e]) : NaN))
    );
    meanLabels
      .attr("x", (li) => x(d.layerAgg(li).ixt[e]) + 10)
      .attr("y", (li) => y(d.layerAgg(li).ity[e]) + 4)
      .attr("display", (li) => (state.visible[li] && keep.has(li) ? null : "none"));

    paintFocus();

    tableView(document.getElementById("table"), {
      summary: `Table view — ensemble at epoch ${d.epochs[e].toLocaleString("en-US")} (mean ± sd over ${d.nSeeds} seeds)`,
      columns: ["Layer", "I(X;T) mean", "I(X;T) sd", "I(T;Y) mean", "I(T;Y) sd"],
      rows: d.layers.map((name, li) => {
        const a = d.layerAgg(li);
        return {
          layer: li,
          cells: [name, bits(a.ixt[e]), bits(a.ixtSd[e]), bits(a.ity[e]), bits(a.itySd[e])],
        };
      }),
    });
  };
}

/** Pointer position relative to an element, without pulling in d3-selection's. */
function pointer(event, node) {
  const r = node.getBoundingClientRect();
  return [event.clientX - r.left, event.clientY - r.top];
}
