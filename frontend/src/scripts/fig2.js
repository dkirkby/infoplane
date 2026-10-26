/** Figure 2 - the information plane, one mark per (seed, layer), at one epoch.
 *
 * This figure plots `runs`, not `aggregated`, on purpose. Phase 1 measured that
 * the fitting phase is seed-dependent and largely cancels in the ensemble mean
 * The paper's Figure 2 is itself a scatter of 50 networks, and the per-seed spread
 * is part of its content. (Under an earlier configuration the mean also actively
 * destroyed the subject: the arc was over in two epochs and cancelled to +0.49
 * bits. That is no longer so - it survives averaging at +1.28 - but the scatter
 * is still the faithful figure.)
 *
 * Two things here depart from the rest of the site, both because this is the one
 * SCATTER of layers rather than a set of lines:
 *
 * - Series are identified CATEGORICALLY (colour + shape), not by the ordinal blue
 *   ramp the other figures use. In a scatter the reader has to tell any two marks
 *   apart, not just neighbours, and adjacent steps of one hue cannot do that when
 *   the marks interleave. The four hues are catalogue slots 1/4/5/6, the subset
 *   that clears the all-pairs gate in both themes; the shapes are the secondary
 *   channel its dark-mode 6.9 CVD separation obliges. See main.css.
 * - Layers pinned at the H(X) ceiling are left out of the plot. They cannot move -
 *   a layer whose binning separates all 4096 inputs is invertible in X, so it sits
 *   at (H(X), H(Y)) from initialisation - and two immovable stacks of 50 marks at
 *   the corner crowd out the layers the figure is about. They stay in the table.
 */
import { line, symbol, symbolCircle, symbolDiamond, symbolSquare, symbolTriangle } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { load, nearest, provenance } from "./data.js";
import { initTheme } from "./theme.js";
import { scrubber } from "./scrubber.js";
import {
  bits,
  frame,
  gridX,
  gridY,
  legend,
  responsive,
  selectiveLabels,
  short,
  tableView,
  xAxis,
  yAxis,
} from "./chart.js";

// Categorical slots, in catalogue order, with a shape each. Never cycled: a fifth
// plotted series would need the palette re-validated, not a repeat of slot 1.
const SERIES = [
  { cls: "c1", key: "dot-key", sym: symbolCircle },
  { cls: "c4", key: "sq-key", sym: symbolSquare },
  { cls: "c5", key: "tri-key", sym: symbolTriangle },
  { cls: "c6", key: "dia-key", sym: symbolDiamond },
];

initTheme(document.getElementById("theme"));

const state = { epochIdx: 0, focusSeed: null, focusMean: null, focusLayer: null, visible: null };
let d = null;
let shown = []; // indices into d.layers that are actually plotted
let update = () => {};

/** Layers that never leave the I(X;T) ceiling, found from the data rather than
 * named: the architecture has changed twice in this project's life and a
 * hardcoded "drop L1 and L2" would quietly drop the wrong ones next time. */
function pinnedLayers(data) {
  return data.layers
    .map((_, li) => li)
    .filter((li) => {
      const t = data.layerAgg(li).ixt;
      const lo = Math.min(...t);
      return data.bounds.H_X - lo < 0.05; // never falls more than 0.05 bits below H(X)
    });
}

load("data/infoplane.json").then((data) => {
  d = data;
  const pinned = new Set(pinnedLayers(d));
  shown = d.layers.map((_, li) => li).filter((li) => !pinned.has(li));
  if (shown.length > SERIES.length) shown = shown.slice(-SERIES.length);
  state.visible = shown.map(() => true);
  state.focusLayer = shown[shown.length - 1];

  document.getElementById("provenance").textContent = provenance(d);
  document.getElementById("binsnote").textContent = String(d.meta.bins);
  // Written from the data, not hardcoded: the series count is a property of the
  // run (--widths, --no-measure-output), and a stale "7 layers x 50 seeds" here
  // outlived two architectures.
  const omitted = [...pinned].map((li) => short(d.layers[li]));
  document.getElementById("markcount").textContent =
    `${shown.length * d.nSeeds} marks: ${shown.length} series × ${d.nSeeds} seeds. ` +
    `Large labelled marks are the ensemble mean. ` +
    (omitted.length
      ? `${omitted.join(" and ")} ${omitted.length > 1 ? "are" : "is"} pinned at the ` +
        `H(X) = ${d.bounds.H_X}-bit ceiling and cannot move, so ${omitted.length > 1 ? "they are" : "it is"} ` +
        `left to the table view below.`
      : "");

  legend(document.getElementById("legend"), shown.map((li) => short(d.layers[li])), {
    classFor: (i) => SERIES[i].cls,
    keyClassFor: (i) => SERIES[i].key,
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
  const readout = document.getElementById("focus-readout");
  const height = Math.max(380, Math.min(560, Math.round(width * 0.62)));
  const { plot, w, h } = frame(host, { width, height, margin: { left: 54, right: 24, top: 14 } });

  const x = scaleLinear([0, d.bounds.H_X], [0, w]);
  const y = scaleLinear([0, d.bounds.H_Y], [h, 0]).nice();

  gridY(plot, y, w, 5);
  gridX(plot, x, h, 6);
  xAxis(plot, x, { w, h, label: "I(X;T)  —  bits the layer keeps about the input", ticks: 6 });
  yAxis(plot, y, { h, label: "I(T;Y)  —  bits about the label", ticks: 5, tickFormat: bits });

  // The ceilings, as solid hairlines: I(T;Y) cannot exceed H(Y), and I(X;T)
  // cannot exceed H(X). Drawing them stops the reader wondering how far there is
  // to go, and shows where the omitted pinned layers sit.
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

  const pathLayer = plot.append("g");
  const traceLayer = plot.append("g");
  const dotLayer = plot.append("g");
  const meanLayer = plot.append("g");

  // One mark per (seed, series), created once and moved on every update: the
  // scrub path does no DOM churn. Paths rather than circles, because the shape
  // carries identity alongside the colour.
  const keys = [];
  for (let s = 0; s < d.nSeeds; s++)
    for (let si = 0; si < shown.length; si++) keys.push({ s, si, li: shown[si] });

  const markPath = (si, size) => symbol().type(SERIES[si].sym).size(size)();

  const dots = dotLayer
    .selectAll("path")
    .data(keys, (k) => `${k.s}-${k.li}`)
    .join("path")
    .attr("class", (k) => `dot ${SERIES[k.si].cls}`)
    .attr("d", (k) => markPath(k.si, 44));

  const meanDots = meanLayer
    .selectAll("path")
    .data(shown.map((_, si) => si))
    .join("path")
    .attr("class", (si) => `dot ${SERIES[si].cls}`)
    .attr("d", (si) => markPath(si, 120));

  // Direct labels on the ensemble means only - selective by design. A value
  // beside every mark would be unreadable; the table carries the rest. They are
  // also what discharges the relief rule for the two low-contrast light-mode
  // hues: identity never rests on colour alone here.
  const meanLabels = meanLayer
    .selectAll("text")
    .data(shown.map((_, si) => si))
    .join("text")
    .attr("class", "point-label lg")
    .text((si) => short(d.layers[shown[si]]));

  const trace = traceLayer.append("path").attr("class", "trace").attr("display", "none");
  // One path per plotted series, created once: hovering a seed lights exactly one
  // of them, hovering an ensemble mean lights them all. Every halo is appended
  // before any coloured path, so where two paths cross, neither is broken by the
  // other's halo.
  const epochHalos = shown.map(() =>
    pathLayer.append("path").attr("class", "epoch-path-halo").attr("display", "none")
  );
  const epochPaths = shown.map((_, si) =>
    pathLayer.append("path").attr("class", `epoch-path ${SERIES[si].cls}`).attr("display", "none")
  );
  const hidePaths = () => {
    epochHalos.forEach((n) => n.attr("display", "none"));
    epochPaths.forEach((n) => n.attr("display", "none"));
  };

  const path = line()
    .x((p) => x(p[0]))
    .y((p) => y(p[1]));

  const marks = []; // pixel positions for nearest-point hover
  const meanMarks = []; // the same, for the ensemble-mean markers

  // Hit layer covers the plot so the pointer only has to be *closest* to a mark,
  // never dead-centre on it.
  const hit = plot
    .append("rect")
    .attr("width", w)
    .attr("height", h)
    .attr("fill", "transparent")
    .style("cursor", "crosshair");

  hit.on("pointermove", (event) => {
    const [px, py] = pointer(event, hit.node());
    // Means are tested first, and with a tight radius: they sit inside the cloud
    // they summarise, so some seed dot is nearly always closer, but the big
    // labelled mark is what the pointer is aiming at when it is over one.
    const m = nearest(meanMarks, px, py, 14);
    if (m) return focusMeanOn(m.si);
    const k = nearest(marks, px, py, 30);
    if (!k) return clearFocus();
    focusOn(k.s, k.li);
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
      focusOn((cur + dir + d.nSeeds) % d.nSeeds, state.focusLayer);
      e.preventDefault();
    } else if (e.key === "Escape") clearFocus();
  });

  // What is on screen is said once, statically, above the chart. Only the
  // screen-reader line names the thing currently under the pointer, because it
  // is the one readout that costs no layout.
  const announce = (text) => {
    readout.textContent = text;
  };

  /** Highlight one (seed, layer). */
  function focusOn(s, li) {
    state.focusSeed = s;
    state.focusMean = null;
    state.focusLayer = li;
    paintFocus();
    announce(
      `Seed ${d.seeds[s]}, ${short(d.layers[li])}, at epoch ` +
        `${d.epochs[state.epochIdx].toLocaleString("en-US")}.`
    );
  }

  /** Highlight one layer's ensemble mean, `si` indexing the plotted series. */
  function focusMeanOn(si) {
    state.focusSeed = null;
    state.focusMean = si;
    state.focusLayer = shown[si];
    paintFocus();
    announce(
      `Ensemble mean of ${short(d.layers[shown[si]])} over ${d.nSeeds} seeds, at epoch ` +
        `${d.epochs[state.epochIdx].toLocaleString("en-US")}.`
    );
  }

  function clearFocus() {
    state.focusSeed = null;
    state.focusMean = null;
    paintFocus();
    announce("");
  }

  function paintFocus() {
    const s = state.focusSeed;
    const mi = state.focusMean;
    // Focusing a mean dims every seed, since no single network is being shown.
    // Nothing else dims: hovering any mean puts the whole ensemble on screen, so
    // the other layers' means are the subject too, not context.
    dots.classed("dim", (k) => mi != null || (s != null && k.s !== s));
    meanDots.classed("dim", false);
    meanLabels.classed("dim", false);
    if (s == null && mi == null) {
      trace.attr("display", "none");
      hidePaths();
      return;
    }
    const e = state.epochIdx;
    // The chain across layers at this epoch -- one seed's, or the ensemble's.
    // Either way it is the data processing chain, which must march leftward as
    // depth increases.
    const at = (li) =>
      mi != null
        ? [d.layerAgg(li).ixt[e], d.layerAgg(li).ity[e]]
        : [d.seedPts[s][li].ixt[e], d.seedPts[s][li].ity[e]];
    const chain = shown.filter((_, si) => state.visible[si]).map(at);
    trace.attr("display", null).attr("d", path(chain));

    // ...and the paths through every epoch, which are the arcs the paper's figure
    // is about: right during fitting, then left as it compresses. Hovering a seed
    // draws that one layer's own path; hovering a mean draws the mean path of
    // every layer, which is the comparison that mark is for.
    const lit = new Set(
      mi != null
        ? shown.map((_, si) => si).filter((si) => state.visible[si])
        : [shown.indexOf(state.focusLayer)].filter((si) => si >= 0 && state.visible[si])
    );
    epochPaths.forEach((node, si) => {
      if (!lit.has(si)) {
        node.attr("display", "none");
        epochHalos[si].attr("display", "none");
        return;
      }
      const src = mi != null ? d.layerAgg(shown[si]) : d.seedPts[s][shown[si]];
      const dAttr = path(d.epochs.map((_, i) => [src.ixt[i], src.ity[i]]));
      const heavy = mi != null ? " mean" : "";
      epochHalos[si].attr("display", null).attr("class", `epoch-path-halo${heavy}`).attr("d", dAttr);
      node.attr("display", null).attr("class", `epoch-path ${SERIES[si].cls}${heavy}`).attr("d", dAttr);
    });
  }

  return function render() {
    const e = state.epochIdx;
    marks.length = 0;
    dots
      .attr("transform", (k) => {
        k._px = x(d.seedPts[k.s][k.li].ixt[e]);
        k._py = y(d.seedPts[k.s][k.li].ity[e]);
        return `translate(${k._px},${k._py})`;
      })
      .attr("display", (k) => (state.visible[k.si] ? null : "none"))
      .attr("opacity", (k) =>
        state.emphasis == null || state.emphasis === k.si ? 0.75 : 0.12
      );
    for (const k of keys) if (state.visible[k.si]) marks.push({ s: k.s, li: k.li, px: k._px, py: k._py });

    meanMarks.length = 0;
    meanDots
      .attr("transform", (si) => {
        const a = d.layerAgg(shown[si]);
        const px = x(a.ixt[e]);
        const py = y(a.ity[e]);
        if (state.visible[si]) meanMarks.push({ si, px, py });
        return `translate(${px},${py})`;
      })
      .attr("display", (si) => (state.visible[si] ? null : "none"));
    // Means can still collide near the ceiling, so only label the ones with room.
    const keep = selectiveLabels(
      shown.map((li, si) => (state.visible[si] ? y(d.layerAgg(li).ity[e]) : NaN))
    );
    meanLabels
      .attr("x", (si) => x(d.layerAgg(shown[si]).ixt[e]) + 10)
      .attr("y", (si) => y(d.layerAgg(shown[si]).ity[e]) + 4)
      .attr("display", (si) => (state.visible[si] && keep.has(si) ? null : "none"));

    paintFocus();

    // The table keeps EVERY layer, including the pinned ones left out of the
    // plot: this is the path that has to show the whole measurement.
    tableView(document.getElementById("table"), {
      summary: `Table view — ensemble at epoch ${d.epochs[e].toLocaleString("en-US")} (mean ± sd over ${d.nSeeds} seeds, all layers)`,
      columns: ["Layer", "I(X;T) mean", "I(X;T) sd", "I(T;Y) mean", "I(T;Y) sd"],
      swatchFor: (si) => SERIES[si].cls,
      rows: d.layers.map((name, li) => {
        const a = d.layerAgg(li);
        const si = shown.indexOf(li);
        return {
          layer: si >= 0 ? si : null,
          cells: [
            si >= 0 ? name : `${name} (pinned, not plotted)`,
            bits(a.ixt[e]),
            bits(a.ixtSd[e]),
            bits(a.ity[e]),
            bits(a.itySd[e]),
          ],
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
