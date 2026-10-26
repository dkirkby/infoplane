/** Figure 5 - the plane at four training-set sizes, as small multiples.
 *
 * Faceted rather than overlaid. These are scatter panels, where any two marks can
 * be neighbours, and under that condition a categorical palette only clears the
 * colourblind-separation floor for three series - so four overlaid sizes could not
 * be coloured safely. Faceting leaves colour doing its one job here: depth, on the
 * validated ordinal ramp, identically in every panel.
 *
 * Each size is a separate Phase 1 export (~1.9 MB), fetched only when its panel is
 * switched on.
 */
import { line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import { load, loadSweepIndex } from "./data.js";
import { initTheme } from "./theme.js";
import { scrubber } from "./scrubber.js";
import { bits, frame, gridX, gridY, lclass, legend, responsive, short, tableView, xAxis, yAxis } from "./chart.js";

initTheme(document.getElementById("theme"));

const state = { epochIdx: 0, emphasis: null, visible: null, on: new Set(), width: 480 };
const sets = new Map(); // trainSize -> prepared dataset
const panels = new Map(); // trainSize -> { card, update }
let manifest = [];
let scrub = null;

loadSweepIndex().then(async (sizes) => {
  manifest = sizes;
  sizes.forEach((s) => state.on.add(s.train_size));

  const group = document.getElementById("sizes");
  for (const s of sizes) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "toggle";
    b.setAttribute("aria-pressed", "true");
    b.textContent = `N = ${s.train_size.toLocaleString("en-US")}`;
    b.addEventListener("click", async () => {
      if (state.on.has(s.train_size)) {
        if (state.on.size === 1) return; // never leave the page empty
        state.on.delete(s.train_size);
      } else {
        state.on.add(s.train_size);
      }
      b.setAttribute("aria-pressed", String(state.on.has(s.train_size)));
      await ensureLoaded();
      rebuild();
    });
    group.append(b);
  }

  await ensureLoaded();

  const any = sets.values().next().value;
  state.visible = any.layers.map(() => true);
  legend(document.getElementById("legend"), any.layers.map(short), {
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

  scrub = scrubber({
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

async function ensureLoaded() {
  const wanted = manifest.filter((m) => state.on.has(m.train_size) && !sets.has(m.train_size));
  if (!wanted.length) return;
  const host = document.getElementById("panels");
  host.classList.add("loading"); // hold the previous render, no skeleton
  await Promise.all(
    wanted.map(async (m) => {
      sets.set(m.train_size, await load(`data/trainsize/${m.file}`));
    })
  );
  host.classList.remove("loading");
}

function rebuild() {
  const host = document.getElementById("panels");
  const active = manifest.filter((m) => state.on.has(m.train_size));

  // Drop panels that are off, keep the rest in manifest order.
  for (const [size, p] of panels) {
    if (!state.on.has(size)) {
      p.card.remove();
      panels.delete(size);
    }
  }
  host.textContent = "";
  panels.clear();

  // Two passes on purpose. The CSS grid decides the column count from how many
  // cards are in it, so a card measured while it is still the only one reports
  // the full grid width and its chart would then be built too large and scaled
  // down. Append every card first, let the grid settle, and only then measure.
  const built = active.map((m) => {
    const card = document.createElement("section");
    card.className = "card";
    const head = document.createElement("div");
    head.className = "card-head";
    const left = document.createElement("div");
    const h2 = document.createElement("h2");
    h2.textContent = `N = ${m.train_size.toLocaleString("en-US")} training samples`;
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = `final test accuracy ${m.final_test_acc.toFixed(4)} · train ${m.final_train_acc.toFixed(4)}`;
    left.append(h2, note);
    head.append(left);
    const plot = document.createElement("div");
    plot.className = "plot";
    const table = document.createElement("div");
    card.append(head, plot, table);
    host.append(card);
    return { m, card, plot, table };
  });

  for (const { m, card, plot, table } of built) {
    const panelW = Math.max(240, Math.floor(plot.getBoundingClientRect().width));
    panels.set(m.train_size, {
      card,
      update: buildPanel(plot, table, sets.get(m.train_size), panelW, m),
    });
  }
  paint();
}

function paint() {
  for (const p of panels.values()) p.update();
}

function buildPanel(host, tableHost, d, width, meta) {
  const height = Math.max(240, Math.round(width * 0.72));
  const { plot, w, h } = frame(host, {
    width,
    height,
    margin: { left: 48, right: 18, bottom: 40, top: 12 },
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
    .attr("x", w)
    .attr("y", y(d.bounds.H_Y) - 4)
    .attr("text-anchor", "end")
    .text("H(Y)");

  const trailLayer = plot.append("g");
  const markLayer = plot.append("g");
  const mkPath = (k) =>
    line()
      .x((i) => x(d.layerAgg(k).ixt[i]))
      .y((i) => y(d.layerAgg(k).ity[i]))(d.epochs.map((_, i) => i));

  const trails = d.layers.map((_, k) =>
    trailLayer.append("path").attr("class", `series-line ${lclass(k)}`).attr("opacity", 0.3).attr("d", mkPath(k))
  );
  const marks = d.layers.map((_, k) => markLayer.append("circle").attr("class", `dot ${lclass(k)}`).attr("r", 5.5));

  return function render() {
    const e = Math.min(state.epochIdx, d.nEpochs - 1);
    d.layers.forEach((_, k) => {
      const on = state.visible[k];
      const emph = state.emphasis == null || state.emphasis === k;
      trails[k].attr("display", on ? null : "none").classed("dim", !emph);
      marks[k]
        .attr("cx", x(d.layerAgg(k).ixt[e]))
        .attr("cy", y(d.layerAgg(k).ity[e]))
        .attr("display", on ? null : "none")
        .classed("dim", !emph);
    });

    tableView(tableHost, {
      summary: `Table view — N = ${meta.train_size} at epoch ${d.epochs[e].toLocaleString("en-US")}`,
      columns: ["Layer", "I(X;T)", "I(T;Y)"],
      rows: d.layers.map((name, k) => ({
        layer: k,
        cells: [name, bits(d.layerAgg(k).ixt[e]), bits(d.layerAgg(k).ity[e])],
      })),
    });
  };
}

function accuracyTable() {
  tableView(document.getElementById("acc-table"), {
    summary: "Table view — accuracy by training-set size",
    columns: ["Training size", "Final train accuracy", "Final test accuracy", "Generalisation gap"],
    rows: manifest.map((m) => ({
      cells: [
        m.train_size.toLocaleString("en-US"),
        m.final_train_acc.toFixed(4),
        m.final_test_acc.toFixed(4),
        (m.final_train_acc - m.final_test_acc).toFixed(4),
      ],
    })),
  });
}
