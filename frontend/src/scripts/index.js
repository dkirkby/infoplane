/** Overview page: the run's headline numbers, the accuracy context, and the
 *  caveats a reader needs before taking any bit value literally.
 */
import { line } from "d3-shape";
import { scaleLinear, scaleLog } from "d3-scale";
import { load, provenance } from "./data.js";
import { initTheme } from "./theme.js";
import { Tooltip, bits, frame, gridX, gridY, legend, pct, responsive, tableView, xAxis, yAxis } from "./chart.js";
import { format } from "d3-format";

const tip = new Tooltip();
initTheme(document.getElementById("theme"));

const TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000, 9000];
// Train and test accuracy are one measure on one scale, so they share a chart.
// They are not layers, so they do not take the ordinal layer ramp: slot 1 and
// slot 2 of the categorical palette, which clear every adjacent gate.
const SERIES = [
  { key: "train_acc", name: "Train accuracy", cls: "c1" },
  { key: "test_acc", name: "Test accuracy", cls: "c2" },
];

load("data/infoplane.json").then((d) => {
  document.getElementById("provenance").textContent = provenance(d);
  tiles(d);
  caveats(d);
  responsive(document.getElementById("acc"), (w) => buildAccuracy(d, w));
});

function tiles(d) {
  const host = document.getElementById("tiles");
  const g = d.diagnostics;
  const peakDrop = d.layers.reduce((best, name, k) => {
    const a = d.layerAgg(k);
    let p = 0;
    for (let i = 0; i < d.nEpochs; i++) if (a.ixt[i] > a.ixt[p]) p = i;
    return Math.max(best, a.ixt[p] - a.ixt[d.nEpochs - 1]);
  }, 0);

  const items = [
    { label: "Final test accuracy", value: pct(g.final_test_acc_mean), sub: `train ${pct(g.final_train_acc_mean)}` },
    { label: "Deepest compression", value: `${bits(peakDrop)} bits`, sub: "largest fall from a layer's peak I(X;T)" },
    { label: "Seeds", value: String(d.meta.seeds), sub: `${d.meta.epochs.toLocaleString("en-US")} epochs each` },
    { label: "Checkpoints", value: String(d.meta.checkpoints), sub: "log-spaced, epoch 0 included" },
    {
      label: "DPI violations",
      value: `${g.dpi_violations} / ${g.dpi_measurements.toLocaleString("en-US")}`,
      sub: g.dpi_violations === 0 ? "the ordering holds everywhere" : `worst ${bits(g.max_dpi_violation_bits)} bits`,
    },
  ];

  host.textContent = "";
  for (const it of items) {
    const el = document.createElement("div");
    el.className = "tile";
    const l = document.createElement("div");
    l.className = "label";
    l.textContent = it.label;
    const v = document.createElement("div");
    v.className = "value";
    v.textContent = it.value;
    const s = document.createElement("div");
    s.className = "sub";
    s.textContent = it.sub;
    el.append(l, v, s);
    host.append(el);
  }
}

function buildAccuracy(d, width) {
  const host = document.getElementById("acc");
  const height = Math.max(220, Math.min(300, Math.round(width * 0.32)));
  const { plot, w, h } = frame(host, { width, height, margin: { left: 54, right: 24, bottom: 42, top: 12 } });

  const first = d.epochs.findIndex((e) => e >= 1);
  const idx = d.epochs.map((_, i) => i).slice(first);
  const x = scaleLog([d.epochs[first], d.epochs[d.nEpochs - 1]], [0, w]);
  const y = scaleLinear([0, 1], [h, 0]);

  gridY(plot, y, w, 5);
  gridX(plot, x, h, TICKS);
  xAxis(plot, x, {
    w,
    h,
    label: "Training epoch (log scale)",
    ticks: TICKS,
    tickFormat: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)),
  });
  // ".0%" rather than ".1%": "100.0%" is wide enough to collide with the axis title.
  yAxis(plot, y, { h, label: "Accuracy", ticks: 5, tickFormat: format(".0%") });

  for (const s of SERIES) {
    const vals = d.curves[s.key].mean;
    plot
      .append("path")
      .attr("class", `series-line ${s.cls}`)
      .attr(
        "d",
        line()
          .x((i) => x(d.epochs[i]))
          .y((i) => y(vals[i]))(idx.filter((i) => vals[i] != null))
      );
  }

  // Direct-label both series at their end: only two, and they separate clearly.
  for (const s of SERIES) {
    const vals = d.curves[s.key].mean;
    plot
      .append("text")
      .attr("class", "point-label")
      .attr("x", w - 4)
      .attr("y", y(vals[d.nEpochs - 1]) + (s.key === "train_acc" ? -7 : 14))
      .attr("text-anchor", "end")
      .text(`${s.name} ${pct(vals[d.nEpochs - 1])}`);
  }

  const cross = plot.append("line").attr("class", "crosshair").attr("y1", 0).attr("y2", h).attr("display", "none");
  const hit = plot.append("rect").attr("width", w).attr("height", h).attr("fill", "transparent");
  hit.on("pointermove", (event) => {
    const r = hit.node().getBoundingClientRect();
    const target = x.invert(event.clientX - r.left);
    let best = first;
    for (const i of idx)
      if (Math.abs(Math.log(d.epochs[i]) - Math.log(target)) < Math.abs(Math.log(d.epochs[best]) - Math.log(target)))
        best = i;
    cross.attr("display", null).attr("x1", x(d.epochs[best])).attr("x2", x(d.epochs[best]));
    tip.show(
      event.clientX,
      event.clientY,
      `Epoch ${d.epochs[best].toLocaleString("en-US")}`,
      SERIES.map((s) => ({ name: s.name, value: pct(d.curves[s.key].mean[best]) }))
    );
  });
  hit.on("pointerleave", () => {
    cross.attr("display", "none");
    tip.hide();
  });

  // The legend defaults to the layer ramp's classes; these two series are
  // categorical, so swap in their slots.
  legend(document.getElementById("acc-legend"), SERIES.map((s) => s.name), { mark: "line" });
  document.querySelectorAll("#acc-legend .legend button").forEach((b, i) => {
    b.classList.remove("l1", "l2");
    b.classList.add(SERIES[i].cls);
  });

  tableView(document.getElementById("acc-table"), {
    summary: "Table view — accuracy at selected epochs",
    columns: ["Epoch", "Train accuracy", "Test accuracy"],
    rows: [1, 10, 35, 100, 1000, 9000]
      .map((e) => {
        let i = first;
        for (let k = first; k < d.nEpochs; k++) if (Math.abs(d.epochs[k] - e) < Math.abs(d.epochs[i] - e)) i = k;
        return i;
      })
      .filter((i, k, arr) => arr.indexOf(i) === k)
      .map((i) => ({
        cells: [
          d.epochs[i].toLocaleString("en-US"),
          pct(d.curves.train_acc.mean[i]),
          pct(d.curves.test_acc.mean[i]),
        ],
      })),
  });
}

function caveats(d) {
  const m = d.meta;
  document.getElementById("caveats").textContent =
    `Every bit value on these pages is a plug-in estimate from binning each tanh unit into ` +
    `${m.bins} intervals over [-1, 1]. That number is not incidental: the same network measured ` +
    `with 10 bins puts Layer 5 at 3.08 bits and with 120 bins at 7.52, so the shape of a ` +
    `trajectory is robust while its absolute height is a property of the measurement. Inputs are ` +
    `${m.encoding === "spin" ? "centred to ±1" : "the authors' raw 0/1"}; the authors' raw 0/1 inputs ` +
    `saturate the narrow layers and make the estimator violate the data processing inequality by up ` +
    `to 0.38 bits, which is why this run does not use them. I(X;T) is bounded by H(X) = ` +
    `${d.bounds.H_X} bits and I(T;Y) by H(Y) = ${bits(d.bounds.H_Y)} bits.`;
}
