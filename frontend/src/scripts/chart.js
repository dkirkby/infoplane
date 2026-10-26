/** Shared chart furniture: frames, axes, legends, tooltips, table views.
 *
 * Everything here is deliberately plain: the figures own their marks, this module
 * owns the chrome so that it is identical across pages. Colour never appears as a
 * literal - marks get a `l1`..`l7` class and pick up `--c` from the stylesheet, so
 * a theme switch repaints without a re-render and a layer's colour can never
 * follow its rank.
 */
import { axisBottom, axisLeft } from "d3-axis";
import { format } from "d3-format";
import { select } from "d3-selection";

export const bits = format(".2f");
export const pct = format(".1%");
export const int = format(",d");

/** Layer index (0-based) -> the class that carries its ordinal colour. */
export const lclass = (i) => `l${i + 1}`;

/** Short label for a layer, e.g. "Layer 3" -> "L3". */
export const short = (name) => name.replace(/^Layer\s*/, "L");

/**
 * Build a fresh SVG frame sized to `width`, clearing whatever was there.
 * Returns the plot group plus inner dimensions.
 */
export function frame(container, { width, height, margin }) {
  const m = { top: 12, right: 16, bottom: 38, left: 48, ...margin };
  const w = Math.max(80, width - m.left - m.right);
  const h = Math.max(80, height - m.top - m.bottom);
  container.textContent = "";
  const svg = select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", width)
    .attr("height", height)
    .attr("role", "img");
  const plot = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
  return { svg, plot, w, h, m };
}

/** Horizontal gridlines at the y scale's ticks. Hairline, solid, recessive. */
export function gridY(plot, y, w, ticks = 5) {
  plot
    .append("g")
    .attr("class", "grid")
    .selectAll("line")
    .data(y.ticks(ticks))
    .join("line")
    .attr("x1", 0)
    .attr("x2", w)
    .attr("y1", (d) => y(d))
    .attr("y2", (d) => y(d));
}

/** Vertical gridlines; `ticks` may be a count or an explicit array. */
export function gridX(plot, x, h, ticks = 5) {
  const vals = Array.isArray(ticks) ? ticks : x.ticks(ticks);
  plot
    .append("g")
    .attr("class", "grid")
    .selectAll("line")
    .data(vals)
    .join("line")
    .attr("y1", 0)
    .attr("y2", h)
    .attr("x1", (d) => x(d))
    .attr("x2", (d) => x(d));
}

export function xAxis(plot, x, { w, h, label, ticks, tickFormat }) {
  const ax = axisBottom(x).tickSizeOuter(0);
  if (Array.isArray(ticks)) ax.tickValues(ticks);
  else if (ticks) ax.ticks(ticks);
  if (tickFormat) ax.tickFormat(tickFormat);
  plot.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(ax);
  if (label)
    plot
      .append("text")
      .attr("class", "axis-title")
      .attr("x", w / 2)
      .attr("y", h + 32)
      .attr("text-anchor", "middle")
      .text(label);
}

export function yAxis(plot, y, { h, label, ticks = 5, tickFormat }) {
  const ax = axisLeft(y).ticks(ticks).tickSizeOuter(0);
  if (tickFormat) ax.tickFormat(tickFormat);
  plot.append("g").attr("class", "axis").call(ax);
  if (label)
    plot
      .append("text")
      .attr("class", "axis-title")
      .attr("transform", "rotate(-90)")
      .attr("x", -h / 2)
      .attr("y", -36)
      .attr("text-anchor", "middle")
      .text(label);
}

/**
 * Legend. Always rendered for two or more series - identity is never carried by
 * colour alone. Each entry is a real button: hovering or focusing it emphasises
 * that layer, clicking toggles it, so keyboard users get the same affordance as
 * the pointer.
 */
export function legend(
  container,
  names,
  { onEmphasis, onToggle, mark = "line", classFor = lclass, keyClassFor = null } = {}
) {
  container.textContent = "";
  const ul = document.createElement("ul");
  ul.className = "legend";
  const state = names.map(() => true);
  const buttons = names.map((name, i) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.className = classFor(i);
    b.setAttribute("aria-pressed", "true");
    const key = document.createElement("span");
    // keyClassFor lets a chart repeat its mark SHAPE in the legend, which is what
    // makes a categorical palette legible without relying on colour alone.
    key.className = keyClassFor ? `key ${keyClassFor(i)}` : mark === "dot" ? "key dot-key" : "key";
    b.append(key, document.createTextNode(name));
    if (onEmphasis) {
      b.addEventListener("pointerenter", () => onEmphasis(i));
      b.addEventListener("focus", () => onEmphasis(i));
      b.addEventListener("pointerleave", () => onEmphasis(null));
      b.addEventListener("blur", () => onEmphasis(null));
    }
    b.addEventListener("click", () => {
      if (!onToggle) return;
      // Never let the last series be switched off: an empty chart is not a state
      // the reader can recover from without guessing.
      if (state[i] && state.filter(Boolean).length === 1) return;
      state[i] = !state[i];
      b.setAttribute("aria-pressed", String(state[i]));
      onToggle(state.slice());
    });
    li.append(b);
    ul.append(li);
    return b;
  });
  container.append(ul);
  return { state, buttons };
}

/** One shared tooltip element per page. Values lead; names follow. */
export class Tooltip {
  constructor() {
    this.el = document.createElement("div");
    this.el.className = "tip";
    this.el.setAttribute("role", "status");
    document.body.append(this.el);
  }

  /** `rows` are {layer, name, value}; `layer` indexes the ordinal ramp. */
  show(x, y, title, rows) {
    const el = this.el;
    el.textContent = "";
    if (title) {
      const h = document.createElement("h3");
      h.textContent = title; // untrusted label -> textContent, never innerHTML
      el.append(h);
    }
    const table = document.createElement("table");
    for (const r of rows) {
      const tr = document.createElement("tr");
      const k = document.createElement("td");
      k.className = "tkey";
      if (r.layer != null) {
        k.classList.add(lclass(r.layer));
        k.append(document.createElement("i"));
      }
      const n = document.createElement("td");
      n.className = "tname";
      n.textContent = r.name;
      const v = document.createElement("td");
      v.className = "tval";
      v.textContent = r.value;
      tr.append(k, n, v);
      table.append(tr);
    }
    el.append(table);
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.dataset.show = "1";
  }

  hide() {
    this.el.dataset.show = "0";
  }
}

/**
 * The table-view twin. Every chart ships one, so no value is reachable only by
 * hovering and the page still works in forced-colors or print.
 */
export function tableView(
  container,
  { summary = "Table view", columns, rows, swatchFor = lclass }
) {
  // The table is rebuilt on every render (scrub, toggle, focus), so carry the
  // disclosure state across: a reader who opened it should not have it snap shut
  // the moment the epoch moves.
  const wasOpen = container.querySelector("details")?.open ?? false;
  container.textContent = "";
  const d = document.createElement("details");
  d.className = "tableview";
  d.open = wasOpen;
  const s = document.createElement("summary");
  s.textContent = summary;
  const scroll = document.createElement("div");
  scroll.className = "scroll";
  const t = document.createElement("table");
  t.className = "data";
  const thead = document.createElement("thead");
  const htr = document.createElement("tr");
  for (const c of columns) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = c;
    htr.append(th);
  }
  thead.append(htr);
  const tbody = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    r.cells.forEach((cell, i) => {
      const td = document.createElement("td");
      if (i === 0 && r.layer != null) {
        const sw = document.createElement("span");
        sw.className = `swatch ${swatchFor(r.layer)}`;
        td.append(sw);
      }
      td.append(document.createTextNode(cell));
      tr.append(td);
    });
    tbody.append(tr);
  }
  t.append(thead, tbody);
  scroll.append(t);
  d.append(s, scroll);
  container.append(d);
  return d;
}

/**
 * Which direct labels to show when several would land on top of each other.
 *
 * Deliberately *drops* the crowded ones rather than nudging them apart: a label
 * pushed off its mark stops pointing at anything and reads as noise. The legend
 * and the table keep every series identifiable, so losing a label costs nothing.
 * `ys` are the anchor positions; returns a Set of indices worth drawing.
 */
export function selectiveLabels(ys, minGap = 13) {
  const order = ys.map((y, i) => ({ y, i })).filter((d) => Number.isFinite(d.y)).sort((a, b) => a.y - b.y);
  const keep = new Set();
  let last = -Infinity;
  for (const { y, i } of order) {
    if (y - last < minGap) continue;
    keep.add(i);
    last = y;
  }
  return keep;
}

/** Re-render on container resize, so text keeps its true size at every width. */
export function responsive(container, render) {
  let last = -1;
  const run = () => {
    const w = Math.max(280, Math.floor(container.getBoundingClientRect().width));
    if (Math.abs(w - last) < 8) return;
    last = w;
    render(w);
  };
  run();
  const ro = new ResizeObserver(run);
  ro.observe(container);
  return ro;
}

/** Mark a card as reloading without a skeleton or a layout jump. */
export function setLoading(el, on) {
  el.classList.toggle("loading", !!on);
}
