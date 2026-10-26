/** Loading and reshaping the Phase 1 export.
 *
 * The export schema is documented in the repo README ("Export schema"). Nothing
 * here recomputes information quantities - Phase 1 owns the estimator; the
 * frontend only reads what it produced, and reads `metadata.bins` so a page can
 * say under what binning its bit values hold.
 */

const cache = new Map();

/** Flatten the per-seed trajectories into typed arrays, once, at load. */
function prepare(raw) {
  const layers = raw.layers;
  const epochs = raw.epochs;
  const nE = epochs.length;

  const seeds = raw.runs.map((r) => r.seed);
  // seedPts[seed][layer] = { ixt: Float32Array(nE), ity: Float32Array(nE) }
  const seedPts = raw.runs.map((run) =>
    layers.map((name) => {
      const t = run.trajectories[name];
      return { ixt: Float32Array.from(t.IXT, num), ity: Float32Array.from(t.ITY, num) };
    })
  );

  const agg = {};
  for (const name of layers) {
    const a = raw.aggregated[name];
    agg[name] = {
      ixt: Float32Array.from(a.IXT_mean, num),
      ixtSd: Float32Array.from(a.IXT_std, num),
      ity: Float32Array.from(a.ITY_mean, num),
      itySd: Float32Array.from(a.ITY_std, num),
    };
  }

  return {
    meta: raw.metadata,
    bounds: raw.bounds,
    diagnostics: raw.diagnostics,
    curves: raw.curves,
    epochs,
    nEpochs: nE,
    layers,
    seeds,
    nSeeds: seeds.length,
    seedPts,
    agg,
    /** Ensemble mean/sd for one layer index. */
    layerAgg: (li) => agg[layers[li]],
  };
}

// Phase 1 writes JSON nulls where a curve is undefined (e.g. test loss with an
// empty test split). NaN keeps those out of the scales instead of plotting zero.
const num = (v) => (v == null ? NaN : v);

/** Fetch + prepare, memoised per URL. */
export async function load(url) {
  if (!cache.has(url)) {
    cache.set(
      url,
      (async () => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
        return prepare(await res.json());
      })().catch((err) => {
        cache.delete(url); // don't memoise a failure
        throw err;
      })
    );
  }
  return cache.get(url);
}

/** The training-size manifest (Figure 3): one entry per training-set size. */
export async function loadSweepIndex(url = "data/trainsize/index.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()).sizes;
}

/**
 * Nearest mark to a pointer, by squared distance in *pixels* so the hit area is
 * isotropic on screen rather than in data units. Returns null past `maxDist`,
 * which keeps the tooltip from latching onto something across the plot.
 */
export function nearest(marks, px, py, maxDist = 28) {
  let best = null;
  let bestD = maxDist * maxDist;
  for (const m of marks) {
    const dx = m.px - px;
    const dy = m.py - py;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = m;
    }
  }
  return best;
}

/** Index of the epoch closest to `epoch` (epochs are log-spaced, so scan). */
export function epochIndex(epochs, epoch) {
  let bi = 0;
  let bd = Infinity;
  for (let i = 0; i < epochs.length; i++) {
    const d = Math.abs(epochs[i] - epoch);
    if (d < bd) {
      bd = d;
      bi = i;
    }
  }
  return bi;
}

/** Human-readable description of the run behind a dataset, for figure captions. */
export function provenance(d) {
  const m = d.meta;
  return `${m.dataset_name} · ${m.seeds} seeds · ${int(m.epochs)} epochs · ${m.bins} bins · ${m.encoding} inputs · lr ${m.lr}`;
}
const int = (n) => n.toLocaleString("en-US");
