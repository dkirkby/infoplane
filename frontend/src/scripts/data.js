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

  // Figure 4: the gradient statistics, one entry per weight matrix. FOUR
  // curves travel per matrix, because two independent choices are offered.
  //
  // Which gradients are compared: `mean`/`std` reduce over the epoch's
  // MINIBATCHES, which is the quantity the paper defines; `stepMean`/`stepStd`
  // reduce over the epoch's optimiser STEPS, the whole training set's gradient
  // re-evaluated at each weight state, which carries no minibatch noise at all
  // and is the one that reproduces the published figure. `stepStd` has no value
  // at the init checkpoint -- one weight state, no spread.
  //
  // Whether they are divided by ‖W‖: the `*Norm` forms are, which is what the
  // caption describes. Phase 1 does that division per seed, before the median
  // over seeds, so they are not reconstructable here by dividing one aggregate
  // by another; older exports that predate those keys fall back to exactly
  // that, which is close but not identical.
  const weightLayers = raw.weight_layers ?? [];
  const grads = {};
  for (const name of weightLayers) {
    const g = raw.gradients?.[name];
    if (!g) continue;
    const mean = Float32Array.from(g.mean, num);
    const std = Float32Array.from(g.std, num);
    const wnorm = Float32Array.from(g.weight_norm, num);
    const over = (a) => a.map((v, i) => v / wnorm[i]);
    const stepMean = g.step_mean ? Float32Array.from(g.step_mean, num) : null;
    const stepStd = g.step_std ? Float32Array.from(g.step_std, num) : null;
    grads[name] = {
      mean,
      std,
      meanNorm: g.mean_norm ? Float32Array.from(g.mean_norm, num) : over(mean),
      stdNorm: g.std_norm ? Float32Array.from(g.std_norm, num) : over(std),
      stepMean,
      stepStd,
      stepMeanNorm: g.step_mean_norm ? Float32Array.from(g.step_mean_norm, num)
        : stepMean && over(stepMean),
      stepStdNorm: g.step_std_norm ? Float32Array.from(g.step_std_norm, num)
        : stepStd && over(stepStd),
      weightNorm: wnorm,
    };
  }

  return {
    meta: raw.metadata,
    weightLayers,
    grads,
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

/** The architecture manifest (Figure 5): one entry per depth, 1..6 hidden layers. */
export async function loadArchIndex(url = "data/arch/index.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()).architectures;
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
