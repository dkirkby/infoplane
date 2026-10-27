"""Serialisation of an ensemble to the JSON contract consumed by the frontend.

The schema is the interface between Phase 1 and Phase 2 (``PLAN.md``), so prefer
additive changes: every ``fig*.html`` page reads these keys.

One deliberate deviation from the schema sketched in ``PLAN.md``: ``epochs`` holds
the ~256 *checkpoint* epochs, not all 9000.  The sketch listed every epoch, but the
trajectories only have one entry per checkpoint, so a 9000-long axis could not be
zipped against them.  ``metadata.epochs`` keeps the total training length.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .datasets import Dataset
from .mi import entropy
from .train import SeedResult, TrainConfig, dpi_stats

SIZE_WARN_BYTES = 50 * 1024 * 1024  # PLAN.md's JSON budget


def _r(a: np.ndarray, nd: int) -> list[float]:
    """Round to ``nd`` decimals and drop NaNs to ``None`` so the JSON stays valid."""
    out = np.round(np.asarray(a, dtype=float), nd)
    return [None if not np.isfinite(v) else float(v) for v in out]


def build_payload(
    dataset: Dataset,
    cfg: TrainConfig,
    results: list[SeedResult],
    round_to: int = 5,
) -> dict[str, Any]:
    """Assemble the export document from a finished ensemble."""
    if not results:
        raise ValueError("no results to export")
    layers = results[0].layer_names
    epochs = results[0].epochs
    for r in results:
        if r.layer_names != layers or not np.array_equal(r.epochs, epochs):
            raise ValueError("results disagree on layers or checkpoint epochs")
        if r.weight_names != results[0].weight_names:
            raise ValueError("results disagree on weight matrices")

    ixt = np.stack([r.ixt for r in results])  # (seeds, layers, checkpoints)
    ity = np.stack([r.ity for r in results])

    runs = [
        {
            "seed": r.seed,
            "n_train": r.n_train,
            "seconds": round(r.seconds, 2),
            "trajectories": {
                name: {"IXT": _r(r.ixt[li], round_to), "ITY": _r(r.ity[li], round_to)}
                for li, name in enumerate(layers)
            },
            "curves": {
                "train_loss": _r(r.train_loss, 6),
                "test_loss": _r(r.test_loss, 6),
                "train_acc": _r(r.train_acc, 5),
                "test_acc": _r(r.test_acc, 5),
            },
        }
        for r in results
    ]

    aggregated = {
        name: {
            "IXT_mean": _r(ixt[:, li].mean(0), round_to),
            "IXT_std": _r(ixt[:, li].std(0), round_to),
            "ITY_mean": _r(ity[:, li].mean(0), round_to),
            "ITY_std": _r(ity[:, li].std(0), round_to),
        }
        for li, name in enumerate(layers)
    }

    curves = {
        key: {
            "mean": _r(np.nanmean(np.stack([getattr(r, key) for r in results]), axis=0), 6),
            "std": _r(np.nanstd(np.stack([getattr(r, key) for r in results]), axis=0), 6),
        }
        for key in ("train_loss", "test_loss", "train_acc", "test_acc")
    }

    # Figure 4: the gradient statistics, aggregated over seeds only. Unlike the
    # trajectories these are not kept per seed -- the figure is one curve per
    # weight matrix, and 50 x 6 x n_checkpoints of them would roughly double the
    # file for something nothing plots.
    gradients = {}
    if results[0].grad_mean is not None:
        wnames = results[0].weight_names
        g_mean = np.stack([r.grad_mean for r in results])      # (seeds, layers, ck)
        g_std = np.stack([r.grad_std for r in results])
        s_mean = np.stack([r.step_mean for r in results])
        s_std = np.stack([r.step_std for r in results])
        w_norm = np.stack([r.weight_norm for r in results])
        # The initialisation checkpoint has no gradients, so its column is all-NaN
        # and nanmean warns about an empty slice. NaN is the right answer there.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # The MEDIAN over seeds, not the mean, is what the figure plots.
            # These norms are log-distributed across initialisations -- at epoch 0
            # the spread of a 50-seed ensemble covers more than a decade, because
            # the backpropagated signal is a product of five random matrices -- so
            # an arithmetic mean sits near the largest few seeds and is not a
            # typical network. The paper's own figure is a single run. The mean is
            # kept alongside for anyone who wants it.
            # Figure 4's caption normalises each curve by that layer's weight
            # norm, so both forms are exported.  The division happens HERE, per
            # seed, and not on the aggregate in the frontend: ||g|| and ||W||
            # are correlated across seeds, so a ratio of medians is not the
            # median of the ratio.  Which one the figure should plot is a
            # separate question -- see the note in `fig4.js`.
            n_mean = g_mean / w_norm
            n_std = g_std / w_norm
            # The second reduction (`train.run_seed`): the same two numbers over
            # the whole training set's gradient at each of the epoch's weight
            # states, which is the published figure's dotted curve where the
            # minibatch spread is not.  Both travel so the page can switch.
            sn_mean = s_mean / w_norm
            sn_std = s_std / w_norm
            gradients = {
                name: {
                    "mean": _r(np.nanmedian(g_mean[:, wi], axis=0), 8),
                    "mean_amean": _r(np.nanmean(g_mean[:, wi], axis=0), 8),
                    "mean_sd": _r(np.nanstd(g_mean[:, wi], axis=0), 8),
                    "std": _r(np.nanmedian(g_std[:, wi], axis=0), 8),
                    "std_amean": _r(np.nanmean(g_std[:, wi], axis=0), 8),
                    "std_sd": _r(np.nanstd(g_std[:, wi], axis=0), 8),
                    "mean_norm": _r(np.nanmedian(n_mean[:, wi], axis=0), 8),
                    "std_norm": _r(np.nanmedian(n_std[:, wi], axis=0), 8),
                    # 12 decimals, not 8: the step spread runs down to 1e-7 and
                    # below once divided by ||W||, where rounding to 8 would
                    # leave one significant figure and visibly quantise a log
                    # axis -- or round a positive value to zero and drop the
                    # point from the plot entirely.
                    "step_mean": _r(np.nanmedian(s_mean[:, wi], axis=0), 12),
                    "step_std": _r(np.nanmedian(s_std[:, wi], axis=0), 12),
                    "step_mean_norm": _r(np.nanmedian(sn_mean[:, wi], axis=0), 12),
                    "step_std_norm": _r(np.nanmedian(sn_std[:, wi], axis=0), 12),
                    "weight_norm": _r(np.nanmedian(w_norm[:, wi], axis=0), 6),
                }
                for wi, name in enumerate(wnames)
            }

    worst, n_viol, n_total = zip(*(dpi_stats(r.ixt) for r in results))

    h_x = entropy(np.unique(dataset.X, axis=0, return_inverse=True)[1].reshape(-1))
    h_y = entropy(dataset.Y)

    metadata = cfg.as_metadata() | {
        "infoplane_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_name": dataset.name,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_classes": dataset.n_classes,
        "seeds": len(results),
        "checkpoints": int(len(epochs)),
        "log_points_requested": cfg.log_points,
    }

    return {
        "metadata": metadata,
        # Information-theoretic ceilings, so the frontend can draw axis bounds
        # without recomputing anything: I(X;T) <= H(X) and I(T;Y) <= H(Y).
        "bounds": {"H_X": round(h_x, round_to), "H_Y": round(h_y, round_to)},
        "epochs": [int(e) for e in epochs],
        "layers": layers,
        "weight_layers": results[0].weight_names,
        "runs": runs,
        "aggregated": aggregated,
        "gradients": gradients,
        "curves": curves,
        "diagnostics": {
            "max_dpi_violation_bits": round(max(worst), 6),
            "dpi_violations": int(sum(n_viol)),
            "dpi_measurements": int(sum(n_total)),
            "final_train_acc_mean": round(float(np.mean([r.train_acc[-1] for r in results])), 5),
            "final_test_acc_mean": round(float(np.mean([r.test_acc[-1] for r in results])), 5),
            "seconds_total": round(sum(r.seconds for r in results), 1),
        },
    }


def write_json(payload: dict[str, Any], path: str | Path) -> int:
    """Write the payload compactly; returns the byte size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    return len(text.encode())


def write_arrow(payload: dict[str, Any], path: str | Path) -> int:
    """Long-format Feather fallback for when the JSON exceeds the size budget.

    One row per (seed, layer, checkpoint); ``metadata`` travels in the schema's
    key-value metadata so the file stays self-describing.
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    epochs = payload["epochs"]
    layers = payload["layers"]
    seeds, layer_col, epoch_col, ixt, ity = [], [], [], [], []
    for run in payload["runs"]:
        for name in layers:
            traj = run["trajectories"][name]
            seeds.extend([run["seed"]] * len(epochs))
            layer_col.extend([name] * len(epochs))
            epoch_col.extend(epochs)
            ixt.extend(traj["IXT"])
            ity.extend(traj["ITY"])
    table = pa.table(
        {
            "seed": pa.array(seeds, pa.int32()),
            "layer": pa.array(layer_col, pa.string()),
            "epoch": pa.array(epoch_col, pa.int32()),
            "IXT": pa.array(ixt, pa.float32()),
            "ITY": pa.array(ity, pa.float32()),
        }
    )
    meta = {k: json.dumps(v) for k, v in payload.items() if k in ("metadata", "bounds", "diagnostics")}
    table = table.replace_schema_metadata(meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(table, path, compression="zstd")
    return path.stat().st_size
