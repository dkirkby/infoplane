"""Command line entry point: ``infoplane run`` / ``infoplane sweep`` / ``infoplane verify``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from .datasets import DEFAULT_VAR_U_PATH, build_dataset, fetch_var_u
from .export import SIZE_WARN_BYTES, build_payload, write_arrow, write_json
from .model import PAPER_HIDDEN_WIDTHS
from .runner import run_ensemble
from .train import TrainConfig, dpi_stats


# The data processing inequality holds exactly for the true quantities, but the
# plug-in binned estimator can invert it by a hairline: binning layer i+1 is not a
# coarsening of binning layer i, so where a layer saturates and its bin grid merges
# patterns that the next layer's affine map pulls back apart, the deeper layer's
# entropy can come out marginally higher.
#
# The *rate* is the discriminator, not the worst single magnitude, which is noisy.
# Measured over 50 seeds x 5 layers x 257 checkpoints on the authors' architecture:
# 0/51400 on the full dataset and at every --train-size, under BOTH input encodings.
# The runs these thresholds must reject are the ones made with the paper's literal
# "7 hidden layers" reading, whose extra wide and 2-unit layers saturate: 4/77100
# (+0.068) on the full dataset but 43/77100 (+0.412 bits) at --train-size 500, and
# 126/77100 (+0.381) with binary inputs. So the check still catches a measurement
# that has become badly conditioned -- which is what it is for.
DPI_TOL_BITS = 0.1
DPI_MAX_VIOLATION_FRAC = 1e-4


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", default="var_u",
                   choices=("var_u", "quadric", "matfile", "mnist"),
                   help="var_u (default) = the authors' own var_u.mat, fetched on demand; "
                        "quadric = a synthetic deterministic rule that is NOT the paper's "
                        "problem, for offline use and contrast")
    g.add_argument("--n-bits", type=int, default=12, help="toy input width (default: 12)")
    g.add_argument("--input-encoding", default="binary", choices=("binary", "spin"),
                   help="binary (default) = the authors' raw 0/1 inputs; spin = inputs "
                        "centred to -+1, which trains slightly better and compresses "
                        "further but is not what the paper ran. Never compare runs made "
                        "with different encodings")
    g.add_argument("--geometry", default="circle", choices=("circle", "sphere"),
                   help="latent dimensionality of the quadric rule (default: circle = 2D)")
    g.add_argument("--data-path", default=None, help="var_u.mat, or the MNIST root")
    g.add_argument("--n-samples", type=int, default=4096, help="mnist subsample size")

    g = p.add_argument_group("model / training")
    g.add_argument("--widths", default=",".join(map(str, PAPER_HIDDEN_WIDTHS)),
                   help="comma-separated hidden widths (default: the paper's 10,7,5,4,3)")
    g.add_argument("--epochs", type=int, default=9000)
    g.add_argument("--log-points", type=int, default=256, help="log-spaced checkpoints")
    g.add_argument("--no-init-checkpoint", action="store_true",
                   help="omit epoch 0 (the network at initialisation)")
    g.add_argument("--batch-size", type=int, default=512, help="0 means full batch")
    g.add_argument("--optimizer", default="adam", choices=("sgd", "adam"))
    g.add_argument("--lr", type=float, default=4e-4)
    g.add_argument("--momentum", type=float, default=0.0)
    g.add_argument("--train-frac", type=float, default=0.8)
    g.add_argument("--device", default="cpu",
                   help="cpu (default; fastest for these tiny nets), mps or cuda")

    g = p.add_argument_group("information estimate")
    g.add_argument("--bins", type=int, default=30, help="uniform bins per unit over [-1,1]")
    g.add_argument("--mi-on", default="all", choices=("all", "train", "test"))

    g = p.add_argument_group("ensemble / output")
    g.add_argument("--num-seeds", type=int, default=50)
    g.add_argument("--seed0", type=int, default=0, help="first seed; seeds are seed0..seed0+n-1")
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--round", type=int, default=5, dest="round_to",
                   help="decimals kept in the export (MI is in bits)")
    g.add_argument("--format", default="json", choices=("json", "arrow"))
    g.add_argument("--quiet", action="store_true")


def _config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        n_bits=args.n_bits,
        geometry=args.geometry,
        encoding=args.input_encoding,
        data_path=args.data_path,
        widths=tuple(int(w) for w in args.widths.split(",") if w.strip()),
        epochs=args.epochs,
        log_points=args.log_points,
        include_init=not args.no_init_checkpoint,
        bins=args.bins,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        train_frac=args.train_frac,
        mi_on=args.mi_on,
        device=args.device,
    )


def _dataset(args: argparse.Namespace):
    return build_dataset(
        args.dataset,
        n_bits=args.n_bits,
        geometry=args.geometry,
        path=args.data_path,
        n_samples=args.n_samples,
        encoding=args.input_encoding,
    )


def _write(payload, path: Path, fmt: str, quiet: bool) -> None:
    size = write_arrow(payload, path) if fmt == "arrow" else write_json(payload, path)
    if not quiet:
        d = payload["diagnostics"]
        print(f"  wrote {path}  ({size / 1e6:.1f} MB)")
        print(f"  final accuracy  train {d['final_train_acc_mean']:.4f}  test {d['final_test_acc_mean']:.4f}")
        print(f"  max DPI violation {d['max_dpi_violation_bits']:+.2e} bits in "
              f"{d['dpi_violations']}/{d['dpi_measurements']} measurements "
              f"({d['dpi_violations'] / max(d['dpi_measurements'], 1):.4%}; "
              f"a few hairline inversions are an estimator artifact, a large "
              f"fraction is a bug)")
    if fmt == "json" and size > SIZE_WARN_BYTES:
        print(f"  WARNING: {size / 1e6:.0f} MB exceeds the {SIZE_WARN_BYTES / 1e6:.0f} MB budget in "
              "PLAN.md; re-run with --format arrow or fewer --num-seeds", file=sys.stderr)


def _run_one(dataset, cfg: TrainConfig, args: argparse.Namespace, out: Path) -> dict:
    seeds = list(range(args.seed0, args.seed0 + args.num_seeds))
    results = run_ensemble(dataset, cfg, seeds, num_workers=args.num_workers, progress=not args.quiet)
    payload = build_payload(dataset, cfg, results, round_to=args.round_to)
    _write(payload, out, args.format, args.quiet)
    return payload


def cmd_run(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    cfg = _config(args)
    if not args.quiet:
        print(f"{dataset.name}: N={dataset.n_samples} d={dataset.n_features} "
              f"classes={dataset.n_classes} | hidden {cfg.widths} | "
              f"{cfg.epochs} epochs x {args.num_seeds} seeds on {args.num_workers} workers")
    _run_one(dataset, cfg, args, Path(args.out))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Figure 5: one ensemble per training-set size, written as separate files.

    Kept as separate files with an identical schema rather than a new axis inside
    one document, so the frontend can lazy-load only the sizes being compared.
    """
    dataset = _dataset(args)
    base = _config(args)
    sizes = [int(s) for s in args.train_sizes.split(",")]
    outdir = Path(args.outdir)
    index = []
    for n in sizes:
        cfg = replace(base, train_size=n)
        suffix = "arrow" if args.format == "arrow" else "json"
        out = outdir / f"trainsize_{n:05d}.{suffix}"
        if not args.quiet:
            print(f"train size {n}/{dataset.n_samples}:")
        payload = _run_one(dataset, cfg, args, out)
        index.append({
            "train_size": n,
            "file": out.name,
            "final_train_acc": payload["diagnostics"]["final_train_acc_mean"],
            "final_test_acc": payload["diagnostics"]["final_test_acc_mean"],
        })
    (outdir / "index.json").write_text(json.dumps({"sizes": index}, indent=2))
    if not args.quiet:
        print(f"  wrote {outdir / 'index.json'}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download the authors' var_u.mat so --dataset var_u can be used."""
    path = fetch_var_u(args.data_path or DEFAULT_VAR_U_PATH, force=args.force)
    print(f"  {path} ready (SHA-256 verified)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Fast end-to-end check of the paper's stated invariants on a short run."""
    from .mi import entropy, mutual_information

    dataset = _dataset(args)
    cfg = replace(_config(args), epochs=args.epochs, log_points=args.log_points)
    x_sym = np.unique(dataset.X, axis=0, return_inverse=True)[1].reshape(-1)
    h_x, h_y = entropy(x_sym), entropy(dataset.Y)
    results = run_ensemble(dataset, cfg, list(range(args.num_seeds)),
                           num_workers=args.num_workers, progress=not args.quiet)
    ixt = np.stack([r.ixt for r in results])
    ity = np.stack([r.ity for r in results])
    stats = [dpi_stats(r.ixt) for r in results]
    worst = max(s[0] for s in stats)
    n_viol, n_total = sum(s[1] for s in stats), sum(s[2] for s in stats)
    checks = [
        (f"H(X) = {h_x:.4f} bits (expect {np.log2(dataset.n_samples):.0f} for enumerated inputs)",
         abs(h_x - np.log2(dataset.n_samples)) < 1e-9),
        (f"H(Y) = {h_y:.4f} bits (expect ~1 for a near-balanced binary rule)",
         abs(h_y - 1.0) < 0.02),
        (f"the rule is deterministic, so I(X;Y) = H(Y) = {h_y:.4f}",
         abs(mutual_information(x_sym, dataset.Y) - h_y) < 1e-9),
        (f"I(X;T) <= H(X): max {ixt.max():.4f}", ixt.max() <= h_x + 1e-9),
        (f"I(T;Y) <= H(Y): max {ity.max():.4f}", ity.max() <= h_y + 1e-9),
        (f"data processing inequality: max violation {worst:+.2e} bits "
         f"(tolerance {DPI_TOL_BITS}), {n_viol}/{n_total} measurements affected "
         f"(tolerance {DPI_MAX_VIOLATION_FRAC:.3%})",
         worst <= DPI_TOL_BITS and n_viol <= DPI_MAX_VIOLATION_FRAC * max(n_total, 1)),
        (f"train loss converges: mean final {np.mean([r.train_loss[-1] for r in results]):.4f}",
         np.mean([r.train_loss[-1] for r in results]) < 0.1),
        (f"train accuracy: mean final {np.mean([r.train_acc[-1] for r in results]):.4f}",
         np.mean([r.train_acc[-1] for r in results]) > 0.95),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")
        ok &= bool(passed)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infoplane", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="train an ensemble and export its trajectories")
    _add_common(p)
    p.add_argument("--out", default="data/infoplane.json")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="Figure 5: repeat the ensemble over training-set sizes")
    _add_common(p)
    p.add_argument("--train-sizes", default="50,100,500,4000")
    p.add_argument("--outdir", default="data/trainsize")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("fetch", help="download the authors' var_u.mat (1 KB, checksummed)")
    p.add_argument("--data-path", default=None, help=f"destination (default: {DEFAULT_VAR_U_PATH})")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="short run checking the information-theoretic invariants")
    _add_common(p)
    # 2000 epochs, not 500: at the authors' learning rate the network is only
    # ~0.95 train accuracy by epoch 500, so a shorter run would make the
    # convergence checks below trip on a healthy model.
    p.set_defaults(func=cmd_verify, epochs=2000, log_points=40, num_seeds=2, num_workers=2)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
