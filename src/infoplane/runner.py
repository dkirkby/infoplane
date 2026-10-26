"""Multi-seed execution of the pipeline."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterable, Sequence

import torch
from tqdm import tqdm

from .datasets import Dataset
from .train import SeedResult, TrainConfig, run_seed


def _init_worker() -> None:
    """One BLAS/torch thread per worker.

    The networks here are tiny, so intra-op parallelism buys nothing and workers
    would otherwise fight over the same cores -- the ensemble is embarrassingly
    parallel across seeds, and that is where all the speedup is.
    """
    torch.set_num_threads(1)


def _one(args: tuple[Dataset, TrainConfig, int]) -> SeedResult:
    dataset, cfg, seed = args
    return run_seed(dataset, cfg, seed)


def run_ensemble(
    dataset: Dataset,
    cfg: TrainConfig,
    seeds: Sequence[int],
    num_workers: int = 1,
    progress: bool = True,
) -> list[SeedResult]:
    """Train every seed and return results ordered by ``seeds``.

    ``num_workers <= 1`` runs in-process, which keeps tracebacks readable and is
    what the tests use.
    """
    if num_workers <= 1:
        _init_worker()
        it: Iterable[SeedResult] = (run_seed(dataset, cfg, s) for s in seeds)
        return list(tqdm(it, total=len(seeds), desc="seeds", disable=not progress))

    results: dict[int, SeedResult] = {}
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_one, (dataset, cfg, s)): s for s in seeds}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="seeds", disable=not progress):
            results[futures[fut]] = fut.result()
    return [results[s] for s in seeds]
