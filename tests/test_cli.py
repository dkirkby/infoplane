import json

import pytest

from infoplane.cli import main

# The CLI is exercised at a toy scale; the point is that the argparse surface, the
# in-process runner and both writers agree with each other, not that anything learns.
TINY = ["--num-seeds", "2", "--num-workers", "1", "--epochs", "3", "--log-points", "4", "--quiet"]


def test_run_writes_the_contract(tmp_path):
    out = tmp_path / "e.json"
    assert main(["run", *TINY, "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["metadata"]["seeds"] == 2
    assert payload["metadata"]["epochs"] == 3  # total training length, not checkpoints
    assert len(payload["epochs"]) == payload["metadata"]["checkpoints"]
    assert payload["metadata"]["widths"] == [12, 10, 7, 5, 4, 3, 2]
    assert payload["diagnostics"]["max_dpi_violation_bits"] <= 1e-9


def test_cli_overrides_reach_the_config(tmp_path):
    out = tmp_path / "e.json"
    assert main(["run", *TINY, "--out", str(out), "--widths", "6,4", "--bins", "10",
                 "--lr", "0.05", "--optimizer", "adam", "--n-bits", "8",
                 "--no-init-checkpoint", "--mi-on", "train"]) == 0
    payload = json.loads(out.read_text())
    assert payload["metadata"]["widths"] == [6, 4]
    assert payload["metadata"]["bins"] == 10
    assert payload["metadata"]["optimizer"] == "adam"
    assert payload["layers"] == ["Layer 1", "Layer 2"]
    assert payload["bounds"]["H_X"] == pytest.approx(8.0)  # 2**8 enumerated patterns
    assert payload["epochs"][0] == 1  # no epoch-0 checkpoint


def test_sweep_writes_one_file_per_size_plus_an_index(tmp_path):
    assert main(["sweep", *TINY, "--train-sizes", "50,200", "--outdir", str(tmp_path)]) == 0
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["train_size"] for e in index["sizes"]] == [50, 200]
    for entry in index["sizes"]:
        payload = json.loads((tmp_path / entry["file"]).read_text())
        assert payload["metadata"]["train_size"] == entry["train_size"]
        assert all(run["n_train"] == entry["train_size"] for run in payload["runs"])
        # The information plane is still measured on all 4096 patterns, so the
        # trajectories stay comparable across sizes.
        assert payload["bounds"]["H_X"] == pytest.approx(12.0)


def test_verify_passes_on_its_own_defaults():
    assert main(["verify", "--quiet", "--num-seeds", "1", "--num-workers", "1"]) == 0


def test_arrow_export_round_trips(tmp_path):
    feather = pytest.importorskip("pyarrow.feather")
    out = tmp_path / "e.arrow"
    assert main(["run", *TINY, "--out", str(out), "--format", "arrow"]) == 0

    table = feather.read_table(out)
    assert table.column_names == ["seed", "layer", "epoch", "IXT", "ITY"]
    n_ck = len(set(table.column("epoch").to_pylist()))
    assert table.num_rows == 2 * 7 * n_ck  # seeds x layers x checkpoints
    meta = json.loads(table.schema.metadata[b"metadata"])
    assert meta["seeds"] == 2 and meta["checkpoints"] == n_ck
