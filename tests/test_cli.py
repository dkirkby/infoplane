import json

import pytest

from infoplane.cli import ARCH_DEFAULT, main
from infoplane.model import PAPER_HIDDEN_WIDTHS

# The CLI is exercised at a toy scale; the point is that the argparse surface, the
# in-process runner and both writers agree with each other, not that anything learns.
# `quadric` keeps the CLI tests offline; `var_u` would need the authors' download.
TINY = ["--dataset", "quadric", "--num-seeds", "2", "--num-workers", "1",
        "--epochs", "3", "--log-points", "4", "--quiet"]


def test_run_writes_the_contract(tmp_path):
    out = tmp_path / "e.json"
    assert main(["run", *TINY, "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["metadata"]["seeds"] == 2
    assert payload["metadata"]["epochs"] == 3  # total training length, not checkpoints
    assert len(payload["epochs"]) == payload["metadata"]["checkpoints"]
    assert payload["metadata"]["widths"] == [10, 7, 5, 4, 3]
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
    assert payload["layers"] == ["Layer 1", "Layer 2", "Output"]
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


def test_arch_writes_one_file_per_depth(tmp_path):
    assert main(["arch", *TINY, "--architectures", "10;10,7,5", "--outdir", str(tmp_path)]) == 0
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["n_hidden"] for e in index["architectures"]] == [1, 3]
    assert [e["widths"] for e in index["architectures"]] == [[10], [10, 7, 5]]
    for entry in index["architectures"]:
        payload = json.loads((tmp_path / entry["file"]).read_text())
        assert payload["metadata"]["widths"] == entry["widths"]
        assert payload["layers"][-1] == "Output"
        assert len(payload["layers"]) == entry["n_hidden"] + 1


def test_arch_defaults_to_the_authors_six_architectures():
    # Read off `plot_figures.py`, which feeds its "One hidden layer" ... "Six
    # hidden layers" grid `ff3_5` ... `ff3_1` -- the `net_type` values that
    # `select_network_arch` maps to these -- plus a 10,7,5,4,4,3 run. The
    # five-layer member being the main run's architecture is the check that this
    # is the right family: the caption's 12,10,8,6,4,2 is not it.
    widths = [[int(w) for w in a.split(",")] for a in ARCH_DEFAULT.split(";")]
    assert widths == [[10], [10, 7], [10, 7, 5], [10, 7, 5, 4], [10, 7, 5, 4, 3],
                      [10, 7, 5, 4, 4, 3]]
    assert tuple(widths[4]) == PAPER_HIDDEN_WIDTHS


def test_arch_rejects_two_architectures_of_the_same_depth(tmp_path):
    with pytest.raises(SystemExit):
        main(["arch", *TINY, "--architectures", "10,7;6,4", "--outdir", str(tmp_path)])


def test_verify_passes_on_its_own_defaults():
    assert main(["verify", "--dataset", "quadric", "--quiet",
                 "--num-seeds", "1", "--num-workers", "1"]) == 0


def test_arrow_export_round_trips(tmp_path):
    feather = pytest.importorskip("pyarrow.feather")
    out = tmp_path / "e.arrow"
    assert main(["run", *TINY, "--out", str(out), "--format", "arrow"]) == 0

    table = feather.read_table(out)
    assert table.column_names == ["seed", "layer", "epoch", "IXT", "ITY"]
    n_ck = len(set(table.column("epoch").to_pylist()))
    n_series = len(PAPER_HIDDEN_WIDTHS) + 1  # hidden layers + the measured readout
    assert table.num_rows == 2 * n_series * n_ck  # seeds x layers x checkpoints
    meta = json.loads(table.schema.metadata[b"metadata"])
    assert meta["seeds"] == 2 and meta["checkpoints"] == n_ck


def test_dpi_thresholds_discriminate_conditioning():
    """Pin the tolerances against the measured runs that motivated them.

    These are not arbitrary: every run on the authors' five-layer architecture must
    pass, and the saturated runs from the discarded seven-layer reading must fail,
    otherwise the check stops catching the regression it exists for.
    """
    from infoplane.cli import DPI_MAX_VIOLATION_FRAC, DPI_TOL_BITS

    def passes(violations, worst, total):
        return worst <= DPI_TOL_BITS and violations <= DPI_MAX_VIOLATION_FRAC * total

    five = 64250  # 50 seeds x 5 series pairs x 257 checkpoints (5 hidden + the readout)
    seven = 77100  # the discarded reading: 50 seeds x 6 layer pairs x 257

    # The authors' architecture, as regenerated: clean everywhere, under BOTH input
    # encodings and at every training-set size.
    assert passes(0, 0.0, five)
    # The literal "7 hidden layers" reading saturates its extra wide and 2-unit
    # layers. Its full-dataset run squeaks through on magnitude, so the *rate* is
    # what has to reject it; its small-train-size runs fail on both counts.
    assert not passes(43, 0.412, seven)
    assert not passes(126, 0.3805, seven)
    assert not passes(176, 1.44, seven)
