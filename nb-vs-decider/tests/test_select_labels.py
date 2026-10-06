import pandas as pd
import pytest

from scripts import select_labels as sl


def test_highest_dev_accuracy_wins():
    assert sl.select_version({"v1": 0.6643, "v2": 0.7013, "v3": 0.7078}) == "v3"
    assert sl.select_version({"v1": 0.70, "v2": 0.68, "v3": 0.69}) == "v1"


def test_tie_to_four_decimals_goes_to_the_earlier_version():
    assert sl.select_version({"v1": 0.70001, "v2": 0.70004, "v3": 0.6}) == "v1"
    assert sl.select_version({"v1": 0.5, "v2": 0.7, "v3": 0.70002}) == "v2"


def test_no_scores_raises():
    with pytest.raises(ValueError):
        sl.select_version({})


def test_dev_scores_uses_bs1_dev_runs_only_and_requires_all_versions():
    runs = pd.DataFrame(
        {
            "run_id": ["dev_v1_bs1", "dev_v1_bs1_pass2", "dev_v2_bs1", "dev_v3_bs1", "test_v3_bs1"],
            "accuracy": [0.66, 0.99, 0.70, 0.71, 0.72],
        }
    )
    assert sl.dev_scores(runs) == {"v1": 0.66, "v2": 0.70, "v3": 0.71}
    with pytest.raises(RuntimeError):
        sl.dev_scores(runs[runs["run_id"] != "dev_v2_bs1"])


def test_decider_test_runs_follow_selection_and_best_batch_size(tmp_path, monkeypatch):
    import json

    from scripts import common

    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
    (tmp_path / "selected_labels.json").write_text(json.dumps({"selected": "v2"}))
    (tmp_path / "decider_best_bs.json").write_text(json.dumps({"best_batch_size": 4}))
    assert common.decider_test_runs() == {1: "test_v2_bs1", 4: "test_v2_bs4"}
    (tmp_path / "decider_best_bs.json").write_text(json.dumps({"best_batch_size": 1}))
    assert common.decider_test_runs() == {1: "test_v2_bs1"}
