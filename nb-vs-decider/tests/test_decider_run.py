import numpy as np
import pandas as pd
import pytest

from scripts import decider_run as dr
from scripts.decider_sweep import choose_best


def test_test_split_refused_without_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(SystemExit):
        dr.check_split_allowed("test")
    dr.check_split_allowed("dev")
    monkeypatch.setenv("ALLOW_TEST", "1")
    dr.check_split_allowed("test")


def test_stratified_subset_is_balanced_seeded_and_ordered():
    df = pd.DataFrame({"item_id": range(70), "text": "x", "label": [i % 7 for i in range(70)]})
    a = dr.stratified_subset(df, 21, seed=1)
    assert a["label"].value_counts().to_dict() == dict.fromkeys(range(7), 3)
    assert a["item_id"].is_monotonic_increasing
    assert a.equals(dr.stratified_subset(df, 21, seed=1))
    assert not a.equals(dr.stratified_subset(df, 21, seed=2))
    assert len(dr.stratified_subset(df, 23, seed=1)) == 23


def test_stratified_subset_too_large_raises():
    df = pd.DataFrame({"item_id": range(7), "text": "x", "label": range(7)})
    with pytest.raises(ValueError):
        dr.stratified_subset(df, 14)


def test_latency_stats_and_batch_layout():
    res = dr.InferResult(np.zeros((5, 7), "f4"), np.zeros((5, 7), "f4"), [0.4, 0.2, 0.1], [2, 2, 1])
    assert res.batch_index.tolist() == [0, 0, 1, 1, 2]
    assert res.position_in_batch.tolist() == [0, 1, 0, 1, 0]
    s = dr.latency_stats(res)
    assert s["batch_p50_ms"] == pytest.approx(200.0)
    assert s["item_p50_ms"] == pytest.approx(100.0)


def test_v1_labels_cover_all_classes():
    cfg = dr.load_labels("v1")
    assert [cfg["labels"][i]["name"] for i in range(7)][0] == "Computer Science"


def test_choose_best_prefers_smaller_within_3_percent():
    assert choose_best({1: 10, 8: 40, 16: 50, 32: 49, 64: 50.5}) == 16
    assert choose_best({1: 10, 8: 40, 16: 41}) == 8
    assert choose_best({1: 10, 2: 20}) == 2
