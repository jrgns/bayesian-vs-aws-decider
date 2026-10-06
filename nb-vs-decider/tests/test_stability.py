import numpy as np
import pandas as pd

from scripts import stability_metrics as sm
from scripts import stability_run as sr
from scripts import stability_subset as ss

# ------------------------------------------------------------------ subset


def toy_main(n_per_class: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 7 * n_per_class
    return pd.DataFrame(
        {
            "item_id": [f"i{j:05d}" for j in range(n)],
            "label": np.tile(np.arange(7), n_per_class),
            "margin_top2": rng.random(n),
        }
    )


def small_subset(main: pd.DataFrame) -> pd.DataFrame:
    return ss.build_subset(main, seed=5, n_per_stratum=70, near_pool=140)


def test_subset_sizes_and_strata():
    main = toy_main()
    sub = small_subset(main)
    assert len(sub) == 140 and sub["item_id"].is_unique
    assert sub["stratum"].value_counts().to_dict() == {"near_boundary": 70, "rest": 70}
    ranked = main.sort_values(["margin_top2", "item_id"])
    low = set(ranked["item_id"].iloc[:140])
    near = sub[sub["stratum"] == "near_boundary"]
    assert set(near["item_id"]) <= low
    assert not set(sub[sub["stratum"] == "rest"]["item_id"]) & low
    # labels in the output are the true labels
    assert (
        sub.merge(main, on="item_id")["label_x"] == sub.merge(main, on="item_id")["label_y"]
    ).all()


def test_subset_deterministic_and_seeded():
    main = toy_main()
    a, b = small_subset(main), small_subset(main)
    pd.testing.assert_frame_equal(a, b)
    c = ss.build_subset(main, seed=6, n_per_stratum=70, near_pool=140)
    assert not a["item_id"].equals(c["item_id"])


def test_near_stratum_proportional_rest_balanced():
    main = toy_main()
    # make label 0 twice as common in the low-margin pool
    main.loc[main["label"] == 0, "margin_top2"] *= 0.5
    sub = small_subset(main)
    near = sub[sub["stratum"] == "near_boundary"]["label"].value_counts()
    assert near[0] > near[3]
    rest = sub[sub["stratum"] == "rest"]["label"].value_counts()
    assert rest.max() - rest.min() <= 1


def test_default_counts_are_72_then_71():
    assert ss.balanced_counts(list(range(7)), 500) == {
        0: 72,
        1: 72,
        2: 72,
        3: 71,
        4: 71,
        5: 71,
        6: 71,
    }
    counts = ss.proportional_counts({0: 300, 1: 100, 2: 600}, 500)
    assert sum(counts.values()) == 500 and counts == {0: 150, 1: 50, 2: 300}


def test_margin_ties_broken_by_item_id():
    main = toy_main(20)
    main["margin_top2"] = 0.5  # all tied: the near pool must be the lowest item_ids
    sub = ss.build_subset(main, seed=1, n_per_stratum=14, near_pool=28)
    low = set(main["item_id"].sort_values().iloc[:28])
    assert set(sub[sub["stratum"] == "near_boundary"]["item_id"]) <= low


# ------------------------------------------------------------------ run helpers


def test_unpermute_columns_roundtrip():
    canon = np.arange(21, dtype=np.float32).reshape(3, 7)
    perm = np.array([3, 0, 6, 1, 5, 2, 4])  # slot j shows canonical label perm[j]
    shown = canon[:, perm]
    assert np.array_equal(sr.unpermute_columns(shown, perm), canon)


def test_unpermute_columns_label_mapping():
    perm = sr.option_perm(2)
    inv = np.argsort(perm)
    for label in range(7):
        slot = inv[label]
        assert perm[slot] == label
    shown = np.zeros((1, 7), dtype=np.float32)
    shown[0, inv[4]] = 1.0  # the model scores the slot holding canonical label 4
    assert sr.unpermute_columns(shown, perm).argmax() == 4


def test_item_orders():
    n = 10
    assert sr.item_order("canonical", 0, n, 2).tolist() == list(range(n))
    swap = sr.item_order("swap_within_batch", 3, n, 2)
    assert sorted(swap.tolist()) == list(range(n))
    for b in range(5):
        assert set(swap[2 * b : 2 * b + 2]) == {2 * b, 2 * b + 1}
    assert not np.array_equal(sr.item_order("shuffle_all", 1, 100, 2), np.arange(100))
    assert np.array_equal(
        sr.item_order("shuffle_all", 1, 100, 2), sr.item_order("shuffle_all", 1, 100, 2)
    )


def test_seed_used():
    assert sr.seed_used(sr.CONDITIONS["D2"], 4) == -1
    assert sr.seed_used(sr.CONDITIONS["D3"], 4) == 0
    assert sr.seed_used(sr.CONDITIONS["D6b"], 4) == 4


# ------------------------------------------------------------------ metrics


def toy_runs(preds, logits_first=None, y=None):
    preds = np.asarray(preds)
    r, n = preds.shape
    scores = np.zeros((r, n, 7), dtype=np.float32)
    scores[:, :, 0] = (
        np.asarray(logits_first, dtype=np.float32) if logits_first is not None else 1.0
    )
    y = np.zeros(n, dtype=int) if y is None else y
    return sm.Runs(preds, scores, y)


def test_flip_and_pairwise():
    preds = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 5, 3]]
    assert sm.flipped(np.array(preds)).tolist() == [False, False, True, True]
    assert sm.flip_rate(np.array(preds)) == 0.5
    # pairs: (0,1): 1/4, (0,2): 1/4, (1,2): 2/4
    assert sm.pairwise_disagreement(np.array(preds)) == (0.25 + 0.25 + 0.5) / 3


def test_bitwise_and_max_diff():
    a = np.ones((3, 2, 7), dtype=np.float32)
    assert sm.bitwise_identical_share(a) == 1.0 and sm.max_abs_diff(a) == 0.0
    b = a.copy()
    b[1, 0, 3] = np.nextafter(np.float32(1.0), np.float32(2.0))  # 1 ulp
    assert sm.bitwise_identical_share(b) == 0.5
    assert sm.max_abs_diff(b) == float(b[1, 0, 3] - 1.0)
    z = a.copy()
    z[2, 1, 0] = -1.0
    z[0, 1, 0] = 1.0
    assert sm.bitwise_identical_share(z) == 0.5
    f64 = np.ones((2, 1, 7))
    f64[1, 0, 0] = 1 + 1e-15
    assert sm.bitwise_identical_share(f64) == 0.0


def test_reweighting():
    strata = np.array(["near_boundary"] * 2 + ["rest"] * 8)
    # near: both items flip; rest: 1 of 8 flips
    preds = np.zeros((2, 10), dtype=int)
    preds[1, :2] = 1
    preds[1, 5] = 1
    rows = {r["stratum"]: r for r in sm.condition_rows("X", toy_runs(preds), strata)}
    assert rows["near_boundary"]["flip_rate"] == 1.0
    assert rows["rest"]["flip_rate"] == 1 / 8
    assert rows["all"]["flip_rate"] == 0.3
    assert rows["all"]["n_items"] == 10 and rows["all"]["n_runs"] == 2
    assert rows["near_boundary"]["n_items"] == 2
    assert abs(rows["reweighted"]["flip_rate"] - (0.1 * 1.0 + 0.9 / 8)) < 1e-12
    assert abs(rows["reweighted"]["pairwise_disagreement"] - (0.1 * 1.0 + 0.9 / 8)) < 1e-12
    assert rows["reweighted"]["bitwise_identical"] == 0.1 * 1.0 + 0.9 * 1.0
    assert set(rows) == set(sm.STRATA)


def test_accuracy_min_max():
    strata = np.array(["near_boundary", "rest", "rest", "rest"])
    preds = np.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    row = {r["stratum"]: r for r in sm.condition_rows("X", toy_runs(preds), strata)}["all"]
    assert (row["acc_min"], row["acc_max"]) == (0.5, 1.0)


def test_paired_rows():
    strata = np.array(["near_boundary", "rest", "rest", "rest"])
    a = toy_runs([[0, 0, 0, 0], [0, 0, 0, 0]])
    b = toy_runs([[0, 0, 0, 0], [1, 0, 0, 0]], logits_first=[1, 1, 1, 2])
    b.scores[0, :, 0] = 1.0
    row = {r["stratum"]: r for r in sm.paired_rows("D4_paired", a, b, strata)}["all"]
    assert row["flip_rate"] == 0.25
    assert row["pairwise_disagreement"] == 1 / 8
    assert row["bitwise_identical"] == 0.75


def test_flipped_examples_prefer_different_conditions():
    ids = pd.DataFrame({"item_id": list("abcd"), "label": [0, 0, 1, 1], "stratum": ["rest"] * 4})
    margins = pd.Series([0.4, 0.1, 0.3, 0.2])
    flip = sm.Runs([[0, 0, 0, 0], [1, 1, 0, 0]], np.zeros((2, 4, 7), np.float32), ids["label"])
    flip2 = sm.Runs([[0, 0, 0, 0], [0, 1, 1, 0]], np.zeros((2, 4, 7), np.float32), ids["label"])
    ex = sm.flipped_examples({"A": flip, "B": flip2}, ids, margins, k=3)
    assert [e["condition"] for e in ex][:2] == ["A", "B"]
    assert len({e["item_id"] for e in ex}) == len(ex) == 3
    assert ex[0]["item_id"] == "b" and ex[0]["margin_top2"] == 0.1
