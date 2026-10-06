import pandas as pd

from scripts.dataset import clean_frame, item_id, make_splits
from scripts.ngrams import dedup_key


def frame(rows):
    return pd.DataFrame(rows, columns=["text", "label"])


def test_clean_dedups_after_normalisation():
    df = frame([("Hello  World", 1), ("hello world ", 1), ("other", 2)])
    out, counts = clean_frame(df)
    assert set(out["text"]) == {"Hello  World", "other"}
    assert counts["clean"] == 2
    assert counts["duplicate_copies_dropped"] == 1
    assert counts["conflict_rows_dropped"] == 0


def test_conflicting_labels_drop_every_copy():
    df = frame([("same text", 1), ("Same  text", 2), ("same text", 1), ("unique", 3)])
    out, counts = clean_frame(df)
    assert out["text"].tolist() == ["unique"]
    assert counts["conflict_groups"] == 1
    assert counts["conflict_rows_dropped"] == 3


def test_item_id_is_stable_sha1_prefix():
    assert item_id(dedup_key("A  b")) == item_id("a b")
    assert len(item_id("a b")) == 16


def toy(per_class=30, classes=3):
    rows = [(f"text {c} {i}", c) for c in range(classes) for i in range(per_class)]
    out, _ = clean_frame(frame(rows))
    return out


def test_splits_balanced_disjoint_and_deterministic():
    df = toy()
    a = make_splits(df, seed=7, test_per_class=5, dev_per_class=3)
    b = make_splits(df.sample(frac=1, random_state=1), seed=7, test_per_class=5, dev_per_class=3)
    for name, n in (("test", 5), ("dev", 3), ("pool", 22)):
        assert (a[name]["label"].value_counts() == n).all()
        assert a[name]["item_id"].tolist() == b[name]["item_id"].tolist()
    ids = [set(a[s]["item_id"]) for s in ("test", "dev", "pool")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == len(df)


def test_different_seed_changes_split():
    df = toy()
    a = make_splits(df, 1, 5, 3)["test"]["item_id"].tolist()
    assert a != make_splits(df, 2, 5, 3)["test"]["item_id"].tolist()
