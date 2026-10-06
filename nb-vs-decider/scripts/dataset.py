"""WOS-46985 (level-1 labels): load, clean, split, length check.

Stages (run in order, with scripts.contamination in between):
  python -m scripts.dataset prepare   # download + parse + clean + provisional split
  python -m scripts.contamination     # 13-gram / exact overlap against Decider's corpora
  python -m scripts.dataset split     # drop contaminated items, freeze splits, length check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from scripts.common import (
    DATA_DIR,
    RESULTS_DIR,
    download,
    load_settings,
    setup_logging,
    write_json,
)
from scripts.ngrams import dedup_key

log = logging.getLogger(__name__)

SEED = 20261005
TEST_PER_CLASS = 1428  # floor(10000 / 7)
DEV_PER_CLASS = 286  # 2,002 items
MIN_POOL_PER_CLASS = 1000
MAX_TEST_HIT_SHARE = 0.01

HF_ID = "HDLTex/web_of_science"
HF_REVISION = "26ae29c300edec4acafbda030653640241c9f9a8"
HF_CONFIG = "WOS46985"
# The URL the HF loading script at HF_REVISION downloads (Mendeley Data, WOS release v6).
ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
)
ARCHIVE_SHA256 = "b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a"

# Archive Domain string (stripped, lowercased) -> plain-English name.
PLAIN_NAMES = {
    "cs": "Computer Science",
    "ece": "Electrical Engineering",
    "psychology": "Psychology",
    "mae": "Mechanical Engineering",
    "civil": "Civil Engineering",
    "medical": "Medical Science",
    "biochemistry": "Biochemistry",
}

RAW_DIR = DATA_DIR / "raw" / "wos"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARCHIVE = RAW_DIR / "WOS.zip"
CLEAN_PARQUET = PROCESSED_DIR / "wos46985_l1_clean.parquet"
PROVISIONAL_CSV = PROCESSED_DIR / "provisional_splits.csv"
PREPARE_JSON = PROCESSED_DIR / "prepare_summary.json"
FINAL_PARQUET = PROCESSED_DIR / "wos46985_l1.parquet"
HITS_CSV = RESULTS_DIR / "contamination_hits.csv"
SPLIT_NAMES = ("test", "dev", "pool")

PLACEHOLDER_QUESTION = "Which field is this abstract from?"  # length counting only


# ---------------------------------------------------------------- download and parsing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_archive() -> str:
    """Fetch the Mendeley archive once; verify its sha256. Returns the sha256."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        log.info("downloading %s", ARCHIVE_URL)
        download(ARCHIVE_URL, ARCHIVE)
    sha = sha256_file(ARCHIVE)
    if sha != ARCHIVE_SHA256:
        raise RuntimeError(f"archive sha256 {sha} != expected {ARCHIVE_SHA256}")
    return sha


def try_hf_loader() -> str:
    """Try the preferred source; return 'ok' or the failure reason (recorded in the summary)."""
    try:
        from datasets import load_dataset

        load_dataset(HF_ID, HF_CONFIG, revision=HF_REVISION, trust_remote_code=True)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - any failure just selects the archive route
        return f"{type(exc).__name__}: {exc}"[:500]


def read_xlsx(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    """Minimal stdlib reader for the first sheet of Data.xlsx; rows keyed by header text."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    strings = []
    for _, el in ET.iterparse(zf.open("xl/sharedStrings.xml")):
        if el.tag == ns + "si":
            strings.append("".join(t.text or "" for t in el.iter(ns + "t")))
            el.clear()
    rows, header = [], None
    for _, el in ET.iterparse(zf.open(member)):
        if el.tag != ns + "row":
            continue
        cells = {}
        for c in el.findall(ns + "c"):
            v = c.find(ns + "v")
            if v is not None:
                col = re.match(r"[A-Z]+", c.get("r")).group()
                cells[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
        if header is None:
            header = cells
        else:
            rows.append({header[k]: v for k, v in cells.items()})
        el.clear()
    return rows


def load_raw() -> tuple[pd.DataFrame, dict]:
    """Parse WOS46985 X.txt + YL1.txt and verify the label names against Data.xlsx."""
    with zipfile.ZipFile(ARCHIVE) as zf:
        texts = zf.read("WOS46985/X.txt").decode("utf-8").splitlines()
        labels = [int(x) for x in zf.read("WOS46985/YL1.txt").decode("utf-8").split()]
        with zipfile.ZipFile(io.BytesIO(zf.read("Meta-data/Data.xlsx"))) as xlsx:
            sheet = read_xlsx(xlsx, "xl/worksheets/sheet1.xml")
    if len(texts) != len(labels):
        raise RuntimeError(f"X.txt has {len(texts)} rows but YL1.txt has {len(labels)}")
    domains: dict[int, set[str]] = {}
    for r in sheet:
        domains.setdefault(int(r["Y1"]), set()).add(r["Domain"])
    label_map = {}
    for lab, doms in sorted(domains.items()):
        if len(doms) != 1:
            raise RuntimeError(f"level-1 label {lab} maps to several Domain strings: {doms}")
        dom = next(iter(doms))
        label_map[lab] = {"archive_domain": dom, "name": PLAIN_NAMES[dom.strip().lower()]}
    # The sheet row order should be the .txt order; check labels and abstracts agree.
    mismatch = sum(
        1
        for r, t, lab in zip(sheet, texts, labels, strict=False)
        if int(r["Y1"]) != lab or " ".join(r["Abstract"].split()) != " ".join(t.split())
    )
    info = {
        "label_map": {str(k): v for k, v in label_map.items()},
        "xlsx_rows": len(sheet),
        "xlsx_vs_txt_row_mismatches": mismatch if len(sheet) == len(texts) else None,
    }
    return pd.DataFrame({"text": texts, "label": labels}), info


# ---------------------------------------------------------------- cleaning and splitting


def item_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip, drop exact duplicates (after normalisation); drop all copies on label conflict."""
    df = df.assign(text=df["text"].str.strip())
    n_raw = len(df)
    df = df[df["text"] != ""]
    n_empty = n_raw - len(df)
    df = df.assign(key=[dedup_key(t) for t in df["text"]])
    grp = df.groupby("key")["label"]
    size, n_labels = grp.transform("size"), grp.transform("nunique")
    conflict = n_labels > 1
    dup_extra = (size > 1) & ~conflict
    counts = {
        "raw": n_raw,
        "empty_dropped": n_empty,
        "duplicate_groups": int(df.loc[size > 1, "key"].nunique()),
        "conflict_groups": int(df.loc[conflict, "key"].nunique()),
        "conflict_rows_dropped": int(conflict.sum()),
    }
    kept = df[~conflict].drop_duplicates("key", keep="first")
    ok_groups = df.loc[dup_extra, "key"].nunique()
    counts["duplicate_copies_dropped"] = int(dup_extra.sum()) - int(ok_groups)
    kept = kept.assign(item_id=[item_id(k) for k in kept["key"]])
    kept = kept[["item_id", "text", "label"]].sort_values("item_id").reset_index(drop=True)
    counts["clean"] = len(kept)
    if kept["item_id"].duplicated().any():
        raise RuntimeError("item_id collision")
    return kept, counts


def make_splits(
    df: pd.DataFrame, seed: int, test_per_class: int, dev_per_class: int
) -> dict[str, pd.DataFrame]:
    """Class-balanced test and dev, disjoint; pool is the rest. Independent of row order."""
    df = df.sort_values("item_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLIT_NAMES}
    for lab in sorted(df["label"].unique()):
        cls = df[df["label"] == lab]
        if len(cls) < test_per_class + dev_per_class:
            raise ValueError(f"class {lab} has only {len(cls)} items")
        perm = cls.iloc[rng.permutation(len(cls))]
        parts["test"].append(perm.iloc[:test_per_class])
        parts["dev"].append(perm.iloc[test_per_class : test_per_class + dev_per_class])
        parts["pool"].append(perm.iloc[test_per_class + dev_per_class :])
    return {s: pd.concat(p).sort_values("item_id").reset_index(drop=True) for s, p in parts.items()}


def class_counts(df: pd.DataFrame, names: dict[int, str]) -> dict[str, int]:
    vc = df["label"].value_counts().sort_index()
    return {names[int(k)]: int(v) for k, v in vc.items()}


# ---------------------------------------------------------------- length check


def length_stats(lengths: list[int]) -> dict[str, float]:
    a = np.asarray(lengths)
    return {
        "min": int(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def length_check(texts: list[str], names: list[str]) -> dict:
    """Token lengths with Decider's tokenizer: texts alone and the full rendered prompt."""
    from huggingface_hub import snapshot_download
    from strands_decider.data.format import Example
    from strands_decider.prompting import build_prompt
    from transformers import AutoTokenizer

    s = load_settings()
    ckpt = snapshot_download(
        s.decider_model,
        revision=s.decider_revision,
        allow_patterns=[
            "tokenizer*",
            "chat_template.jinja",
            "strands_decider_config.json",
        ],
    )
    max_length = json.loads((Path(ckpt) / "strands_decider_config.json").read_text())["max_length"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    options = [[n, ""] for n in names]
    question = Example(
        kind="choice", state="", instructions=PLACEHOLDER_QUESTION, options=options, label=0
    ).to_question()
    # Same rendering as SystemOneCollator: build_prompt(state, question) with no option shuffle.
    prompts = [build_prompt(t, question, option_order=None)[0] for t in texts]
    text_len, prompt_len = [], []
    for i in range(0, len(texts), 1000):
        text_len += [len(x) for x in tok(texts[i : i + 1000])["input_ids"]]
        prompt_len += [len(x) for x in tok(prompts[i : i + 1000])["input_ids"]]
    return {
        "tokenizer_repo": s.decider_model,
        "tokenizer_revision": s.decider_revision,
        "max_length": max_length,
        "placeholder_question": PLACEHOLDER_QUESTION,
        "text_tokens": length_stats(text_len),
        "prompt_tokens": length_stats(prompt_len),
        "n_over_max_length": int(sum(n > max_length for n in prompt_len)),
        "example_prompt": prompts[0][-400:],
    }


# ---------------------------------------------------------------- stages


def prepare() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sha = download_archive()
    raw, info = load_raw()
    names = {int(k): v["name"] for k, v in info["label_map"].items()}
    clean, counts = clean_frame(raw)
    clean.to_parquet(CLEAN_PARQUET, index=False)
    prov = make_splits(clean, SEED, TEST_PER_CLASS, DEV_PER_CLASS)
    pd.concat([d.assign(split=s) for s, d in prov.items()])[["item_id", "split"]].to_csv(
        PROVISIONAL_CSV, index=False
    )
    summary = {
        "source": {
            "preferred": {"id": HF_ID, "config": HF_CONFIG, "revision": HF_REVISION},
            "preferred_load_status": try_hf_loader(),
            "used": "original archive (URL in the HF loading script at that revision)",
            "archive_url": ARCHIVE_URL,
            "archive_sha256": sha,
            "files": ["WOS46985/X.txt", "WOS46985/YL1.txt", "Meta-data/Data.xlsx"],
        },
        **info,
        "cleaning": counts,
        "per_class": {
            "raw": class_counts(raw, names),
            "clean": class_counts(clean, names),
            "provisional_test": class_counts(
                clean[clean["item_id"].isin(prov["test"]["item_id"])], names
            ),
        },
    }
    PREPARE_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log.info("clean items: %d (%s)", len(clean), counts)


def split() -> None:
    summary = json.loads(PREPARE_JSON.read_text())
    names = {int(k): v["name"] for k, v in summary["label_map"].items()}
    clean = pd.read_parquet(CLEAN_PARQUET)
    prov = pd.read_csv(PROVISIONAL_CSV)
    if not HITS_CSV.exists():
        sys.exit(f"STOP: {HITS_CSV} missing; run scripts.contamination first")
    hits = pd.read_csv(HITS_CSV)
    hit_ids = set(hits["item_id"])
    prov_test = set(prov.loc[prov["split"] == "test", "item_id"])
    share = len(prov_test & hit_ids) / len(prov_test)
    log.info("provisional test items with a 13-gram hit: %.4f%%", 100 * share)
    if share >= MAX_TEST_HIT_SHARE:
        sys.exit(f"STOP: {100 * share:.2f}% of test items have a 13-gram hit (limit 1%)")

    kept = clean[~clean["item_id"].isin(hit_ids)]
    splits = make_splits(kept, SEED, TEST_PER_CLASS, DEV_PER_CLASS)
    pool_counts = class_counts(splits["pool"], names)
    if min(pool_counts.values()) < MIN_POOL_PER_CLASS:
        sys.exit(f"STOP: pool class counts below {MIN_POOL_PER_CLASS}: {pool_counts}")
    ids = {s: set(d["item_id"]) for s, d in splits.items()}
    if ids["test"] & ids["dev"] or ids["test"] & ids["pool"] or ids["dev"] & ids["pool"]:
        sys.exit("STOP: splits overlap")

    names_list = [names[i] for i in sorted(names)]
    lengths = length_check(kept["text"].tolist(), names_list)
    if lengths["n_over_max_length"]:
        write_json("dataset_summary.json", {**summary, "length": lengths})
        sys.exit(f"STOP: {lengths['n_over_max_length']} prompts exceed max_length")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    sha = {}
    for s, d in splits.items():
        path = SPLITS_DIR / f"{s}.csv"
        d[["item_id", "label"]].to_csv(path, index=False)
        sha[s] = sha256_file(path)
    kept.assign(label_name=kept["label"].map(names))[
        ["item_id", "text", "label", "label_name"]
    ].to_parquet(FINAL_PARQUET, index=False)

    n_hit_clean = len(clean) - len(kept)
    summary.update(
        {
            "seed": SEED,
            "contamination": {
                "provisional_test_hit_share": share,
                "items_with_any_hit_removed": n_hit_clean,
                "removed_from_provisional": {
                    s: len(set(prov.loc[prov["split"] == s, "item_id"]) & hit_ids)
                    for s in SPLIT_NAMES
                },
            },
            "n_items": {s: len(d) for s, d in splits.items()},
            "n_final": len(kept),
            "per_class": {
                **{k: v for k, v in summary["per_class"].items() if k != "provisional_test"},
                **{s: class_counts(d, names) for s, d in splits.items()},
            },
            "split_sha256": sha,
            "length": lengths,
        }
    )
    write_json("dataset_summary.json", summary)
    log.info("splits frozen: %s sha256=%s", {s: len(d) for s, d in splits.items()}, sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["prepare", "split"])
    args = parser.parse_args()
    setup_logging(load_settings().log_level)
    {"prepare": prepare, "split": split}[args.stage]()


if __name__ == "__main__":
    main()
