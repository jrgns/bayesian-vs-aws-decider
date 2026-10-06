"""Contamination check: WOS items vs Decider's training-data inventory and corpora.

1. Name check against the model card `datasets:` list, data/sources.md and recipes.py.
2. Exact (full text, first 200 chars) and 13-gram overlap of every cleaned WOS item against
   each corpus, streamed (no corpus is held in memory). Outputs results/contamination.csv,
   results/contamination_hits.csv and results/contamination_name_check.json.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import json
import logging
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.common import (
    DATA_DIR,
    RESULTS_DIR,
    download,
    fetch_text,
    load_settings,
    setup_logging,
    write_json,
)
from scripts.dataset import CLEAN_PARQUET, PROCESSED_DIR, PROVISIONAL_CSV, SPLIT_NAMES, sha256_file
from scripts.ngrams import NGRAM_N, NgramIndex, hash64, tokens

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/strands-labs/strands-decider"
REPO_COMMIT = "3e94e9d84c620ed5a95f1a3310c3decb971e261c"
REPO_DIR = DATA_DIR / "raw" / "strands-decider"
RAW_ZIP_DIR = DATA_DIR / "raw" / "release-zips"
INDEX_DIR = PROCESSED_DIR / "wos_index"
RAW_URL = "https://raw.githubusercontent.com/strands-labs/strands-decider"
CARD_URL = "https://huggingface.co/{model}/resolve/{rev}/README.md"
ALIASES = ("web_of_science", "web of science", "wos", "hdltex", "wos46985", "wos-46985")
PREFIX_CHARS = 200
MAX_REFS_PER_ITEM_CORPUS = 5
MAX_NGRAM_COUNTER = 200_000

# Configs used by recipes.py / sources.md; a missing key means the default config.
HF_CONFIGS: dict[str, list[str | None]] = {
    "ccdv/arxiv-classification": ["no_ref"],
    "clinc/clinc_oos": ["plus"],
    "google-research-datasets/paws": ["labeled_final"],
    "hotpotqa/hotpot_qa": ["distractor"],
    "mteb/amazon_massive_intent": ["en"],
    "nyu-mll/glue": ["mnli", "wnli", "rte", "qnli", "cola"],
    "qiaojin/PubMedQA": ["pqa_labeled"],
}
# Big or slow corpora first so the pool stays busy.
HEAVY_FIRST = (
    "ccdv/arxiv-classification",
    "Yelp/yelp_review_full",
    "community-datasets/yahoo_answers_topics",
    "hotpotqa/hotpot_qa",
    "google/civil_comments",
    "fancyzhx/dbpedia_14",
    "tasksource/ruletaker",
    "tals/vitaminc",
)
ZIPS = {
    "contract-nli.zip": (
        "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
        "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
    ),
    "musique_data_v1.0.zip": (
        "https://drive.usercontent.google.com/download"
        "?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t",
        "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd",
    ),
}

_INDEX: dict[str, NgramIndex] = {}


# ---------------------------------------------------------------- name check


def alias_hits(text: str) -> list[str]:
    low = text.lower()
    return [a for a in ALIASES if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low)]


def name_check() -> tuple[dict, list[str]]:
    """Compare the dataset id / aliases against the pinned inventory; return (result, HF ids)."""
    s = load_settings()
    card = fetch_text(CARD_URL.format(model=s.decider_model, rev=s.decider_revision))
    ids = yaml.safe_load(card.split("---")[1])["datasets"]
    docs = {
        "model_card": card,
        "data/sources.md": fetch_text(f"{RAW_URL}/{REPO_COMMIT}/data/sources.md"),
        "recipes.py": fetch_text(f"{RAW_URL}/{REPO_COMMIT}/src/strands_decider/data/recipes.py"),
        "training/recipe.sh": fetch_text(f"{RAW_URL}/{REPO_COMMIT}/training/recipe.sh"),
    }
    result = {
        "dataset_id": "HDLTex/web_of_science",
        "aliases_searched": list(ALIASES),
        "model_revision": s.decider_revision,
        "code_commit": REPO_COMMIT,
        "card_datasets": ids,
        "dataset_id_in_card_list": "HDLTex/web_of_science" in ids,
        "alias_hits": {k: alias_hits(v) for k, v in docs.items()},
    }
    result["name_match"] = result["dataset_id_in_card_list"] or any(result["alias_hits"].values())
    return result, ids


# ---------------------------------------------------------------- index


def build_index() -> int:
    """Hash every cleaned WOS item: 13-grams, full normalised text, first 200 chars."""
    df = pd.read_parquet(CLEAN_PARQUET)
    toks = [tokens(t) for t in df["text"]]
    joined = [" ".join(t) for t in toks]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    parts = {
        "ngram": NgramIndex.build(toks),
        "full": NgramIndex.from_hashes([hash64(j) for j in joined]),
        "prefix": NgramIndex.from_hashes([hash64(j[:PREFIX_CHARS]) for j in joined]),
    }
    for name, idx in parts.items():
        np.save(INDEX_DIR / f"{name}_keys.npy", idx.keys)
        np.save(INDEX_DIR / f"{name}_items.npy", idx.items)
    log.info("indexed %d items, %d distinct-per-item 13-grams", len(df), len(parts["ngram"].keys))
    return len(df)


def load_index() -> None:
    for name in ("ngram", "full", "prefix"):
        _INDEX[name] = NgramIndex(
            np.load(INDEX_DIR / f"{name}_keys.npy", mmap_mode="r"),
            np.load(INDEX_DIR / f"{name}_items.npy", mmap_mode="r"),
        )


# ---------------------------------------------------------------- scanning


class Acc:
    """Per-corpus accumulator (all plain data so it pickles back from the worker)."""

    def __init__(self) -> None:
        self.rows = 0
        self.ngram_items: set[int] = set()
        self.full_items: set[int] = set()
        self.prefix_items: set[int] = set()
        self.records: list[tuple[int, str, str, str]] = []  # item, type, text, ref
        self.refs_per_item: collections.Counter = collections.Counter()
        self.ngram_counts: collections.Counter = collections.Counter()

    def record(self, item: int, kind: str, text: str, ref: str) -> None:
        if self.refs_per_item[item] < MAX_REFS_PER_ITEM_CORPUS:
            self.records.append((item, kind, text, ref))
        self.refs_per_item[item] += 1


def iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_strings(v)


def scan_text(text: str, ref: str, acc: Acc) -> None:
    toks = tokens(text)
    if not toks:
        return
    joined = " ".join(toks)
    for i in _INDEX["full"].items_for(hash64(joined)):
        acc.full_items.add(i)
        acc.record(i, "exact_full", joined[:100], ref)
    for i in _INDEX["prefix"].items_for(hash64(joined[:PREFIX_CHARS])):
        acc.prefix_items.add(i)
        acc.record(i, "exact_prefix", joined[:100], ref)
    for pos, i in _INDEX["ngram"].lookup(toks):
        gram = " ".join(toks[pos : pos + NGRAM_N])
        acc.ngram_items.add(i)
        acc.record(i, "13gram", gram, ref)
        if gram in acc.ngram_counts or len(acc.ngram_counts) < MAX_NGRAM_COUNTER:
            acc.ngram_counts[gram] += 1


def scan_row(row, ref: str, acc: Acc) -> None:
    acc.rows += 1
    for s in iter_strings(row):
        scan_text(s, ref, acc)


def jsonl_rows(fh):
    for line in fh:
        line = line.strip()
        if line:
            yield json.loads(line)


def file_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        yield from _json_rows(path.name, fh)


def _json_rows(name: str, fh):
    if ".jsonl" in name:
        yield from jsonl_rows(fh)
        return
    data = json.load(fh)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], (dict, list)):
                yield from v
            else:
                yield v


def hf_split_rows(path: str, config: str | None, split: str, notes: list[str]):
    """Stream one split; fall back to the Hub's parquet conversion if the direct load fails."""
    from datasets import load_dataset

    try:
        yield from enumerate(load_dataset(path, config, split=split, streaming=True))
        return
    except Exception as exc:  # noqa: BLE001
        first = f"{type(exc).__name__}: {exc}"[:200]
    notes.append(f"{split}: direct load failed ({first}); used refs/convert/parquet")
    yield from enumerate(
        load_dataset(
            path, config or "default", split=split, streaming=True, revision="refs/convert/parquet"
        )
    )


def scan_hf(path: str, config: str | None, acc: Acc, notes: list[str]) -> list[str]:
    from datasets import get_dataset_split_names

    try:
        splits = get_dataset_split_names(path, config)
    except Exception:  # noqa: BLE001 - script-based datasets: ask the parquet conversion
        splits = get_dataset_split_names(path, config or "default", revision="refs/convert/parquet")
    failures = []
    for split in splits:
        before = acc.rows
        try:
            for i, row in hf_split_rows(path, config, split, notes):
                scan_row(row, f"{split}:{i}", acc)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{split} after {acc.rows - before} rows: {type(exc).__name__}: {exc}")
    return failures


def scan_zip(path: Path, acc: Acc) -> None:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith((".json", ".jsonl")):
                continue
            with io.TextIOWrapper(zf.open(name), encoding="utf-8") as fh:
                for i, row in enumerate(_json_rows(name, fh)):
                    scan_row(row, f"{name}:{i}", acc)


def download_zip(name: str) -> tuple[Path, str]:
    url, expected = ZIPS[name]
    RAW_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_ZIP_DIR / name
    if not dest.exists():
        download(url, dest)
    sha = sha256_file(dest)
    return dest, "sha256 ok" if sha == expected else f"sha256 differs from SHA256SUMS ({sha})"


def scan_corpus(task: dict) -> dict:
    """Worker entry: scan one corpus; never raises (failures become the status)."""
    if not _INDEX:
        load_index()
    acc, notes, failures = Acc(), [], []
    try:
        kind = task["kind"]
        if kind == "repo_file":
            for i, row in enumerate(file_rows(Path(task["path"]))):
                scan_row(row, f"{task['config']}:{i}", acc)
        elif kind == "hf":
            failures = scan_hf(task["corpus"], task["config"], acc, notes)
        elif kind == "zip":
            path, note = download_zip(task["corpus"])
            notes.append(note)
            scan_zip(path, acc)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{type(exc).__name__}: {exc}")
    status = "ok" if not failures else "failed:" + " | ".join(failures)[:600]
    return {
        "corpus": task["corpus"],
        "config": task["config"] or "",
        "rows_scanned": acc.rows,
        "status": status,
        "notes": " | ".join(notes)[:600],
        "ngram_items": sorted(acc.ngram_items),
        "full_items": sorted(acc.full_items),
        "prefix_items": sorted(acc.prefix_items),
        "records": acc.records,
        "top_ngrams": acc.ngram_counts.most_common(50),
    }


# ---------------------------------------------------------------- orchestration


def make_tasks(hf_ids: list[str]) -> list[dict]:
    tasks = []
    if not (REPO_DIR / "data").is_dir():
        raise RuntimeError(f"{REPO_DIR} missing: git clone {REPO_URL} and checkout {REPO_COMMIT}")
    for p in sorted((REPO_DIR / "data").rglob("*")):
        if p.suffix in (".jsonl", ".json") and ".git" not in p.parts:
            rel = str(p.relative_to(REPO_DIR))
            tasks.append(
                {
                    "kind": "repo_file",
                    "corpus": f"strands-decider@3e94e9d8:{rel}",
                    "config": rel,
                    "path": str(p),
                }
            )
    order = {hf: n for n, hf in enumerate(HEAVY_FIRST)}
    for hf in sorted(hf_ids, key=lambda i: (order.get(i, len(order)), i)):
        tasks += [{"kind": "hf", "corpus": hf, "config": c} for c in HF_CONFIGS.get(hf, [None])]
    tasks += [{"kind": "zip", "corpus": z, "config": None} for z in ZIPS]
    # Heavy corpora first; repo files are small and go last.
    tasks.sort(key=lambda t: t["kind"] == "repo_file")
    return tasks


def run(workers: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nc, hf_ids = name_check()
    clean = pd.read_parquet(CLEAN_PARQUET, columns=["item_id"])
    ids = clean["item_id"].tolist()
    prov = pd.read_csv(PROVISIONAL_CSV).set_index("item_id")["split"]
    split_of = [prov[i] for i in ids]

    sums = REPO_DIR / "data" / "SHA256SUMS"
    if sums.exists():
        listed = [ln.split()[1] for ln in sums.read_text().splitlines() if ln.strip()]
        nc["sha256sums_listed_but_absent_from_repo"] = [
            f for f in listed if not (REPO_DIR / f).exists()
        ]
    write_json("contamination_name_check.json", nc)
    log.info("name check: match=%s", nc["name_match"])

    build_index()
    tasks = make_tasks(hf_ids)
    log.info("scanning %d corpora with %d workers", len(tasks), workers)
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=load_index) as pool:
        futs = {pool.submit(scan_corpus, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            log.info(
                "%s [%s]: %d rows, %d items with 13-gram hits, %s",
                r["corpus"],
                r["config"],
                r["rows_scanned"],
                len(r["ngram_items"]),
                r["status"][:100],
            )
    write_outputs(results, ids, split_of)


def write_outputs(results: list[dict], ids: list[str], split_of: list[str]) -> None:
    results.sort(key=lambda r: (r["corpus"], r["config"]))
    cols = ["corpus", "config", "rows_scanned", "status"]
    cols += [f"wos_items_with_13gram_hit_{s}" for s in SPLIT_NAMES]
    cols += ["exact_full_matches", "exact_prefix_matches", "notes"]
    all_ngram: set[int] = set()
    with (RESULTS_DIR / "contamination.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in ("corpus", "config", "rows_scanned", "status", "notes")}
            for s in SPLIT_NAMES:
                row[f"wos_items_with_13gram_hit_{s}"] = sum(
                    split_of[i] == s for i in r["ngram_items"]
                )
            row["exact_full_matches"] = len(r["full_items"])
            row["exact_prefix_matches"] = len(r["prefix_items"])
            w.writerow(row)
            all_ngram.update(r["ngram_items"])
    hit_items = set()
    with (RESULTS_DIR / "contamination_hits.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "split", "corpus", "config", "match_type", "text", "source_ref"])
        for r in results:
            for item, kind, text, ref in r["records"]:
                hit_items.add(item)
                w.writerow([ids[item], split_of[item], r["corpus"], r["config"], kind, text, ref])
    counts: collections.Counter = collections.Counter()
    for r in results:
        counts.update(dict(r["top_ngrams"]))
    with (RESULTS_DIR / "contamination_top_ngrams.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ngram", "count_in_top_lists"])
        w.writerows(counts.most_common(50))
    by_split = collections.Counter(split_of[i] for i in all_ngram | hit_items)
    totals = collections.Counter(split_of)
    log.info("items with any hit by provisional split: %s (of %s)", dict(by_split), dict(totals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=int(os.environ.get("CONTAM_WORKERS", 8)))
    args = parser.parse_args()
    setup_logging(load_settings().log_level)
    run(args.workers)


if __name__ == "__main__":
    main()
