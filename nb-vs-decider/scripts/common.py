"""Shared helpers: logging, settings (python-dotenv), pinned Decider checkpoint preparation."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(os.environ.get("APP_DIR", Path(__file__).resolve().parent.parent))
RESULTS_DIR = APP_DIR / "results"
DATA_DIR = APP_DIR / "data"

# Pinned defaults; .env / environment variables override (see .env.example).
DEFAULT_MODEL = "StrandsAgents/strands-decider-2B-hobson-v21"
DEFAULT_REVISION = "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
DEFAULT_BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    decider_model: str
    decider_revision: str
    base_model: str
    base_revision: str
    log_level: str
    image: str | None


def load_settings() -> Settings:
    load_dotenv(APP_DIR / ".env")
    return Settings(
        decider_model=os.environ.get("DECIDER_MODEL", DEFAULT_MODEL),
        decider_revision=os.environ.get("DECIDER_REVISION", DEFAULT_REVISION),
        base_model=os.environ.get("DECIDER_BASE_MODEL", DEFAULT_BASE_MODEL),
        base_revision=os.environ.get("DECIDER_BASE_REVISION", DEFAULT_BASE_REVISION),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        image=os.environ.get("DECIDER_IMAGE") or None,
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def write_json(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log.info("wrote %s", path)
    return path


def _open_url(url: str):
    # Some hosts (Mendeley's Cloudflare) refuse urllib's default user agent.
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    return urllib.request.urlopen(req, timeout=120)


def fetch_text(url: str) -> str:
    with _open_url(url) as r:
        return r.read().decode("utf-8")


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (via a .part file, so an interrupted download is never kept)."""
    part = dest.with_name(dest.name + ".part")
    with _open_url(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(dest)


@dataclass(frozen=True)
class Checkpoint:
    path: str  # local dir passed to StrandsDeciderModel.load
    model_id: str
    model_revision: str  # SHA of the snapshot actually loaded
    base_model: str
    base_revision: str  # SHA of the base snapshot actually loaded


def _snapshot(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, revision=revision))
    sha = path.name  # HF cache snapshot dirs are named by commit SHA
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or sha != revision:
        raise RuntimeError(f"{repo}: asked for {revision}, got snapshot {sha}")
    return path


def prepare_checkpoint(s: Settings) -> Checkpoint:
    """Download both repos at pinned revisions and return a local checkpoint dir.

    Package versions without `base_revision` support always load the base model's default
    branch. To pin the base anyway, build a checkpoint dir whose config points `base_model`
    at the local snapshot of the pinned base revision.
    """
    ckpt = _snapshot(s.decider_model, s.decider_revision)
    base = _snapshot(s.base_model, s.base_revision)
    work = Path(os.environ.get("CKPT_WORK_DIR", "/tmp/decider-ckpt"))
    work.mkdir(parents=True, exist_ok=True)
    for src in ckpt.iterdir():
        dst = work / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    for cfg_name in ("strands_decider_config.json", "hobson_config.json"):
        cfg_file = work / cfg_name
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            cfg_file.unlink()  # replace the symlink with a patched copy
            if cfg["base_model"] != s.base_model:
                raise RuntimeError(f"checkpoint base {cfg['base_model']} != {s.base_model}")
            cfg["base_model"] = str(base)
            cfg_file.write_text(json.dumps(cfg, indent=2))
            break
    else:
        raise RuntimeError(f"no Decider config file in {ckpt}")
    return Checkpoint(str(work), s.decider_model, ckpt.name, s.base_model, base.name)
