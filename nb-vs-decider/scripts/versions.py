"""Record every version that matters for reproducibility to results/versions.json."""

from __future__ import annotations

import json
import logging
import platform
from importlib import metadata

from scripts.common import load_settings, prepare_checkpoint, setup_logging, write_json

log = logging.getLogger(__name__)


def cpu_model() -> str | None:
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return None


def decider_install_source() -> dict:
    """Git commit if strands-decider was installed from a VCS URL, else None."""
    dist = metadata.distribution("strands-decider")
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    vcs = info.get("vcs_info", {})
    return {"url": info.get("url"), "git_commit": vcs.get("commit_id")}


def gpu_info() -> dict:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # the only GPU visible to the container
        return {
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "name": pynvml.nvmlDeviceGetName(handle),
            "uuid": pynvml.nvmlDeviceGetUUID(handle),
            "visible_gpu_count": pynvml.nvmlDeviceGetCount(),
        }
    finally:
        pynvml.nvmlShutdown()


def main() -> None:
    s = load_settings()
    setup_logging(s.log_level)
    import torch

    ckpt = prepare_checkpoint(s)
    data = {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_info(),
        "decider_package_version": metadata.version("strands-decider"),
        "decider_install_source": decider_install_source(),
        "decider_model_id": ckpt.model_id,
        "decider_model_revision": ckpt.model_revision,
        "base_model_id": ckpt.base_model,
        "base_model_revision": ckpt.base_revision,
        "image": s.image,
    }
    write_json("versions.json", data)


if __name__ == "__main__":
    main()
