"""Meter sanity check: 60 s idle baseline, then a CPU-only and a GPU-only busy window.

    python -m scripts.energy_sanity [--idle-seconds 60] [--busy-seconds 30]

Shows that the counters move when they should. Records land in results/energy.csv under
run_id `sanity`; the summary goes to results/energy_sanity.json.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time

from scripts.common import load_settings, setup_logging, write_json
from scripts.energy import EnergyRecord, Meter

log = logging.getLogger(__name__)


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    x = 0
    while time.perf_counter() < end:
        x += 1


def cpu_busy(seconds: float, workers: int) -> None:
    procs = [mp.Process(target=_spin, args=(seconds,)) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def gpu_busy(seconds: float) -> None:
    import torch

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def brief(rec: EnergyRecord) -> dict:
    return {
        "duration_s": rec.duration_s,
        "gpu_J": rec.gpu_J,
        "cpu_pkg_J": rec.cpu_pkg_J,
        "gpu_W_counter": rec.gpu_J / rec.duration_s,
        "gpu_W_nvml_samples": rec.gpu_mean_power_W,
        "cpu_pkg_W": rec.cpu_pkg_J / rec.duration_s,
        "rapl_J": rec.rapl_J,
        "loadavg_start": rec.loadavg_start,
        "loadavg_end": rec.loadavg_end,
        "host_cpu_util": rec.host_cpu_util,
        "host_containers_running": rec.host_containers_running,
        "gpu_uuid": rec.gpu_uuid,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--busy-seconds", type=float, default=30)
    args = ap.parse_args(argv)
    setup_logging(load_settings().log_level)

    meter = Meter("sanity")
    idle = meter.idle_baseline(args.idle_seconds)
    workers = len(os.sched_getaffinity(0))
    with meter.measure("cpu_busy", idle) as cpu:
        cpu_busy(args.busy_seconds, workers)
    with meter.measure("gpu_busy", idle) as gpu:
        gpu_busy(args.busy_seconds)
    write_json(
        "energy_sanity.json",
        {
            "idle": {
                **brief(idle.record),
                "idle_gpu_W": idle.gpu_W,
                "idle_cpu_pkg_W": idle.cpu_pkg_W,
            },
            "cpu_busy": {**brief(cpu.record), "cpu_workers": workers,
                         "cpu_pkg_marginal_J": cpu.record.cpu_pkg_marginal_J},
            "gpu_busy": {**brief(gpu.record), "gpu_marginal_J": gpu.record.gpu_marginal_J},
        },
    )  # fmt: skip


if __name__ == "__main__":
    main()
