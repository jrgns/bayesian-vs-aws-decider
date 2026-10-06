"""Hardware-counter energy meter. Counters only: an unreadable counter raises, never estimates.

GPU: NVML total-energy counter (mJ) plus ~10 Hz instantaneous power samples.
CPU: RAPL `energy_uj` for every `intel-rapl:*` domain under RAPL_ROOT, with wrap-around.
`package-0` is the CPU figure. Both counters cover the whole host, not just this process, so
every record also carries host-noise indicators (loadavg, /proc/stat utilisation).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from scripts.common import RESULTS_DIR, append_csv

log = logging.getLogger(__name__)

RAPL_ROOT = Path(os.environ.get("RAPL_ROOT", "/host/powercap"))
CPU_DOMAIN = "package-0"
SAMPLE_HZ = 10
ENERGY_CSV = "energy.csv"
SAMPLES_DIR = "power_samples"


class EnergyError(RuntimeError):
    """A hardware counter is missing or unreadable. Never work around this with an estimate."""


# ------------------------------------------------------------------------------ RAPL


@dataclass(frozen=True)
class RaplDomain:
    key: str  # the domain's `name` file, e.g. package-0, core
    energy_path: Path
    max_range_uj: int


def read_int(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise EnergyError(f"cannot read counter {path}: {exc}") from exc


def _domain_dirs(root: Path) -> Iterator[Path]:
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.is_symlink() and d.name.startswith("intel-rapl"):
            if (d / "name").exists():
                yield d
            yield from _domain_dirs(d)


def discover_rapl(root: Path = RAPL_ROOT) -> list[RaplDomain]:
    """Every RAPL domain below `root` (package and nested sub-domains). Raises if none or if
    `package-0` is missing. Each counter is read once here so an unreadable one fails early."""
    if not root.is_dir():
        raise EnergyError(f"RAPL root {root} not found (bind-mount the host powercap tree)")
    domains: list[RaplDomain] = []
    seen: set[str] = set()
    for d in _domain_dirs(root):
        key = (d / "name").read_text().strip()
        if key in seen:
            key = f"{key}@{d.name}"
        seen.add(key)
        dom = RaplDomain(key, d / "energy_uj", read_int(d / "max_energy_range_uj"))
        read_int(dom.energy_path)
        domains.append(dom)
    if not any(d.key == CPU_DOMAIN for d in domains):
        raise EnergyError(f"no {CPU_DOMAIN} RAPL domain under {root}: {[d.key for d in domains]}")
    return domains


def rapl_delta_uj(start: int, end: int, max_range_uj: int) -> int:
    """Counter difference, assuming at most one wrap-around between the two reads."""
    return end - start if end >= start else end + max_range_uj - start


class RaplReader:
    def __init__(self, root: Path = RAPL_ROOT):
        self.domains = discover_rapl(root)

    def read_uj(self) -> dict[str, int]:
        return {d.key: read_int(d.energy_path) for d in self.domains}

    def delta_j(self, start: dict[str, int], end: dict[str, int]) -> dict[str, float]:
        return {
            d.key: rapl_delta_uj(start[d.key], end[d.key], d.max_range_uj) / 1e6
            for d in self.domains
        }


# ------------------------------------------------------------------------------ GPU


class NvmlGpu:
    """One NVML device. Index 0 is the host GPU given to the container (DECIDER_GPU)."""

    def __init__(self, index: int = 0):
        import pynvml

        self._nvml = pynvml
        try:
            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.uuid = pynvml.nvmlDeviceGetUUID(self._h)
            self.energy_mj()
        except pynvml.NVMLError as exc:
            raise EnergyError(f"NVML energy counter unavailable on GPU {index}: {exc}") from exc

    def energy_mj(self) -> int:
        try:
            return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h))
        except self._nvml.NVMLError as exc:
            raise EnergyError(f"NVML total energy read failed: {exc}") from exc

    def power_w(self) -> float:
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        except self._nvml.NVMLError as exc:
            raise EnergyError(f"NVML power read failed: {exc}") from exc


class PowerSampler:
    """Background thread sampling instantaneous power at SAMPLE_HZ as (perf_counter, watts)."""

    def __init__(self, read_w: Callable[[], float], hz: float = SAMPLE_HZ):
        self._read_w = read_w
        self._period = 1.0 / hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[tuple[float, float]] = []
        self.error: Exception | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.perf_counter(), self._read_w()))
            except EnergyError as exc:
                self.error = exc
                return
            self._stop.wait(self._period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[tuple[float, float]]:
        self._stop.set()
        self._thread.join()
        if self.error:
            raise self.error
        return self.samples


# ------------------------------------------------------------------------------ host noise


def read_loadavg(proc: Path = Path("/proc")) -> tuple[float, float, float]:
    a, b, c = (proc / "loadavg").read_text().split()[:3]
    return float(a), float(b), float(c)


def read_cpu_times(proc: Path = Path("/proc")) -> tuple[int, int]:
    """(idle+iowait jiffies, total jiffies) of the aggregate `cpu` line of /proc/stat."""
    fields = [int(x) for x in (proc / "stat").read_text().splitlines()[0].split()[1:]]
    return fields[3] + fields[4], sum(fields[:8])  # user..steal; guest is already in user


def cpu_utilisation(start: tuple[int, int], end: tuple[int, int]) -> float | None:
    total = end[1] - start[1]
    return None if total <= 0 else 1.0 - (end[0] - start[0]) / total


def host_containers_running() -> int | None:
    """Set by run_all.sh from the host (a container cannot reach the docker daemon)."""
    raw = os.environ.get("HOST_CONTAINERS_RUNNING", "")
    return int(raw) if raw.strip().isdigit() else None


# ------------------------------------------------------------------------------ wall power


def wall_energy(t0: float, t1: float) -> dict:
    """Optional Prometheus cross-check: mean of PROM_WALL_QUERY over the window x duration."""
    url, query = os.environ.get("PROM_URL"), os.environ.get("PROM_WALL_QUERY")
    if not url or not query:
        return {"wall": "not configured"}
    step = 1 if t1 - t0 < 30 else 5
    params = urllib.parse.urlencode({"query": query, "start": t0, "end": t1, "step": step})
    try:
        endpoint = f"{url.rstrip('/')}/api/v1/query_range?{params}"
        with urllib.request.urlopen(endpoint, timeout=10) as r:
            result = json.load(r)["data"]["result"]
        values = [float(v) for _, v in result[0]["values"]]
        mean_w = sum(values) / len(values)
    except Exception as exc:  # noqa: BLE001 - optional cross-check must never fail a run
        log.warning("wall power query failed: %s", exc)
        return {"wall": f"error: {exc}"}
    return {"wall": "ok", "wall_mean_W": mean_w, "wall_J": mean_w * (t1 - t0)}


# ------------------------------------------------------------------------------ records


@dataclass
class EnergyRecord:
    energy_id: str
    run_id: str
    label: str
    t_start_wall: float
    t_end_wall: float
    t_start_perf: float
    t_end_perf: float
    duration_s: float
    gpu_uuid: str
    gpu_J: float
    cpu_pkg_J: float
    rapl_J: dict[str, float]
    gpu_mean_power_W: float | None  # mean of the NVML samples inside the window
    gpu_power_samples: int
    loadavg_start: tuple[float, float, float]
    loadavg_end: tuple[float, float, float]
    host_cpu_util: float | None
    host_containers_running: int | None
    wall: dict = field(default_factory=dict)
    idle_gpu_W: float | None = None
    idle_cpu_pkg_W: float | None = None
    gpu_marginal_J: float | None = None
    cpu_pkg_marginal_J: float | None = None
    rapl_marginal_J: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            k: getattr(self, k)
            for k in (
                "energy_id", "run_id", "label", "t_start_wall", "t_end_wall", "t_start_perf",
                "t_end_perf", "duration_s", "gpu_uuid", "gpu_J", "cpu_pkg_J",
            )
        }  # fmt: skip
        row.update({f"rapl_{k}_J": v for k, v in sorted(self.rapl_J.items())})
        row.update(
            gpu_mean_power_W=self.gpu_mean_power_W,
            gpu_power_samples=self.gpu_power_samples,
            host_cpu_util=self.host_cpu_util,
            host_containers_running=self.host_containers_running,
        )
        for name, vals in (("start", self.loadavg_start), ("end", self.loadavg_end)):
            row.update({f"loadavg_{name}_{m}m": v for m, v in zip((1, 5, 15), vals, strict=True)})
        row.update(
            wall=self.wall.get("wall"),
            wall_mean_W=self.wall.get("wall_mean_W"),
            wall_J=self.wall.get("wall_J"),
            idle_gpu_W=self.idle_gpu_W,
            idle_cpu_pkg_W=self.idle_cpu_pkg_W,
            gpu_marginal_J=self.gpu_marginal_J,
            cpu_pkg_marginal_J=self.cpu_pkg_marginal_J,
        )
        row.update(
            {f"rapl_{k}_marginal_J": self.rapl_marginal_J.get(k) for k in sorted(self.rapl_J)}
        )
        return row


@dataclass(frozen=True)
class IdleBaseline:
    record: EnergyRecord
    gpu_W: float
    cpu_pkg_W: float
    rapl_W: dict[str, float]


def idle_from_record(rec: EnergyRecord) -> IdleBaseline:
    d = rec.duration_s
    return IdleBaseline(
        rec, rec.gpu_J / d, rec.cpu_pkg_J / d, {k: v / d for k, v in rec.rapl_J.items()}
    )


class Measurement:
    """Handed out by Meter.measure(); `.record` is set when the block exits cleanly."""

    record: EnergyRecord | None = None


@dataclass
class _Snap:
    wall: float
    perf: float
    gpu_mj: int
    rapl_uj: dict[str, int]
    load: tuple[float, float, float]
    cpu_times: tuple[int, int]


class Meter:
    def __init__(
        self,
        run_id: str,
        *,
        gpu=None,
        rapl: RaplReader | None = None,
        results_dir: Path = RESULTS_DIR,
        proc: Path = Path("/proc"),
    ):
        self.run_id = run_id
        self.gpu = gpu if gpu is not None else NvmlGpu(0)
        self.rapl = rapl if rapl is not None else RaplReader()
        self.results_dir = results_dir
        self.proc = proc

    def _snap(self) -> _Snap:
        return _Snap(
            time.time(),
            time.perf_counter(),
            self.gpu.energy_mj(),
            self.rapl.read_uj(),
            read_loadavg(self.proc),
            read_cpu_times(self.proc),
        )

    @contextmanager
    def measure(self, label: str, idle: IdleBaseline | None = None) -> Iterator[Measurement]:
        """Measure the enclosed block. Callers must synchronise the GPU before leaving it."""
        m = Measurement()
        sampler = PowerSampler(self.gpu.power_w)
        sampler.start()
        try:
            start = self._snap()
            yield m
            end = self._snap()
        finally:
            samples = sampler.stop()
        m.record = self._finish(label, start, end, samples, idle)
        append_csv(self.results_dir / ENERGY_CSV, m.record.to_row())
        self._save_samples(m.record, samples)
        log.info(
            "energy %s: %.1f s, gpu %.1f J, cpu pkg %.1f J%s",
            m.record.energy_id,
            m.record.duration_s,
            m.record.gpu_J,
            m.record.cpu_pkg_J,
            ""
            if m.record.gpu_marginal_J is None
            else f" (marginal gpu {m.record.gpu_marginal_J:.1f} J, "
            f"cpu {m.record.cpu_pkg_marginal_J:.1f} J)",
        )

    def _finish(self, label, a: _Snap, b: _Snap, samples, idle) -> EnergyRecord:
        dur = b.perf - a.perf
        rapl_j = self.rapl.delta_j(a.rapl_uj, b.rapl_uj)
        win = [w for t, w in samples if a.perf <= t <= b.perf]
        rec = EnergyRecord(
            energy_id=f"{self.run_id}:{label}",
            run_id=self.run_id,
            label=label,
            t_start_wall=a.wall,
            t_end_wall=b.wall,
            t_start_perf=a.perf,
            t_end_perf=b.perf,
            duration_s=dur,
            gpu_uuid=self.gpu.uuid,
            gpu_J=(b.gpu_mj - a.gpu_mj) / 1000.0,
            cpu_pkg_J=rapl_j[CPU_DOMAIN],
            rapl_J=rapl_j,
            gpu_mean_power_W=sum(win) / len(win) if win else None,
            gpu_power_samples=len(win),
            loadavg_start=a.load,
            loadavg_end=b.load,
            host_cpu_util=cpu_utilisation(a.cpu_times, b.cpu_times),
            host_containers_running=host_containers_running(),
            wall=wall_energy(a.wall, b.wall),
        )
        if idle is not None:
            rec.idle_gpu_W, rec.idle_cpu_pkg_W = idle.gpu_W, idle.cpu_pkg_W
            rec.gpu_marginal_J = rec.gpu_J - idle.gpu_W * dur
            rec.cpu_pkg_marginal_J = rec.cpu_pkg_J - idle.cpu_pkg_W * dur
            rec.rapl_marginal_J = {k: v - idle.rapl_W[k] * dur for k, v in rapl_j.items()}
        return rec

    def _save_samples(self, rec: EnergyRecord, samples) -> None:
        out = self.results_dir / SAMPLES_DIR
        out.mkdir(parents=True, exist_ok=True)
        name = rec.energy_id.replace(":", "__").replace("/", "_")
        lines = ["t_s,gpu_W"] + [f"{t - rec.t_start_perf:.3f},{w:.3f}" for t, w in samples]
        (out / f"{name}.csv").write_text("\n".join(lines) + "\n")

    def idle_baseline(self, seconds: float = 60, label: str = "idle") -> IdleBaseline:
        """Same counters while sleeping. Includes the noise indicators of the idle window."""
        with self.measure(label) as m:
            time.sleep(seconds)
        return idle_from_record(m.record)
