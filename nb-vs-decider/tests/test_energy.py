from pathlib import Path

import pytest

from scripts import energy
from scripts.energy import EnergyError, Meter, RaplReader, rapl_delta_uj


def make_domain(parent: Path, name: str, key: str, uj: int, max_uj: int = 1000) -> Path:
    d = parent / name
    d.mkdir(parents=True)
    (d / "name").write_text(key + "\n")
    (d / "energy_uj").write_text(f"{uj}\n")
    (d / "max_energy_range_uj").write_text(f"{max_uj}\n")
    return d


@pytest.fixture
def tree(tmp_path):
    base = tmp_path / "intel-rapl"
    pkg = make_domain(base, "intel-rapl:0", "package-0", 100)
    make_domain(pkg, "intel-rapl:0:0", "core", 40)
    return tmp_path


def test_wraparound_delta():
    assert rapl_delta_uj(100, 250, 1000) == 150
    assert rapl_delta_uj(900, 50, 1000) == 150  # counter wrapped past max_energy_range_uj


def test_discovers_every_domain(tree):
    r = RaplReader(tree)
    assert [d.key for d in r.domains] == ["package-0", "core"]
    assert r.read_uj() == {"package-0": 100, "core": 40}


def test_delta_in_joules_with_wrap(tree):
    r = RaplReader(tree)
    start = r.read_uj()
    (tree / "intel-rapl/intel-rapl:0/energy_uj").write_text("50")  # 100 -> wrapped -> 50
    (tree / "intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj").write_text("1040")
    d = r.delta_j(start, r.read_uj())
    assert d["package-0"] == pytest.approx(950 / 1e6)
    assert d["core"] == pytest.approx(1000 / 1e6)


def test_duplicate_names_get_distinct_keys(tmp_path):
    base = tmp_path / "intel-rapl"
    make_domain(base, "intel-rapl:0", "package-0", 1)
    make_domain(base, "intel-rapl:1", "package-0", 2)
    assert len({d.key for d in RaplReader(tmp_path).domains}) == 2


def test_unreadable_counter_raises(tree):
    path = tree / "intel-rapl/intel-rapl:0/energy_uj"
    path.write_text("not a number")
    with pytest.raises(EnergyError):
        RaplReader(tree)


def test_counter_vanishing_later_raises(tree):
    r = RaplReader(tree)
    (tree / "intel-rapl/intel-rapl:0/energy_uj").unlink()
    with pytest.raises(EnergyError):
        r.read_uj()


def test_missing_root_or_package_raises(tmp_path):
    with pytest.raises(EnergyError):
        RaplReader(tmp_path / "absent")
    make_domain(tmp_path / "intel-rapl", "intel-rapl:0", "dram", 1)
    with pytest.raises(EnergyError, match="package-0"):
        RaplReader(tmp_path)


class FakeGpu:
    uuid = "GPU-fake"

    def __init__(self):
        self.mj = 1_000_000

    def energy_mj(self):
        self.mj += 5_000  # 5 J per read
        return self.mj

    def power_w(self):
        return 100.0


@pytest.fixture
def proc(tmp_path):
    p = tmp_path / "proc"
    p.mkdir()
    (p / "loadavg").write_text("0.50 0.40 0.30 1/200 123\n")
    (p / "stat").write_text("cpu 100 0 100 700 100 0 0 0 0 0\n")
    return p


def test_meter_record_and_marginal(tree, proc, tmp_path):
    out = tmp_path / "results"
    meter = Meter("t1", gpu=FakeGpu(), rapl=RaplReader(tree), results_dir=out, proc=proc)
    zero = {"package-0": 0.0, "core": 0.0}
    idle = energy.IdleBaseline(None, gpu_W=1.0, cpu_pkg_W=0.0, rapl_W=zero)
    with meter.measure("phase", idle) as m:
        (tree / "intel-rapl/intel-rapl:0/energy_uj").write_text("300")
        (proc / "stat").write_text("cpu 200 0 200 800 100 0 0 0 0 0\n")
    r = m.record
    assert r.gpu_J == pytest.approx(5.0)
    assert r.cpu_pkg_J == pytest.approx(200 / 1e6)
    assert r.host_cpu_util == pytest.approx(200 / 300)
    assert r.loadavg_start == (0.5, 0.4, 0.3)
    assert r.gpu_marginal_J == pytest.approx(5.0 - r.duration_s)
    assert r.wall == {"wall": "not configured"}
    row = (out / "energy.csv").read_text().splitlines()
    assert "rapl_core_J" in row[0] and "gpu_marginal_J" in row[0] and len(row) == 2
    assert (out / "power_samples" / "t1__phase.csv").exists()


def test_meter_block_error_writes_nothing(tree, proc, tmp_path):
    meter = Meter("t2", gpu=FakeGpu(), rapl=RaplReader(tree), results_dir=tmp_path, proc=proc)
    with pytest.raises(ZeroDivisionError), meter.measure("boom"):
        raise ZeroDivisionError
    assert not (tmp_path / "energy.csv").exists()


def test_idle_baseline_watts(tree, proc, tmp_path):
    meter = Meter("t3", gpu=FakeGpu(), rapl=RaplReader(tree), results_dir=tmp_path, proc=proc)
    idle = meter.idle_baseline(0.2)
    assert idle.gpu_W == pytest.approx(5.0 / idle.record.duration_s)
    assert idle.cpu_pkg_W == 0.0


def test_csv_columns_identical_with_and_without_idle(tree, proc, tmp_path):
    meter = Meter("t4", gpu=FakeGpu(), rapl=RaplReader(tree), results_dir=tmp_path, proc=proc)
    idle = meter.idle_baseline(0.1)
    with meter.measure("after", idle):
        pass
    assert len((tmp_path / "energy.csv").read_text().splitlines()) == 3
