from scripts.decider_smoke import compare


def test_within_tolerance_passes():
    out = compare({"a": 0.5}, {"a": 0.504}, 0.005)
    assert out["a"]["pass"] is True
    assert abs(out["a"]["abs_diff"] - 0.004) < 1e-12


def test_outside_tolerance_fails():
    assert compare({"a": 0.5}, {"a": 0.51}, 0.005)["a"]["pass"] is False


def test_boundary_is_inclusive():
    assert compare({"a": 0.5}, {"a": 0.5}, 0.0)["a"]["pass"] is True


def test_missing_actual_fails():
    out = compare({"a": 0.5}, {}, 0.005)
    assert out["a"]["pass"] is False
    assert out["a"]["actual"] is None
