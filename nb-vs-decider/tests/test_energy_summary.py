import math

import pytest

from scripts import energy_summary as es


def test_kwh_conversion():
    assert es.kwh(3.6e6) == 1.0
    assert es.kwh(29.6 * 1000) == pytest.approx(0.0082222, rel=1e-4)


def test_break_even_is_train_over_per_prediction_saving():
    # 300 J to train, saving 29.6 - 0.005 J per prediction
    assert es.break_even(300.0, 29.6, 0.005) == pytest.approx(300 / 29.595)
    assert es.break_even(0.0, 10.0, 1.0) == 0.0


def test_break_even_infinite_when_nb_is_not_cheaper():
    assert math.isinf(es.break_even(5.0, 1.0, 1.0))
    assert math.isinf(es.break_even(5.0, 1.0, 2.0))


def test_break_even_scales_linearly_with_training_energy():
    assert es.break_even(200.0, 10.0, 0.0) == 2 * es.break_even(100.0, 10.0, 0.0)
