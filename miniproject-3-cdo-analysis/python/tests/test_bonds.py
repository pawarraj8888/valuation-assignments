import numpy as np
import pytest

from cdo.bonds import bis_cash_flows, bis_expected_value, promised_cash_flows


def test_promised_cash_flows_of_a_collateral_bond():
    cf = promised_cash_flows(10.0, 0.06, 20, 4)
    assert cf.shape == (20,)
    assert np.allclose(cf[:-1], 0.15)
    assert cf[-1] == pytest.approx(10.15)
    assert cf.sum() == pytest.approx(13.0)


def test_bond_that_never_defaults_pays_what_it_promised():
    promised = promised_cash_flows(10.0, 0.06, 20, 4)
    assert np.array_equal(bis_cash_flows(promised, 21, 0.60), promised)


def test_default_reduces_the_payment_of_the_default_period_and_all_later_ones():
    promised = promised_cash_flows(10.0, 0.06, 20, 4)
    cf = bis_cash_flows(promised, 3, 0.60)
    assert np.allclose(cf[:2], 0.15)
    assert np.allclose(cf[2:-1], 0.06)
    assert cf[-1] == pytest.approx(4.06)


def test_default_in_the_first_period_leaves_the_recovery_floor():
    promised = promised_cash_flows(10.0, 0.06, 20, 4)
    assert np.allclose(bis_cash_flows(promised, 1, 0.60), 0.40 * promised)


def test_lgd_of_zero_makes_default_harmless_and_lgd_of_one_stops_all_payments():
    promised = promised_cash_flows(10.0, 0.06, 20, 4)
    assert np.array_equal(bis_cash_flows(promised, 5, 0.0), promised)
    stopped = bis_cash_flows(promised, 5, 1.0)
    assert np.allclose(stopped[:4], 0.15)
    assert np.allclose(stopped[4:], 0.0)


def test_bis_cash_flows_broadcast_over_cases_and_bonds():
    promised = promised_cash_flows(10.0, 0.06, 20, 4)
    periods = np.array([[1, 21, 7], [20, 3, 21]])
    cf = bis_cash_flows(promised, periods, 0.60)
    assert cf.shape == (2, 3, 20)
    assert np.array_equal(cf[0, 1], promised)
    assert cf[1, 0, -1] == pytest.approx(4.06)
    assert cf[1, 0, -2] == pytest.approx(0.15)


def test_bis_function_reproduces_the_lecture_slide():
    # Lecture 5, "The BIS Defaultable Bond Model": face 1000, 5.5% annual coupon, LGD 60%, pi 3%, r 4%.
    promised = promised_cash_flows(1000.0, 0.055, 5, 1)
    rows, value = bis_expected_value(promised, 0.03, 0.60, 0.04)
    assert value == pytest.approx(984.73, abs=0.005)
    assert [round(pv, 2) for _, _, pv in rows] == [426.71, 458.44, 488.95, 518.29, 546.50, 1066.78]
    assert [round(prob, 4) for _, prob, _ in rows] == [0.0300, 0.0291, 0.0282, 0.0274, 0.0266, 0.8587]
    assert [period for period, _, _ in rows] == [1, 2, 3, 4, 5, None]


def test_default_probabilities_on_the_slide_sum_to_one():
    promised = promised_cash_flows(1000.0, 0.055, 5, 1)
    rows, _ = bis_expected_value(promised, 0.03, 0.60, 0.04)
    assert sum(prob for _, prob, _ in rows) == pytest.approx(1.0)
