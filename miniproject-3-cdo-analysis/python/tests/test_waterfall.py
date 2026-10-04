import numpy as np
import pytest

from cdo.bonds import promised_cash_flows
from cdo.waterfall import waterfall

A_DUE = promised_cash_flows(20.0, 0.02, 20, 4)
B_DUE = promised_cash_flows(10.0, 0.04, 20, 4)


def test_amounts_due_to_the_classes():
    assert A_DUE[0] == pytest.approx(0.10) and A_DUE[-1] == pytest.approx(20.10)
    assert B_DUE[0] == pytest.approx(0.10) and B_DUE[-1] == pytest.approx(10.10)
    assert A_DUE.sum() == pytest.approx(22.0) and B_DUE.sum() == pytest.approx(12.0)


def test_everyone_is_paid_when_the_pool_pays_as_promised():
    pool = 10 * promised_cash_flows(10.0, 0.06, 20, 4)
    paid = waterfall(pool[None, :], A_DUE, B_DUE)
    assert np.allclose(paid["A"], A_DUE)
    assert np.allclose(paid["B"], B_DUE)
    assert paid["Equity"].sum() == pytest.approx(96.0)


def test_class_a_is_paid_before_class_b():
    pool = np.array([[0.15, 0.05, 0.0]])
    paid = waterfall(pool, np.full(3, 0.10), np.full(3, 0.10))
    assert np.allclose(paid["A"], [[0.10, 0.05, 0.0]])
    assert np.allclose(paid["B"], [[0.05, 0.0, 0.0]])
    assert np.allclose(paid["Equity"], 0.0)


def test_shortfalls_are_not_carried_forward():
    # Period 1 leaves nothing for Class B; the surplus in period 2 goes to equity, not back to B.
    pool = np.array([[0.10, 1.00]])
    paid = waterfall(pool, np.full(2, 0.10), np.full(2, 0.10))
    assert np.allclose(paid["B"], [[0.0, 0.10]])
    assert np.allclose(paid["Equity"], [[0.0, 0.80]])


def test_payments_always_add_up_to_the_pool_and_are_never_negative():
    rng = np.random.default_rng(5)
    pool = rng.uniform(0.0, 40.0, size=(500, 20))
    paid = waterfall(pool, A_DUE, B_DUE)
    assert np.allclose(paid["A"] + paid["B"] + paid["Equity"], pool)
    for name in ("A", "B", "Equity"):
        assert (paid[name] >= 0).all()
    assert (paid["A"] <= A_DUE + 1e-12).all()
    assert (paid["B"] <= B_DUE + 1e-12).all()
