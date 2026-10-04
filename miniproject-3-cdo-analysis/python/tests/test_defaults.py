import math

import numpy as np
import pytest

from cdo.defaults import correlate, correlation_matrix, default_period, default_times


def test_correlation_matrix_has_unit_diagonal_and_constant_off_diagonal():
    corr = correlation_matrix(4, 0.2)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr[~np.eye(4, dtype=bool)], 0.2)


def test_correlate_with_zero_correlation_returns_the_inputs():
    Z = np.random.default_rng(1).standard_normal((50, 10))
    assert np.allclose(correlate(Z, 0.0), Z)


def test_correlate_leaves_the_first_bond_unchanged():
    Z = np.random.default_rng(2).standard_normal((50, 10))
    assert np.allclose(correlate(Z, 0.2)[:, 0], Z[:, 0])


def test_correlate_reproduces_the_target_correlation_in_a_large_sample():
    Z = np.random.default_rng(3).standard_normal((200_000, 10))
    X = correlate(Z, 0.2)
    off_diagonal = np.corrcoef(X.T)[~np.eye(10, dtype=bool)]
    assert np.allclose(off_diagonal, 0.2, atol=0.01)
    assert np.allclose(X.std(axis=0), 1.0, atol=0.01)


def test_default_time_formula_for_a_single_draw():
    # z = 0 gives u = 0.5, so t = ln(0.5) / ln(0.96).
    T = default_times(np.array([[0.0]]), 0.04, 0.0)
    assert T[0, 0] == pytest.approx(math.log(0.5) / math.log(0.96))


def test_a_higher_normal_means_a_later_default():
    T = default_times(np.array([[-2.0], [0.0], [2.0]]), 0.04, 0.0)[:, 0]
    assert T[0] < T[1] < T[2]
    # by hand: u = N(-2) = 0.02275, so t = ln(1 - 0.02275) / ln(0.96) = 0.564 years
    assert T[0] == pytest.approx(0.5638, abs=1e-4)


def test_each_bond_keeps_its_own_default_probability_under_correlation():
    Z = np.random.default_rng(4).standard_normal((200_000, 10))
    T = default_times(Z, 0.04, 0.2)
    assert np.allclose((T <= 1.0).mean(axis=0), 0.04, atol=0.003)
    assert np.allclose((T <= 5.0).mean(axis=0), 1 - 0.96**5, atol=0.005)


def test_zero_default_probability_means_no_default_ever():
    T = default_times(np.zeros((3, 10)), 0.0, 0.2)
    assert np.isinf(T).all()
    assert (default_period(T, 4, 20) == 21).all()


@pytest.mark.parametrize(
    "years, expected",
    [
        (0.0, 1),
        (0.10, 1),
        (0.25, 1),
        (0.2501, 2),
        (1.0, 4),
        (4.76, 20),
        (5.0, 20),
        (5.0001, 21),
        (60.0, 21),
        (math.inf, 21),
    ],
)
def test_default_period_maps_years_to_quarters(years, expected):
    assert default_period(np.array([years]), 4, 20)[0] == expected


def test_default_period_handles_annual_periods():
    assert default_period(np.array([0.5, 1.5, 7.0]), 1, 5).tolist() == [1, 2, 6]
