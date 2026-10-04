import numpy as np
import pytest

from cdo import BASE_DEAL, moment_match, sampling_error_check, simulate
from cdo.defaults import correlate


def test_matched_numbers_have_zero_means_and_identity_covariance(raw_normals, normals):
    assert np.abs(normals.mean(axis=0)).max() < 1e-14
    assert np.allclose(np.cov(normals, rowvar=False, bias=True), np.eye(10), atol=1e-12)
    # the stored draws are close to this but not exactly there, which is why the step is needed
    assert np.abs(raw_normals.mean(axis=0)).max() > 0.01


def test_moment_match_is_the_formula_from_class(raw_normals, normals):
    demeaned = raw_normals - raw_normals.mean(axis=0)
    chol = np.linalg.cholesky(demeaned.T @ demeaned / len(raw_normals))
    assert np.allclose(normals, demeaned @ np.linalg.inv(chol).T, atol=1e-12)


def test_matching_already_matched_numbers_changes_nothing(normals):
    assert np.allclose(moment_match(normals), normals, atol=1e-12)


def test_correlated_normals_have_exactly_the_target_correlation(normals):
    X = correlate(normals, 0.2)
    corr = np.corrcoef(X, rowvar=False)
    assert np.allclose(corr[~np.eye(10, dtype=bool)], 0.2, atol=1e-12)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-12)
    assert np.abs(X.mean(axis=0)).max() < 1e-14


def test_moment_match_does_not_modify_the_stored_draws(raw_normals):
    before = raw_normals.copy()
    moment_match(raw_normals)
    assert np.array_equal(raw_normals, before)


def test_moment_matching_moves_the_default_rate_towards_theory(raw_normals, normals):
    theory = 10 * (1 - 0.96**5)
    as_drawn = simulate(raw_normals, BASE_DEAL)["defaulted"].sum(axis=1).mean()
    matched = simulate(normals, BASE_DEAL)["defaulted"].sum(axis=1).mean()
    assert abs(matched - theory) < abs(as_drawn - theory)


def test_moment_matching_lowers_the_sampling_error_of_the_mean():
    check = sampling_error_check(BASE_DEAL, 1000, n_tables=40, seed=1)
    assert check["sd_of_mean_matched"] < 0.6 * check["sd_of_mean_as_drawn"]
    # the std dev / sqrt(n) formula does not see the gain: it is an upper bound after matching
    assert check["average_formula_std_error"] > 1.5 * check["sd_of_mean_matched"]
    assert abs(check["mean_matched"] - check["exact_mean"]) < 0.15


@pytest.mark.parametrize("shape", [(10, 10), (5, 10)])
def test_too_few_cases_are_rejected(shape):
    with pytest.raises(ValueError):
        moment_match(np.random.default_rng(1).standard_normal(shape))


def test_a_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        moment_match(np.zeros(10))
