import pytest
from pytest import fixture, approx
from src.utils import distributions as dst
import random


def test_uniform_distribution_input():
    # Arrange
    lower_bound = 1
    upper_bound = 6
    rand_gen = random

    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    result1 = dist.a
    result2 = dist.b
    result3 = dist.rand

    # Assert
    assert result1 == lower_bound
    assert result2 == upper_bound
    assert result3 == rand_gen


def test_uniform_distribution_pdf():
    # Arrange
    lower_bound = 1
    upper_bound = 6
    rand_gen = random
    test_value = 1.2
    expected = 0.2

    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    result = dist.pdf(test_value)

    # Assert
    assert result == approx(expected)


def test_uniform_distribution_cdf():
    # Arrange
    lower_bound = 1
    upper_bound = 6
    rand_gen = random
    test_value = 1.2
    expected = 0.039999999999999994

    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    result = dist.cdf(test_value)

    # Assert
    assert result == approx(expected)


def test_uniform_distribution_ppf():
    # Arrange
    lower_bound = 1
    upper_bound = 6
    rand_gen = random
    test_value = 0.51
    expected = 3.55


    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    result = dist.ppf(test_value)

    # Assert
    assert result == approx(expected)


def test_uniform_distribution_moments():
    # Arrange
    lower_bound = 1
    upper_bound = 6
    rand_gen = random
    expected = [7/2, 5/12, 0, -6/5]

    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    result = dist.mvsk()

    # Assert
    assert result == approx(expected)


def test_normal_distribution_input():
    # Arrange
    expected_value = 1
    variance = 4
    rand_gen = random

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    result1 = dist.loc
    result2 = dist.scale
    result3 = dist.rand

    # Assert
    assert result1 == expected_value
    assert result2 == variance
    assert result3 == rand_gen


def test_normal_distribution_pdf():
    # Arrange
    expected_value = 1
    variance = 4
    rand_gen = random
    test_value = 0.52
    expected = 0.19380830756250708

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    result = dist.pdf(test_value)

    # Assert
    assert result == approx(expected)

def test_normal_distribution_cdf():
    # Arrange
    expected_value = 1
    variance = 4
    rand_gen = random
    test_value = 0.52
    expected = 0.4051651283022042

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    result = dist.cdf(test_value)

    # Assert
    assert result == approx(expected)

def test_normal_distribution_ppf():
    # Arrange
    expected_value = 1
    variance = 4
    rand_gen = random
    test_value = 0.52
    expected = 1.1003071669294673

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    result = dist.ppf(test_value)

    # Assert
    assert result == approx(expected)


def test_normal_distribution_moments():
    # Arrange
    expected_value = 1
    variance = 4
    rand_gen = random
    expected = [expected_value, variance, 0, 0]

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    result = dist.mvsk()

    # Assert
    assert result == approx(expected)



def test_cauchy_distribution_input():
    # Arrange
    x0 = 1
    gamma = 4
    rand_gen = random

    # Act
    dist = dst.CauchyDistribution(rand_gen, x0, gamma)
    result1 = dist.loc
    result2 = dist.scale
    result3 = dist.rand

    # Assert
    assert result1 == x0
    assert result2 == gamma
    assert result3 == rand_gen


def test_cauchy_distribution_pdf():
    # Arrange
    x0 = 2
    gamma = 4
    rand_gen = random
    test_value = 1
    expected = 0.07489644380795074

    # Act
    dist = dst.CauchyDistribution(rand_gen, x0, gamma)
    result = dist.pdf(test_value)

    # Assert
    assert result == approx(expected)

def test_cauchy_distribution_cdf():
    # Arrange
    x0 = 2
    gamma = 4
    rand_gen = random
    test_value = 1
    expected = 0.4220208696226307

    # Act
    dist = dst.CauchyDistribution(rand_gen, x0, gamma)
    result = dist.cdf(test_value)

    # Assert
    assert result == approx(expected)

def test_cauchy_distribution_ppf():
    # Arrange
    x0 = 2
    gamma = 4
    rand_gen = random
    test_value = 0.7
    expected = 4.906170112021442

    # Act
    dist = dst.CauchyDistribution(rand_gen, x0, gamma)
    result = dist.ppf(test_value)

    # Assert
    assert result == approx(expected)


def test_cauchy_distribution_moments():
    # Arrange
    x0 = 2
    gamma = 4
    rand_gen = random
    random.seed(42)

    # Act
    dist = dst.CauchyDistribution(rand_gen, x0, gamma)

    # Assert
    with pytest.raises(Exception, match='Moments undefined'):
        dist.mvsk()
