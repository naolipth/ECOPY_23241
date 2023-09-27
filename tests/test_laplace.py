from pytest import approx
from src.weekly import weekly_test_2 as wt
import random


def test_input():
    # Arrange
    location = 1
    scale = 2
    rand_gen = random

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    result1 = dist.loc
    result2 = dist.scale
    result3 = dist.rand

    # Assert
    assert result1 == location
    assert result2 == scale
    assert result3 == rand_gen


def test_pdf():
    # Arrange
    location = 2
    scale = 7
    rand_gen = random
    test_value = 3.5
    expected = 0.05765126764324209

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    result = dist.pdf(test_value)

    # Assert
    assert result == approx(expected)


def test_cdf():
    # Arrange
    location = 2
    scale = 7
    rand_gen = random
    test_value = 3.5
    expected = 0.5964411264973053

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    result = dist.cdf(test_value)

    # Assert
    assert result == approx(expected)


def test_ppf():
    # Arrange
    location = 2
    scale = 7
    rand_gen = random
    test_value = 0.7
    expected = 5.5757793663619335

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    result = dist.ppf(test_value)

    # Assert
    assert result == approx(expected)


def test_gen_rand():
    # Arrange
    location = 2
    scale = 7
    rand_gen = random
    expected = [4.288391747003989, -18.967115100124428, -2.184112759308719, -3.645442313605634, 6.483118445736963, 5.052180797670531, 12.738986432940045, -10.245823101852176,
                0.8114835592904552, -17.74135111698083, -3.7903357368063384, 2.0753784311434567, -18.552748574618594, -4.4548360265187625, 4.494413744638389, 2.659274515777191,
                -3.732858093179381, 3.376630680192722, 8.752136395642065, -28.400977309420618, 8.620730770714243, 5.532499322894779, -0.6944815718619446, -6.176659466511236,
                19.208628275148516, -0.7701031584914975, -9.793216492730831, -9.499777472278582, 10.311845927512355, 3.627516004772487, 8.668079311917191, 6.306352749204251,
                2.526506704291568, 22.461476035910522, 0.05189156631549996, 2.7693389544857765, 9.5271966281789, 3.8938431719683155, 10.996629283484287, 3.1764813255223032,
                5.6832767418152015, -14.728542283216019, -3.499961099146642, -1.827879131720905, -10.846295971775422, -3.3512727188544993, -9.196314003766526, -2.1095735945687952,
                4.216113864821759, -0.2061944395135069, -0.10431278427016988, -4.088955598982746, -2.3920975487086, 16.46203928949908, 4.457542179021512, 3.7236495320542575,
                -5.504938435896339, 6.290700933067067, -5.828741580882879, 0.06890354947702781, 29.05821431517482, 4.29952379930206, 2.8464742272526244, 5.2256812614280825,
                10.101936878612932, 7.620731561790759, -3.4647333696039153, -17.220211290409217, -1.2241877155438683, -2.372119264307627, -4.039818947885618, 17.189818872854588,
                11.781069661663658, -1.241410084143474, 4.606351888231268, 0.3611329643437413, 14.366539378159297, 1.3988350880456395, -2.4473140362295487, -2.9471027626060007,
                2.9166332808552893, -2.5040591840335464, 3.2973272309086203, 13.115302490101897, 0.42749611114787056, -3.7685095053494653, 39.19431976426973, 2.1346549870257996,
                -9.933211908330556, -14.533912389841166, -8.621257883098, 4.059583516232389, 8.142161556793678, 0.8154334842191047, -12.441923792300395, 0.10870766724126102,
                36.013901180408425, 2.4199496335336463, 21.950130225136665, 10.949853807098918, -24.417388844583716, 6.076798250791675, 5.161524386539684, 2.5377187563899355]

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    dist.rand.seed(42)
    result = [dist.gen_rand() for _ in range(104)]

    # Assert
    assert result == approx(expected)


def test_mvsk():
    # Arrange
    location = 2
    scale = 7
    rand_gen = random
    expected = [2, 98, 0, 3]

    # Act
    dist = wt.LaplaceDistribution(rand_gen, location, scale)
    result = dist.mvsk()

    # Assert
    assert result == approx(expected)
