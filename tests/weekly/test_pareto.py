from pytest import approx
from src.weekly import weekly_test_2 as wt
import random


def test_input():
    # Arrange
    scale = 1
    shape = 2
    rand_gen = random

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    result1 = dist.scale
    result2 = dist.shape
    result3 = dist.rand

    # Assert
    assert result1 == scale
    assert result2 == shape
    assert result3 == rand_gen

def test_pdf():
    # Arrange
    scale = 2
    shape = 5
    rand_gen = random
    test_value = 7.1
    expected = 0.0012490214315112114

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    result = dist.pdf(test_value)

    # Assert
    assert result == approx(expected)


def test_cdf():
    # Arrange
    scale = 2
    shape = 5
    rand_gen = random
    test_value = 3.1
    expected = 0.8882258157079126

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    result = dist.cdf(test_value)

    # Assert
    assert result == approx(expected)


def test_ppf():
    # Arrange
    scale = 2
    shape = 5
    rand_gen = random
    test_value = 0.5
    expected = 2.29739670999407

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    result = dist.ppf(test_value)

    # Assert
    assert result == approx(expected)


def test_gen_rand():
    # Arrange
    scale = 5
    shape = 1
    rand_gen = random
    expected = [13.86680978679438, 5.128261698047648, 6.896830625966959, 6.4367522126714665, 18.973259350543398, 15.465487388634394, 46.37339967542217, 5.476084383698963,
                8.64934911966024, 5.15356181220596, 6.399082421164813, 10.108265346309441, 5.136296611161103, 6.240932320752874, 14.280999017265536, 10.987597829456972,
                6.413879611183414, 12.173319354425203, 26.237141113084856, 5.032706349093007, 25.7492055581892, 16.563936853585087, 7.578634201592905, 5.920519394006006,
                116.85812134407179, 7.536868988888331, 5.511134849608682, 5.535359959778838, 32.78567407805867, 12.617533312235455, 25.92396555412719, 18.50014080026343,
                10.781161833806115, 185.9825957547498, 8.04549731574824, 11.161726596346028, 29.309124883783795, 13.106838508645707, 36.15509386394679, 11.830179530989318,
                16.924588150090834, 5.240125522339513, 6.475830635165732, 7.036188164423873, 5.433554016715007, 6.517128004755271, 5.561744104573007, 6.92495457442726,
                13.724365927511272, 7.871935312926542, 7.938788348633324, 6.325166945987879, 6.821076019796779, 78.93231442407361, 14.205973528057932, 12.792009784588105,
                6.032371987341731, 18.45882118336499, 5.9765896535880225, 8.057439121152138, 477.25182227298234, 13.888879624006677, 11.2854014409074, 15.853601552045637,
                31.8171243710976, 22.321419757083042, 6.485488677274828, 5.165824217342065, 7.304100888658313, 6.828183952862695, 6.336997819827572, 87.58057420388926,
                40.442481668401044, 7.295839226414531, 14.511204526993907, 8.273103773635613, 58.51210030025666, 9.239614002018794, 6.80161216189898, 6.636823153305446,
                11.399080617320605, 6.781882794148629, 12.036185305090157, 48.93463601524392, 8.325015326842406, 6.404679077418108, 2030.5446672398818, 10.194226388767527,
                5.50000194365214, 5.247230481296895, 5.615763594378493, 13.420874718005837, 24.04763714135825, 8.652913804375826, 5.339186255532006, 8.085633802763892,
                1289.1183744510072, 10.618289064364607, 172.88103458381087, 35.9143033038317, 5.058071833711196, 17.90329623512075, 15.708962886125128, 10.798444091961432]

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    dist.rand.seed(42)
    result = [dist.gen_rand() for _ in range(104)]

    # Assert
    assert result == approx(expected)


def test_mvsk():
    # Arrange
    scale = 2
    shape = 5
    rand_gen = random
    expected = [2.5, 5 / 12, 6 * (3 / 5)**(1/2), 354 / 5]

    # Act
    dist = wt.ParetoDistribution(rand_gen, scale, shape)
    result = dist.mvsk()

    # Assert
    assert result == approx(expected)
