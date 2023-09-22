from pytest import fixture, approx
from random import seed
from src.utils import rand_wrapper as rw


@fixture
def test_list():
    return [1, 2, 3, 3, 4, 5, 6, 7, 8, 9]


@fixture
def test_string():
    return 'A kis piros labdán számos, kis fekete pötty van'


def test_random_from_list(test_list):
    # Arrange
    input_list = test_list
    seed(42)
    expected = 2

    # Act
    result = rw.random_from_list(input_list)

    # Assert
    assert result == expected


def test_random_sublist_from_list(test_list):
    # Arrange
    input_list = test_list
    number_of_elements = 3
    seed(42)
    expected = [6, 1, 3]

    # Act
    result = rw.random_sublist_from_list(input_list, number_of_elements)

    # Assert
    assert result == expected


def test_random_from_string(test_string):
    # Arrange
    input_string = test_string
    seed(42)
    expected = 't'

    # Act
    result = rw.random_from_string(input_string)

    # Assert
    assert result == expected


def test_hundred_small_random():
    # Arrange
    seed(42)
    expected = [0.6394267984578837,
                0.025010755222666936,
                0.27502931836911926,
                0.22321073814882275,
                0.7364712141640124,
                0.6766994874229113,
                0.8921795677048454,
                0.08693883262941615,
                0.4219218196852704,
                0.029797219438070344,
                0.21863797480360336,
                0.5053552881033624,
                0.026535969683863625,
                0.1988376506866485,
                0.6498844377795232,
                0.5449414806032167,
                0.2204406220406967,
                0.5892656838759087,
                0.8094304566778266,
                0.006498759678061017,
                0.8058192518328079,
                0.6981393949882269,
                0.3402505165179919,
                0.15547949981178155,
                0.9572130722067812,
                0.33659454511262676,
                0.09274584338014791,
                0.09671637683346401,
                0.8474943663474598,
                0.6037260313668911,
                0.8071282732743802,
                0.7297317866938179,
                0.5362280914547007,
                0.9731157639793706,
                0.3785343772083535,
                0.552040631273227,
                0.8294046642529949,
                0.6185197523642461,
                0.8617069003107772,
                0.577352145256762,
                0.7045718362149235,
                0.045824383655662215,
                0.22789827565154686,
                0.28938796360210717,
                0.0797919769236275,
                0.23279088636103018,
                0.10100142940972912,
                0.2779736031100921,
                0.6356844442644002,
                0.36483217897008424,
                0.37018096711688264,
                0.2095070307714877,
                0.26697782204911336,
                0.936654587712494,
                0.6480353852465935,
                0.6091310056669882,
                0.171138648198097,
                0.7291267979503492,
                0.1634024937619284,
                0.3794554417576478,
                0.9895233506365952,
                0.6399997598540929,
                0.5569497437746462,
                0.6846142509898746,
                0.8428519201898096,
                0.7759999115462448,
                0.22904807196410437,
                0.03210024390403776,
                0.3154530480590819,
                0.26774087597570273,
                0.21098284358632646,
                0.9429097143350544,
                0.8763676264726689,
                0.3146778807984779,
                0.65543866529488,
                0.39563190106066426,
                0.9145475897405435,
                0.4588518525873988,
                0.26488016649805246,
                0.24662750769398345,
                0.5613681341631508,
                0.26274160852293527,
                0.5845859902235405,
                0.897822883602477,
                0.39940050514039727,
                0.21932075915728333,
                0.9975376064951103,
                0.5095262936764645,
                0.09090941217379389,
                0.04711637542473457,
                0.10964913035065915,
                0.62744604170309,
                0.7920793643629641,
                0.42215996679968404,
                0.06352770615195713,
                0.38161928650653676,
                0.9961213802400968,
                0.529114345099137,
                0.9710783776136181,
                0.8607797022344981]

    # Act
    result = rw.hundred_small_random()

    # Assert
    assert result == expected


def test_hundred_large_random():
    # Arrange
    seed(42)
    expected = [664,
                124,
                35,
                769,
                291,
                260,
                238,
                152,
                764,
                114,
                702,
                768,
                923,
                568,
                99,
                614,
                442,
                42,
                40,
                105,
                233,
                248,
                527,
                626,
                37,
                584,
                213,
                743,
                675,
                728,
                568,
                439,
                235,
                469,
                613,
                294,
                838,
                900,
                16,
                787,
                835,
                173,
                724,
                442,
                358,
                294,
                169,
                230,
                990,
                791,
                354,
                114,
                104,
                399,
                109,
                377,
                877,
                362,
                628,
                280,
                836,
                54,
                757,
                480,
                559,
                137,
                954,
                397,
                90,
                575,
                310,
                859,
                653,
                643,
                916,
                892,
                380,
                601,
                206,
                731,
                81,
                56,
                687,
                243,
                801,
                306,
                91,
                885,
                248,
                897,
                113,
                399,
                294,
                474,
                660,
                864,
                383,
                176,
                389,
                373]

    # Act
    result = rw.hundred_large_random()

    # Assert
    assert result == expected


def test_five_random_number_div_three():
    # Arrange
    seed(42)
    expected = [990, 180, 45, 429, 384]

    # Act
    result = rw.five_random_number_div_three()

    # Assert
    assert result == expected


def test_random_reorder(test_list):
    # Arrange
    input_list = test_list
    seed(42)
    expected = [2, 1, 4, 9, 6, 5, 8, 3, 3, 7]

    # Act
    result = rw.random_reorder(input_list)

    # Assert
    assert result == expected


def test_uniform_one_to_five():
    # Arrange
    seed(42)
    expected = 4.197133992289419

    # Act
    result = rw.uniform_one_to_five()

    # Assert
    assert result == approx(expected)

