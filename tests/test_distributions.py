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


def test_uniform_distribution_gen_random():
    # Arrange
    lower_bound = 0
    upper_bound = 1
    rand_gen = random
    expected = [0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 0.22321073814882275, 0.7364712141640124,
                0.6766994874229113, 0.8921795677048454, 0.08693883262941615,
                0.4219218196852704, 0.029797219438070344, 0.21863797480360336, 0.5053552881033624, 0.026535969683863625,
                0.1988376506866485, 0.6498844377795232, 0.5449414806032167,
                0.2204406220406967, 0.5892656838759087, 0.8094304566778266, 0.006498759678061017, 0.8058192518328079,
                0.6981393949882269, 0.3402505165179919, 0.15547949981178155,
                0.9572130722067812, 0.33659454511262676, 0.09274584338014791, 0.09671637683346401, 0.8474943663474598,
                0.6037260313668911, 0.8071282732743802, 0.7297317866938179,
                0.5362280914547007, 0.9731157639793706, 0.3785343772083535, 0.552040631273227, 0.8294046642529949,
                0.6185197523642461, 0.8617069003107772, 0.577352145256762,
                0.7045718362149235, 0.045824383655662215, 0.22789827565154686, 0.28938796360210717, 0.0797919769236275,
                0.23279088636103018, 0.10100142940972912, 0.2779736031100921,
                0.6356844442644002, 0.36483217897008424, 0.37018096711688264, 0.2095070307714877, 0.26697782204911336,
                0.936654587712494, 0.6480353852465935, 0.6091310056669882,
                0.171138648198097, 0.7291267979503492, 0.1634024937619284, 0.3794554417576478, 0.9895233506365952,
                0.6399997598540929, 0.5569497437746462, 0.6846142509898746,
                0.8428519201898096, 0.7759999115462448, 0.22904807196410437, 0.03210024390403776, 0.3154530480590819,
                0.26774087597570273, 0.21098284358632646, 0.9429097143350544,
                0.8763676264726689, 0.3146778807984779, 0.65543866529488, 0.39563190106066426, 0.9145475897405435,
                0.4588518525873988, 0.26488016649805246, 0.24662750769398345,
                0.5613681341631508, 0.26274160852293527, 0.5845859902235405, 0.897822883602477, 0.39940050514039727,
                0.21932075915728333, 0.9975376064951103, 0.5095262936764645,
                0.09090941217379389, 0.04711637542473457, 0.10964913035065915, 0.62744604170309, 0.7920793643629641,
                0.42215996679968404, 0.06352770615195713, 0.38161928650653676,
                0.9961213802400968, 0.529114345099137, 0.9710783776136181, 0.8607797022344981, 0.011481021942819636,
                0.7207218193601946, 0.6817103690265748, 0.5369703304087952]

    # Act
    dist = dst.UniformDistribution(rand_gen, lower_bound, upper_bound)
    dist.rand.seed(42)
    result = [dist.gen_rand() for _ in range(104)]

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


def test_normal_distribution_gen_random():
    # Arrange
    expected_value = 2
    variance = 1
    rand_gen = random
    expected = [2.3569270708684202, 0.040220005035863426, 1.402327737513401, 1.2386055079365632, 2.6325040365145673,
                2.4584891926826047, 3.2382028242997416, 0.640150850153381,
                1.8030205757199995, 0.11621777518302956, 1.2231985982110944, 2.01342411975449, 0.06566489228213013,
                1.1542196715540782, 2.3850084912801304, 2.1128909153430118,
                1.2292940494954125, 2.2256564500200304, 2.8757994177707253, -0.48383725525960486, 2.862592607409397,
                2.5190566888467996, 1.5882204790694017, 0.9867881879196907,
                3.7192225537107695, 1.5782247741276278, 0.6759658001856772, 0.6995090430631992, 3.025746123557081,
                2.263003427582639, 2.867362442337795, 2.6120020116542166,
                2.0909355314968727, 3.9286971357294815, 1.6906676885242782, 2.1308186901524557, 2.9518152797052206,
                2.301595425973794, 3.088020178981573, 2.1951242106524,
                2.5375955145460214, 0.3132361404850703, 1.2542137565594773, 1.4448264039009358, 0.5935277994004471,
                1.2703133982786134, 0.7241339067558314, 1.4111280956859877,
                2.3469470329438633, 1.6544279550689527, 1.668625910909558, 1.1918670680910513, 1.3780209504710874,
                3.5272823172999788, 2.3800218045840524, 2.277054851322021,
                1.0503247637097153, 2.6101742217458517, 1.019430310336593, 1.6930886944354917, 4.308824917643308,
                2.358458151351882, 2.143240161622287, 2.4806412426035793,
                3.0062482660344427, 2.7587532488261512, 1.2580145380087953, 0.1492149857044469, 1.5195480979271214,
                1.3803401462847624, 1.1969843467488075, 3.579678240030068,
                3.157018634645539, 1.5173661806042777, 2.400045954961469, 1.7353301454403307, 3.3693022322679735,
                1.8966733247997656, 1.3716280883261456, 1.3148591032893364,
                2.1544388392859126, 1.3650840241901614, 2.213639723576354, 3.2692434941851385, 1.7451008726448132,
                1.225510742268578, 4.81191268452039, 2.023881146827212,
                0.6648242275156562, 0.3265195439893258, 0.7716037654062655, 2.3250966601676577, 2.8136573541287966,
                1.8036291796594455, 0.4741828657634508, 1.6987692893071888,
                4.662457368477558, 2.073043741459248, 3.896884036534805, 3.083829065345797, -0.27406557889967464,
                2.5849871438128904, 2.4724869393799453, 2.0928039169305044]

    # Act
    dist = dst.NormalDistribution(rand_gen, expected_value, variance)
    dist.rand.seed(42)
    result = [dist.gen_rand() for _ in range(104)]

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


def test_cauchy_distribution_gen_random():
    # Arrange
    location = 2
    scale = 1
    rand_gen = random
    expected = [2.4683666304212855, -10.700718224987096, 1.1460785957253765, 0.8157349202809552, 2.9184145398363524,
                2.62016902621987, 4.838439819858863, -1.5698103474532727,
                1.749669351301691, -8.651314745526031, 0.780621622122829, 2.016825721308426, -9.967610601341123,
                0.6129952701433399, 2.509068231123391, 2.1421335126859256,
                0.7946088597756067, 2.288026887957848, 3.465806808562584, -46.973292367804085, 3.430669080157282,
                2.7176494270022458, 1.4512697870808333, 0.11818896232165921,
                9.39455862964753, 1.4362304057610396, -1.3343889326827694, -1.189258388522132, 3.9249986624104256,
                2.337911174976789, 3.4432730617111393, 2.8801221615481616,
                2.1143078991906554, 13.811853273696295, 1.5987360567685207, 2.1649628620060746, 3.683712589337559,
                2.3905585394249558, 4.1550287976283045, 2.2479081872828695,
                2.7487199569797767, -4.898245729704455, 0.8505107253265061, 1.2212362268769552, -1.9053366610563596,
                0.8855741187479358, -1.0450531661390938, 1.1619481889446734,
                2.4541084437248024, 1.5478478261007056, 1.567936180306964, 0.7066768293803403, 1.1013644282776154,
                6.958476241384336, 2.501775348242616, 2.3569412923715047,
                0.3228143011864921, 2.8767549068511697, 0.2261871588497888, 1.6020916875774052, 32.371822731777314,
                2.4705633597156145, 2.180846650237332, 2.6551438297930976,
                3.8582383223196883, 3.1783247923420808, 0.8588611620851236, -7.882482376203695, 1.3451583015247892,
                1.1056881796193423, 0.7189946983049806, 7.515639109401866,
                4.4438597666178445, 1.3416731947583487, 2.5312384829905623, 1.659839852135325, 5.63507752801854,
                1.8700043588660877, 1.0893816026958345, 0.9785822838408418,
                2.195218411491572, 1.0770161514105634, 2.2721715487958587, 5.0075337518156875, 1.6729968251671261,
                0.7859421546383472, 131.26590976131035, 2.029936672549596,
                -1.4056745232757368, -4.706410486807678, -0.7872429239831917, 2.4232454012717044, 3.306729842785594,
                1.750464247648456, -2.9438645289833874, 1.6099448064786004,
                84.06376288846677, 2.0917213332770728, 12.975645154257803, 4.138690590310203, -25.71285000922879,
                2.8310987341204648, 2.642182488281769, 2.1166708138920254]

    # Act
    dist = dst.CauchyDistribution(rand_gen, location, scale)
    dist.rand.seed(42)
    result = [dist.gen_rand() for _ in range(104)]

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
