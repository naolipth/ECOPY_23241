import pytest
from pytest import fixture, approx
from src.utils import distributions as dst
import random


class TestUniformDistribution:
    lower_bound = 1
    upper_bound = 6
    rand_gen = random

    def test_input(self):
        # Arrange

        # Act
        dist = dst.UniformDistribution(self.rand_gen, self.lower_bound, self.upper_bound)
        result1 = dist.a
        result2 = dist.b
        result3 = dist.rand

        # Assert
        assert result1 == self.lower_bound
        assert result2 == self.upper_bound
        assert result3 == self.rand_gen

    def test_pdf(self):
        # Arrange
        test_value = 1.2
        expected = 0.2

        # Act
        dist = dst.UniformDistribution(self.rand_gen, self.lower_bound, self.upper_bound)
        result = dist.pdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_cdf(self):
        # Arrange
        test_value = 1.2
        expected = 0.039999999999999994

        # Act
        dist = dst.UniformDistribution(self.rand_gen, self.lower_bound, self.upper_bound)
        result = dist.cdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_ppf(self):
        # Arrange
        test_value = 0.51
        expected = 3.55

        # Act
        dist = dst.UniformDistribution(self.rand_gen, self.lower_bound, self.upper_bound)
        result = dist.ppf(test_value)

        # Assert
        assert result == approx(expected)

    def test_gen_random(self):
        # Arrange
        lower_bound = 0
        upper_bound = 1
        expected = [0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 0.22321073814882275,
                    0.7364712141640124,
                    0.6766994874229113, 0.8921795677048454, 0.08693883262941615,
                    0.4219218196852704, 0.029797219438070344, 0.21863797480360336, 0.5053552881033624,
                    0.026535969683863625,
                    0.1988376506866485, 0.6498844377795232, 0.5449414806032167,
                    0.2204406220406967, 0.5892656838759087, 0.8094304566778266, 0.006498759678061017,
                    0.8058192518328079,
                    0.6981393949882269, 0.3402505165179919, 0.15547949981178155,
                    0.9572130722067812, 0.33659454511262676, 0.09274584338014791, 0.09671637683346401,
                    0.8474943663474598,
                    0.6037260313668911, 0.8071282732743802, 0.7297317866938179,
                    0.5362280914547007, 0.9731157639793706, 0.3785343772083535, 0.552040631273227, 0.8294046642529949,
                    0.6185197523642461, 0.8617069003107772, 0.577352145256762,
                    0.7045718362149235, 0.045824383655662215, 0.22789827565154686, 0.28938796360210717,
                    0.0797919769236275,
                    0.23279088636103018, 0.10100142940972912, 0.2779736031100921,
                    0.6356844442644002, 0.36483217897008424, 0.37018096711688264, 0.2095070307714877,
                    0.26697782204911336,
                    0.936654587712494, 0.6480353852465935, 0.6091310056669882,
                    0.171138648198097, 0.7291267979503492, 0.1634024937619284, 0.3794554417576478, 0.9895233506365952,
                    0.6399997598540929, 0.5569497437746462, 0.6846142509898746,
                    0.8428519201898096, 0.7759999115462448, 0.22904807196410437, 0.03210024390403776,
                    0.3154530480590819,
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
        dist = dst.UniformDistribution(self.rand_gen, lower_bound, upper_bound)
        dist.rand.seed(42)
        result = [dist.gen_rand() for _ in range(104)]

        # Assert
        assert result == approx(expected)

    def test_moments(self):
        # Arrange
        expected = [7 / 2, 5 / 12, 0, -6 / 5]

        # Act
        dist = dst.UniformDistribution(self.rand_gen, self.lower_bound, self.upper_bound)
        result = dist.mvsk()

        # Assert
        assert result == approx(expected)


class TestNormalDistribution:
    expected_value = 1
    variance = 4
    rand_gen = random

    def test_input(self):
        # Arrange

        # Act
        dist = dst.NormalDistribution(self.rand_gen, self.expected_value, self.variance)
        result1 = dist.loc
        result2 = dist.scale
        result3 = dist.rand

        # Assert
        assert result1 == self.expected_value
        assert result2 == self.variance
        assert result3 == self.rand_gen

    def test_pdf(self):
        # Arrange
        test_value = 0.52
        expected = 0.19380830756250708

        # Act
        dist = dst.NormalDistribution(self.rand_gen, self.expected_value, self.variance)
        result = dist.pdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_cdf(self):
        # Arrange
        test_value = 0.52
        expected = 0.4051651283022042

        # Act
        dist = dst.NormalDistribution(self.rand_gen, self.expected_value, self.variance)
        result = dist.cdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_ppf(self):
        # Arrange
        test_value = 0.52
        expected = 1.1003071669294673

        # Act
        dist = dst.NormalDistribution(self.rand_gen, self.expected_value, self.variance)
        result = dist.ppf(test_value)

        # Assert
        assert result == approx(expected)

    def test_gen_random(self):
        # Arrange
        expected_value = 2
        variance = 1
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
        dist = dst.NormalDistribution(self.rand_gen, expected_value, variance)
        dist.rand.seed(42)
        result = [dist.gen_rand() for _ in range(104)]

        # Assert
        assert result == approx(expected)

    def test_moments(self):
        # Arrange
        expected = [self.expected_value, self.variance, 0, 0]

        # Act
        dist = dst.NormalDistribution(self.rand_gen, self.expected_value, self.variance)
        result = dist.mvsk()

        # Assert
        assert result == approx(expected)


class TestCauchyDistribution:
    x0 = 2
    gamma = 4
    rand_gen = random

    def test_cauchy_distribution_input(self):
        # Arrange

        # Act
        dist = dst.CauchyDistribution(self.rand_gen, self.x0, self.gamma)
        result1 = dist.loc
        result2 = dist.scale
        result3 = dist.rand

        # Assert
        assert result1 == self.x0
        assert result2 == self.gamma
        assert result3 == self.rand_gen

    def test_pdf(self):
        # Arrange
        test_value = 1
        expected = 0.07489644380795074

        # Act
        dist = dst.CauchyDistribution(self.rand_gen, self.x0, self.gamma)
        result = dist.pdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_cdf(self):
        # Arrange
        test_value = 1
        expected = 0.4220208696226307

        # Act
        dist = dst.CauchyDistribution(self.rand_gen, self.x0, self.gamma)
        result = dist.cdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_ppf(self):
        # Arrange
        test_value = 0.7
        expected = 4.906170112021442

        # Act
        dist = dst.CauchyDistribution(self.rand_gen, self.x0, self.gamma)
        result = dist.ppf(test_value)

        # Assert
        assert result == approx(expected)

    def test_gen_random(self):
        # Arrange
        location = 2
        scale = 1
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
        dist = dst.CauchyDistribution(self.rand_gen, location, scale)
        dist.rand.seed(42)
        result = [dist.gen_rand() for _ in range(104)]

        # Assert
        assert result == approx(expected)

    def test_moments(self):
        # Arrange

        # Act
        dist = dst.CauchyDistribution(self.rand_gen, self.x0, self.gamma)

        # Assert
        with pytest.raises(Exception, match='Moments undefined'):
            dist.mvsk()


class TestLogisticDistribution:
    location = 3
    scale = 3.3
    rand_gen = random

    def test_input(self):
        # Arrange

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result1 = dist.location
        result2 = dist.scale
        result3 = dist.rand

        # Assert
        assert result1 == self.location
        assert result2 == self.scale
        assert result3 == self.rand_gen

    def test_pdf(self):
        # Arrange
        test_value = 3.7
        expected = 0.07491174022979666

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.pdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_cdf(self):
        # Arrange
        test_value = 3.7
        expected = 0.5528323503326417

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.cdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_ppf(self):
        # Arrange
        test_value = 0.7
        expected = 5.796082939277771

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.ppf(test_value)

        # Assert
        assert result == approx(expected)

    def test_gen_rand(self):
        # Arrange
        expected = [4.890494614545768, -9.08829764563687, -0.19853658521740059, -1.115274088121958, 6.39143489627801,
                    5.437528551442299, 9.973560798518974, -4.760273651519319,
                    1.9608660223975298, -8.494197066211642, -1.2029503036748164, 3.070692506244696, -8.887787063165854,
                    -1.5987973143950907, 5.041153228265085, 3.594832884090807,
                    -1.1682316143275804, 4.1910709320078645, 7.772835129759591, -13.597759102631315, 7.696131155619216,
                    5.766896388636377, 0.8148124914340094, -2.5844419025151257,
                    13.255718160967225, 0.7609259980423357, -4.5258471501419235, -4.373037700864995, 8.65977275986942,
                    4.389348485425751, 7.723808980556951, 6.277765268596625,
                    3.4790503058864743, 14.843577761919189, 1.3639467816739945, 3.6894330875902384, 8.218665794255237,
                    4.594791176165999, 9.037481492997145, 4.029313053295263,
                    5.868243137310822, -7.018903603659071, -1.026715897585591, 0.03541728217669915, -5.06908342985585,
                    -0.9356421569284756, -4.214284358021298, -0.1499672908205949,
                    4.837050203831584, 1.1703091099750509, 1.2462462102578824, -1.3820682310709103, -0.3330342090484901,
                    11.889249904455728, 5.014368369575562, 4.464081001771206,
                    -2.206010229656729, 6.267649534717904, -2.3893178927406735, 1.3768612232827007, 18.008645693725573,
                    4.89869823859244, 3.755012941404183, 5.557694874085183,
                    8.542689042329002, 7.1002696662134355, -1.0051905917330783, -8.240674156134277, 0.4433336391393925,
                    -0.3201788855274543, -1.3527371083702127, 12.254310652303905,
                    9.462961676670162, 0.4314797827684127, 5.122007629805321, 1.6017926022184226, 10.822550636112258,
                    2.4556132408394853, -0.36849480951604097, -0.6850461478011374,
                    3.8141641664267487, -0.4048322801831592, 4.127373227653882, 10.171774737137756, 1.653720025091197,
                    -1.1897758662804767, 22.813714807975046, 3.125762295285856,
                    -4.59851797880116, -6.922678151816346, -3.911289253779622, 4.720210127371877, 7.413766984675341,
                    1.9640878895415166, -5.87912456827137, 1.4071532623956535,
                    21.309686206371207, 3.3847445849787063, 14.595598372548812, 9.011877479720216, -11.703191035273809,
                    6.128548131705801, 5.513422333313912, 3.488900640107457]

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        dist.rand.seed(42)
        result = [dist.gen_rand() for _ in range(104)]

        # Assert
        assert result == approx(expected)

    def test_mean(self):
        # Arrange
        expected = 3

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.mean()

        # Assert
        assert result == approx(expected)

    def test_variance(self):
        # Arrange
        expected = 35.82666397595437

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.variance()

        # Assert
        assert result == approx(expected)

    def test_skewness(self):
        # Arrange
        expected = 0

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.skewness()

        # Assert
        assert result == approx(expected)

    def test_ex_kurtosis(self):
        # Arrange
        expected = 1.2

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.ex_kurtosis()

        # Assert
        assert result == approx(expected)

    def test_mvsk(self):
        # Arrange
        expected = [3, 35.82666397595437, 0, 1.2]

        # Act
        dist = dst.LogisticDistribution(self.rand_gen, self.location, self.scale)
        result = dist.mvsk()

        # Assert
        assert result == approx(expected)


class TestChiSquaredistribution:
    degree_of_freedom = 1.1
    rand_gen = random

    def test_input(self):
        # Arrange

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result1 = dist.dof
        result2 = dist.rand

        # Assert
        assert result1 == self.degree_of_freedom
        assert result2 == self.rand_gen

    def test_pdf(self):
        # Arrange
        test_value = 3.7
        expected = 0.03688274050827955

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.pdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_cdf(self):
        # Arrange
        test_value = 3.7
        expected = 0.9373915184962788

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.cdf(test_value)

        # Assert
        assert result == approx(expected)

    def test_ppf(self):
        # Arrange
        test_value = 0.95
        expected = 4.083997198848105

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.ppf(test_value)

        # Assert
        assert result == approx(expected)

    def test_gen_rand(self):
        # Arrange
        expected = [0.9575064383890693, 0.0019760013574234006, 0.16264812489377542, 0.10941464336492936,
                    1.3990986881976346, 1.1072359292775515, 2.792871440193376, 0.019141828343995942,
                    0.37881061282764794, 0.0027174519230520207, 0.10523326712892933, 0.5547934544992152,
                    0.002200699841339705, 0.0880682721638088, 0.997430577023333, 0.6558115003055914,
                    0.1068718276050943, 0.7853705309833245, 1.8860542477573594, 0.00017035668016866432,
                    1.85717430066405, 1.203666678805163, 0.24583310043291276, 0.055731688633074115,
                    4.352493357604958, 0.24066203475937803, 0.02154642145620562, 0.023265712539038655,
                    2.2342279651032917, 0.8319552417763427, 1.8675717267841347, 1.3623276074408814,
                    0.6324705581174398, 5.163676801053626, 0.3038680499231531, 0.6753185891765239, 2.0578755330475094,
                    0.8820723194643166, 2.3899423453732225, 0.7486632352109283,
                    1.234271626981233, 0.005949364328237066, 0.11378633870773883, 0.17935952796429155,
                    0.01636286600053116, 0.1184419150035495, 0.025189343249626076, 0.1660036938114138,
                    0.9435852167002345, 0.2822644872925927, 0.2905847177687687, 0.09712849159921537, 0.1536572485090279,
                    3.6801427358332193, 0.9902593526388412, 0.8499682684043329,
                    0.06658613068485307, 1.359080809902508, 0.061107744508571635, 0.30535454053347866,
                    6.845718337160764, 0.9596545283305253, 0.6890726913005981, 1.1418768759102567,
                    2.186825907525089, 1.6399882788710707, 0.11487190876218506, 0.003111742272939515,
                    0.21197267716610288, 0.15449774564769936, 0.09841644102579185, 3.857014534637372,
                    2.5702160695715, 0.21095943343499687, 1.019269889852583, 0.33218347504034473, 3.1768168913654327,
                    0.4511119216506573, 0.15135913698741715, 0.1321246888407862,
                    0.7016413024219676, 0.1490349469929402, 0.77077677123757, 2.8810428711540323, 0.3386334425225475,
                    0.10585240254517161, 9.49558069770342, 0.5648503167352018,
                    0.020771837478063774, 0.006258479316118481, 0.029285811145265895, 0.9135921430527165,
                    1.752648670117106, 0.37925075753760595, 0.010791638588903089, 0.3088639589695633,
                    8.657506907189884, 0.6138915927789751, 5.035229786664457, 2.3792552773013544, 0.0004794752956110301,
                    1.3148608261454935, 1.1290415823167277, 0.6344335041203677]

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        dist.rand.seed(42)
        result = [dist.gen_rand() for _ in range(104)]

        # Assert
        assert result == approx(expected)

    def test_mean(self):
        # Arrange
        expected = 1.1

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.mean()

        # Assert
        assert result == approx(expected)

    def test_variance(self):
        # Arrange
        expected = 2.2

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.variance()

        # Assert
        assert result == approx(expected)

    def test_skewness(self):
        # Arrange
        expected = 2.696799449852968

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.skewness()

        # Assert
        assert result == approx(expected)

    def test_ex_kurtosis(self):
        # Arrange
        expected = 10.909090909090908

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.ex_kurtosis()

        # Assert
        assert result == approx(expected)

    def test_mvsk(self):
        # Arrange
        expected = [1.1, 2.2, 2.696799449852968, 10.909090909090908]

        # Act
        dist = dst.ChiSquaredDistribution(self.rand_gen, self.degree_of_freedom)
        result = dist.mvsk()

        # Assert
        assert result == approx(expected)
