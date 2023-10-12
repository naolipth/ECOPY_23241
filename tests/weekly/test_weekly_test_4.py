from pytest import approx
import pandas as pd
import random
from pandas.testing import assert_frame_equal, assert_series_equal
import src.weekly.weekly_test_4 as wt
from src.weekly.weekly_test_2 import ParetoDistribution

class TestWeekly4:
    refdata = pd.read_csv('../../data/Euro_2012_stats_TEAM.csv')

    def test_number_of_participants(self):
        # Arrange
        expected = 16

        # Act
        result = wt.number_of_participants(self.refdata)

        # Assert
        assert result == expected

    def test_goals(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/goals.csv')

        # Act
        result = wt.goals(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_sorted_by_goal(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/sorted_by_goal.csv')
        data = pd.read_csv('../../data/weekly4/goals.csv')

        # Act
        result = wt.sorted_by_goal(data)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_avg_goal(self):
        # Arrange
        expected = 4.75

        # Act
        result = wt.avg_goal(self.refdata)

        # Assert
        assert result == approx(expected)

    def test_countries_over_five(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/countries_over_five.csv')

        # Act
        result = wt.countries_over_five(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_countries_starting_with_g(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/countries_starting_with_g.csv')
        expected = expected.squeeze()

        # Act
        result = wt.countries_starting_with_g(self.refdata)

        # Assert
        assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_index=False)

    def test_first_seven_columns(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/first_seven_columns.csv')

        # Act
        result = wt.first_seven_columns(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_every_column_except_last_three(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/every_column_except_last_three.csv')

        # Act
        result = wt.every_column_except_last_three(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_sliced_view(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/sliced_view.csv')
        columns_to_keep = ['Team', 'Shooting Accuracy']
        column_to_filter = 'Team'
        rows_to_keep = ['England', 'Italy', 'Russia']

        # Act
        result = wt.sliced_view(self.refdata, columns_to_keep, column_to_filter, rows_to_keep)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_generate_quartile(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/generate_quartile.csv')

        # Act
        result = wt.generate_quarters(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_average_yellow_in_quartiles(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/average_yellow_in_quartiles.csv')
        data = pd.read_csv('../../data/weekly4/generate_quartile.csv')
        expected = expected.squeeze()

        # Act
        result = wt.average_yellow_in_quartiles(data)

        # Assert
        assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_index=False)

    def test_minmax_block_in_quartile(self):
        # Arrange
        expected = pd.read_csv('../../data/weekly4/minmax_block_in_quartile.csv')
        data = pd.read_csv('../../data/weekly4/generate_quartile.csv')

        # Act
        result = wt.minmax_block_in_quartile(data)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_gen_pareto_mean_trajectories(self):
        # Arrange
        dist = ParetoDistribution(random, 1, 1)
        expected = [[2.773361957358876, 1.0256523396095296, 1.3793661251933917, 1.2873504425342932, 3.7946518701086793, 3.0930974777268787, 9.274679935084434, 1.0952168767397927, 1.729869823932048, 1.030712362441192,
                     1.2798164842329627, 2.0216530692618884, 1.0272593222322206, 1.2481864641505749, 2.8561998034531073, 2.1975195658913944, 1.2827759222366828, 2.4346638708850405, 5.247428222616971, 1.0065412698186014],
                    [5.14984111163784, 3.3127873707170172, 1.515726840318581, 1.1841038788012013, 23.371624268814358, 1.5073737977776662, 1.1022269699217364, 1.1070719919557677, 6.557134815611734, 2.523506662447091,
                     5.184793110825438, 3.7000281600526863, 2.156232366761223, 37.19651915094996, 1.6090994631496478, 2.2323453192692058, 5.861824976756759, 2.6213677017291412, 7.2310187727893585, 2.3660359061978635]]

        # Act
        result = wt.gen_pareto_mean_trajectories(dist, 2, 20)

        # Assert
        assert result == expected
