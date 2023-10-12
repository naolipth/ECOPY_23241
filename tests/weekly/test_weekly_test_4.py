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
        expected = [[2.773361957358876, 1.899507148484203, 1.726126807387266, 1.6164327161740228, 2.0520765469609543, 2.2255800354219417, 3.232594306802298, 2.965422128044485,
                     2.8281385386986586, 2.648395921072912, 2.5239796086329167, 2.482119063685331, 2.370206775881246, 2.290062467900484, 2.327804956937326, 2.319662119996955,
                     2.258668814246351, 2.2684463173929448, 2.425234838720525, 2.3543001602754288],
                    [5.14984111163784, 4.2313142411774285, 3.3261184408911464, 2.79061480036866, 6.9068166940578, 6.006909544677778, 5.306240605426915, 4.781344528743022, 4.978654560617323,
                     4.7331397708003, 4.77419916534804, 4.68468491490676, 4.490188565049411, 6.826355035470878, 6.4785379973161294, 6.213150954938197, 6.192484720927524, 5.994089330972058,
                     6.05919088054139, 5.874533131824213]]
        # Act
        result = wt.gen_pareto_mean_trajectories(dist, 2, 20)

        # Assert
        assert result == expected
