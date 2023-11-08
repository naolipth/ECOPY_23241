from pathlib import Path

import pandas as pd
from pytest import approx

from src.linear_regression.LinearRegressions import (
    LinearRegressionSM,
    LinearRegressionNP,
)


class TestLinearRegressionSM:
    ref_data = pd.read_parquet(
        Path.cwd()
        .parent.parent.joinpath("data")
        .joinpath("weekly6")
        .joinpath("toclean.parquet")
    )
    model = LinearRegressionSM(ref_data["ex_ret_1"], ref_data[["Mkt-RF", "SMB", "HML"]])
    model.fit()

    def test_get_params(self):
        # Arrange
        expected = [
            0.015819773799455268,
            -0.4718314998337236,
            0.41112646168635775,
            -0.05456049807706942,
        ]

        # Act
        result = self.model.get_params().tolist()

        # Assert
        assert result == approx(expected)

    def test_pvalues(self):
        # Arrange
        expected = [
            0.10417609528945243,
            0.026105629855327488,
            0.17304104205667561,
            0.8460584552608155,
        ]

        # Act
        result = self.model.get_pvalues().tolist()

        # Assert
        assert result == approx(expected)

    def test_get_wald_test_result(self):
        # Arrange
        restriction_matrix = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        expected = "F-value: 1.98, p-value: 0.119"

        # Act
        result = self.model.get_wald_test_result(restriction_matrix)

        # Assert
        assert result == approx(expected)

    def test_get_model_goodness_values(self):
        # Arrange
        expected = (
            "Adjusted R-squared: 0.015, Akaike IC: -2.28e+02, Bayes IC: -2.15e+02"
        )

        # Act
        result = self.model.get_model_goodness_values()

        # Assert
        assert result == approx(expected)


class TestLinearRegressionNP:
    ref_data = pd.read_parquet(
        Path.cwd()
        .parent.parent.joinpath("data")
        .joinpath("weekly6")
        .joinpath("toclean.parquet")
    )
    model = LinearRegressionNP(ref_data["ex_ret_1"], ref_data[["Mkt-RF", "SMB", "HML"]])
    model.fit()

    def test_get_params(self):
        # Arrange
        expected = [
            0.015819773799455268,
            -0.4718314998337236,
            0.41112646168635775,
            -0.05456049807706942,
        ]

        # Act
        result = self.model.get_params().tolist()

        # Assert
        assert result == approx(expected)

    def test_pvalues(self):
        # Arrange
        expected = [
            0.10417609528945243,
            0.026105629855327488,
            0.17304104205667561,
            0.8460584552608155,
        ]

        # Act
        result = self.model.get_pvalues().tolist()

        # Assert
        assert result == approx(expected)

    def test_get_wald_test_result(self):
        # Arrange
        restriction_matrix = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        expected = "Wald: 1.979, p-value: 0.119"

        # Act
        result = self.model.get_wald_test_result(restriction_matrix)

        # Assert
        assert result == approx(expected)

    def test_get_model_goodness_values(self):
        # Arrange
        expected = "Centered R-squared: 0.030, Adjusted R-squared: 0.015"

        # Act
        result = self.model.get_model_goodness_values()

        # Assert
        assert result == approx(expected)
