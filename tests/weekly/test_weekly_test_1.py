import pytest
from src import weekly_test_1 as wt


@pytest.fixture()
def long_list():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


@pytest.fixture()
def test_list_1():
    return [1, 6, 8, 12, 24]


@pytest.fixture()
def test_list_2():
    return [24, 4, 3, 2, 1]


@pytest.fixture()
def even_list():
    return [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


@pytest.fixture()
def odd_list():
    return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


@pytest.fixture()
def test_dict():
    return {'a': [1, 2, 3], 'd': [1, 2, 3], 'b': [5, 6, 7], 'c': [8, 9, 10]}


def test_evens_from_list(long_list, even_list):
    # Arrange
    input_list = long_list
    expected = even_list

    # Act
    answer = wt.evens_from_list(input_list)

    # Assert
    assert answer == expected


def test_every_element_is_odd(odd_list, even_list):
    # Arrange
    odd_input_list = odd_list
    even_input_list = even_list
    expected_odd = True
    expected_even = False

    # Act
    answer_odd = wt.every_element_is_odd(odd_input_list)
    answer_even = wt.every_element_is_odd(even_input_list)

    # Assert
    assert expected_odd == answer_odd
    assert expected_even == answer_even


def test_kth_largest_in_list(long_list):
    # Arrange
    input_list = long_list
    kth_element = 10
    expected = 11

    # Act
    result = wt.kth_largest_in_list(input_list, kth_element)

    # Assert
    assert result == expected


def test_cumavg_list(even_list):
    # Arrange
    input_list = even_list
    expected = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Act
    result = wt.cumavg_list(input_list)

    # Assert
    assert result == expected


def test_element_wise_multiplication(test_list_1, test_list_2):
    # Arrange
    input_list1 = test_list_1
    input_list2 = test_list_2
    expected = [24, 24, 24, 24, 24]

    # Act
    result = wt.element_wise_multiplication(input_list1, input_list2)

    # Assert
    assert result == expected


def test_merge_lists(long_list, test_list_1, test_list_2):
    # Arrange
    list1 = long_list
    list2 = test_list_1
    list3 = test_list_2
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 6, 8, 12, 24, 24, 4, 3, 2, 1]

    # Act
    result = wt.merge_lists(list1, list2, list3)

    # Assert
    assert result == expected


def test_squared_odds(test_list_2):
    # Arrange
    input_list = test_list_2
    expected = [9, 1]

    # Act
    result = wt.squared_odds(input_list)

    # Assert
    assert result == expected


def test_reverse_sort_by_key(test_dict):
    # Arrange
    input_dict = test_dict
    expected = {'d': [1, 2, 3], 'c': [8, 9, 10], 'b': [5, 6, 7], 'a': [1, 2, 3], }

    # Act
    result = wt.reverse_sort_by_key(input_dict)

    # Assert
    assert result == expected


def test_sort_list_by_divisibility(long_list):
    # Arrange
    input_list = long_list
    expected = {'by_two': [2, 4, 6, 8, 12, 14, 16, 18],
                'by_five': [5, 15],
                'by_two_and_five': [10, 20],
                'by_none': [1, 3, 7, 9, 11, 13, 17, 19]}

    # Act
    result = wt.sort_list_by_divisibility(input_list)

    # Assert
    assert result == expected
