import pytest
from src import builtin_wrappers as bw


@pytest.fixture()
def test_list():
    return [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 8, 8, 9, 9, 9, 9, 9]


@pytest.fixture()
def test_list_2():
    return [1, 2, 3, 4, 5]


@pytest.fixture()
def test_dict():
    return {'a': 1, 'c': 2, 'b': 3}


@pytest.fixture()
def test_dict2():
    return {'d': 1, 'e': 2, 'f': 3}


@pytest.fixture()
def test_dict3():
    return {'g': 1, 'h': 2, 'i': 3}


@pytest.fixture()
def test_dol():
    return {'a': [1, 2, 3], 'b': [5, 6, 7], 'c': [8, 9, 10]}


def test_contains_value(test_list):
    assert bw.contains_value(test_list, 1) is True
    assert bw.contains_value(test_list, 0) is False


def test_number_of_elements_in_list(test_list):
    test_empty_list = []
    assert bw.number_of_elements_in_list(test_list) == 23
    assert bw.number_of_elements_in_list(test_empty_list) == 0


def test_remove_every_element_from_list(test_list):
    assert bw.remove_every_element_from_list(test_list) is None


def test_reverse_list(test_list_2):
    assert bw.reverse_list(test_list_2) == [5, 4, 3, 2, 1]


def test_odds_from_list(test_list_2):
    assert bw.odds_from_list(test_list_2) == [1, 3, 5]


def test_number_of_odds_in_list(test_list_2):
    assert bw.number_of_odds_in_list(test_list_2) == 3


def test_contains_odd(test_list_2):
    assert bw.contains_odd(test_list_2) is True
    assert bw.contains_odd([2, 4, 6, 8]) is False


def test_second_largest_in_list(test_list_2):
    assert bw.second_largest_in_list(test_list_2) == 4


def test_sum_of_elements_in_list(test_list_2):
    assert bw.sum_of_elements_in_list(test_list_2) == 15


def test_cumsum_list(test_list_2):
    assert bw.cumsum_list(test_list_2) == [1, 3, 6, 10, 15]


def test_element_wise_sum(test_list_2):
    assert bw.element_wise_sum(test_list_2, test_list_2) == [2, 4, 6, 8, 10]


def test_subset_of_list(test_list_2):
    assert bw.subset_of_list(test_list_2, 1, 4) == [2, 3, 4, 5]


def test_every_nth(test_list_2):
    assert bw.every_nth(test_list_2, 2) == [1, 3, 5]


def test_only_unique_in_list(test_list, test_list_2):
    assert bw.only_unique_in_list(test_list) is False
    assert bw.only_unique_in_list(test_list_2) is True


def test_keep_unique(test_list, test_list_2):
    assert bw.keep_unique(test_list) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert bw.keep_unique(test_list_2) == test_list_2


def test_swap(test_list_2):
    assert bw.swap(test_list_2, 1, 3) == [1, 4, 3, 2, 5]


def test_remove_element_by_value(test_list_2):
    assert bw.remove_element_by_value(test_list_2, 2) == [1, 3, 4, 5]


def test_remove_element_by_index(test_list_2):
    assert bw.remove_element_by_index(test_list_2, 3) == [1, 2, 3, 5]


def test_multiply_every_element(test_list_2):
    assert bw.multiply_every_element(test_list_2, 3) == [3, 6, 9, 12, 15]


def test_remove_key(test_dict):
    assert bw.remove_key(test_dict, 'a') == {'c': 2, 'b': 3}
    assert bw.remove_key(test_dict, 'd') == {'c': 2, 'b': 3}


def test_sort_by_key(test_dict):
    assert bw.sort_by_key(test_dict) == {'a': 1, 'b': 3, 'c': 2}


def test_sum_in_dict(test_dict):
    assert bw.sum_in_dict(test_dict) == 6


def test_merge_two_dicts(test_dict, test_dict2):
    assert bw.merge_two_dicts(test_dict, test_dict2) == {'a': 1, 'c': 2, 'b': 3, 'd': 1, 'e': 2, 'f': 3}


def test_merge_dicts(test_dict, test_dict2, test_dict3):
    assert bw.merge_dicts(test_dict, test_dict2, test_dict3) == {'a': 1, 'c': 2, 'b': 3, 'd': 1, 'e': 2, 'f': 3, 'g': 1,
                                                                 'h': 2, 'i': 3}


def test_sort_list_by_parity(test_list_2):
    assert bw.sort_list_by_parity(test_list_2) == {'odd': [1, 3, 5], 'even': [2, 4]}


def test_mean_by_key_value(test_dol):
    assert bw.mean_by_key_value(test_dol) == {'a': 2, 'b': 6, 'c': 9}


def test_count_frequency(test_list):
    assert bw.count_frequency(test_list) == {1: 2, 2: 3, 3: 2, 4: 4, 5: 2, 6: 1, 7: 1, 8: 3, 9: 5}
