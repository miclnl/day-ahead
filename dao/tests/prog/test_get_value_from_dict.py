"""utils.get_value_from_dict: looking up the tariff in effect on a date.

A date before the earliest known entry used to wrap around to the last
(possibly future) entry, because bisect_left(...) - 1 is -1 in that case and
Python indexes -1 as "the last element". The dict was also assumed to
already be sorted, which insertion order from JSON does not guarantee.
"""

import pytest

from dao.prog.utils import get_value_from_dict


TARIFFS = {"2023-01-01": 0.10, "2024-01-01": 0.12, "2025-01-01": 0.15}


def test_an_exact_match_is_returned_directly():
    assert get_value_from_dict("2024-01-01", TARIFFS) == 0.12


def test_a_date_between_two_entries_uses_the_earlier_one():
    assert get_value_from_dict("2024-06-15", TARIFFS) == 0.12


def test_a_date_after_the_last_entry_uses_the_last_one():
    assert get_value_from_dict("2030-01-01", TARIFFS) == 0.15


def test_a_date_before_the_first_entry_uses_the_first_one_not_the_last():
    assert get_value_from_dict("2020-01-01", TARIFFS) == 0.10


def test_works_regardless_of_insertion_order():
    unsorted = {"2025-01-01": 0.15, "2023-01-01": 0.10, "2024-01-01": 0.12}
    assert get_value_from_dict("2024-06-15", unsorted) == 0.12
    assert get_value_from_dict("2020-01-01", unsorted) == 0.10


def test_a_single_entry_dict_always_returns_that_entry():
    only = {"2024-01-01": 0.12}
    assert get_value_from_dict("2020-01-01", only) == 0.12
    assert get_value_from_dict("2030-01-01", only) == 0.12


def test_an_empty_dict_raises_a_clear_error():
    with pytest.raises(ValueError, match="geen datum"):
        get_value_from_dict("2024-01-01", {})
