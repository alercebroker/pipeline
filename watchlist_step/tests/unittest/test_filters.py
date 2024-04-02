import pytest

import watchlist_step.filters as filters


def test_constant_filter():
    assert filters.constant({"mag": 1}, "mag", 10, "less")
    assert not filters.constant({"mag": 10}, "mag", 10, "less")
    assert filters.constant({"mag": 15}, "mag", 10, "greater")
    assert not filters.constant({"mag": 5}, "mag", 10, "greater")

    with pytest.raises(Exception):
        filters.constant({"mag": 1}, "mag", 10, "l")
