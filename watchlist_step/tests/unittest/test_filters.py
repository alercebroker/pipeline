import pytest

import watchlist_step.filters as filters


def test_constant_filter():
    assert filters.constant({"mag": 1}, "mag", 10, "less")
    assert not filters.constant({"mag": 10}, "mag", 10, "less")
    assert filters.constant({"mag": 15}, "mag", 10, "greater")
    assert not filters.constant({"mag": 5}, "mag", 10, "greater")
    assert filters.constant({"mag": 1}, "mag", 1, "eq")
    assert not filters.constant({"mag": 15}, "mag", 1, "eq")

    with pytest.raises(Exception):
        filters.constant({"mag": 1}, "mag", 10, "l")


def test_all_filter():
    assert filters._all(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 25, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 1, "op": "eq"},
            },
        ],
    )
    assert not filters._all(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 25, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 2, "op": "eq"},
            },
        ],
    )
    assert not filters._all(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 100, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 2, "op": "eq"},
            },
        ],
    )


def test_any_filter():
    assert filters._any(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 25, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 1, "op": "eq"},
            },
        ],
    )
    assert filters._any(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 25, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 2, "op": "eq"},
            },
        ],
    )
    assert not filters._any(
        {"mag": 50, "fid": 1},
        [
            {
                "type": "constant",
                "params": {"field": "mag", "constant": 100, "op": "greater eq"},
            },
            {
                "type": "constant",
                "params": {"field": "fid", "constant": 2, "op": "eq"},
            },
        ],
    )
