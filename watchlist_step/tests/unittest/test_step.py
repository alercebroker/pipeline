from unittest import mock

import pytest

from watchlist_step.step import WatchlistStep


@pytest.fixture
def wl_step():
    strategy_name = "SortingHat"

    config = {
        "CONSUMER_CONFIG": {
            "CLASS": "unittest.mock.MagicMock",
        },
        "PSQL_CONFIG": {
            "ENGINE": "postgresql",
            "HOST": "localhost",
            "USER": "postgres",
            "PASSWORD": "password",
            "PORT": 5433,
            "DB_NAME": "postgres",
        },
    }
    return WatchlistStep(
        strategy_name=strategy_name,
        config=config,
    )


@mock.patch("watchlist_step.step.datetime.datetime")
@mock.patch("psycopg2.connect")
def test_should_insert_matches(
    connect_mock: mock.MagicMock,
    datetime_mock: mock.MagicMock,
    wl_step: WatchlistStep,
):
    datetime_mock.now.return_value = "date"

    cursor_mock: mock.MagicMock = (
        connect_mock.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    )
    executemany_mock: mock.MagicMock = cursor_mock.executemany

    wl_step.insert_matches([(0, 1, 2), (3, 4, 5)])

    executemany_mock.assert_called_once()

    assert (
        len(executemany_mock.call_args.args[1]) == 2
    ), "executemany should be called with two matches"

    assert (
        len(executemany_mock.call_args.args[1][0]) == 5
    ), "executemany should be called with tuples of length 5"


@mock.patch("psycopg2.connect")
def test_should_create_matches(connect_mock: mock.MagicMock, wl_step: WatchlistStep):
    cursor_mock: mock.MagicMock = (
        connect_mock.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    )
    execute_mock: mock.MagicMock = cursor_mock.execute

    wl_step.match_user_targets(
        [("ra1", "dec1", "oid1", "candid1"), ("ra2", "dec2", "oid2", "candid2")]
    )

    execute_mock.assert_called_once()
    assert (
        len(execute_mock.call_args.args[1]) == 8
    ), "execute should be called with a flattened list"
