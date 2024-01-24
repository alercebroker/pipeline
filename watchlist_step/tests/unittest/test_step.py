import pytest
from watchlist_step.step import WatchlistStep
from unittest import mock
import datetime


@pytest.fixture
def step_creator():
    def create_step(consumer, alerts_db_connection, users_db_connection, config):
        return WatchlistStep(
            consumer=consumer,
            alerts_db_connection=alerts_db_connection,
            users_db_connection=users_db_connection,
            config=config,
        )

    return create_step


class TestExecute:
    consumer = mock.MagicMock()
    alerts_db_connection = mock.MagicMock()
    users_db_connection = mock.MagicMock()
    config = {
        "alert_db_config": {"SQL": {}},
        "users_db_config": {"SQL": {}},
    }

    @mock.patch("watchlist_step.step.datetime.datetime")
    def test_should_insert_matches_if_matches_returned(
        self, datetime_mock, step_creator
    ):
        datetime_mock.now.return_value = "date"
        self.alerts_db_connection.query().filter().all.return_value = [
            (1, 2, "oid1", "1234"),
            (3, 4, "oid2", "5678"),
        ]
        step = step_creator(
            self.consumer,
            self.alerts_db_connection,
            self.users_db_connection,
            self.config,
        )
        self.users_db_connection.execute().fetchall.return_value = [
            ("oid1", "1234", 1),
            ("oid2", "5678", 2),
        ]
        step.execute([{"candid": 1234}, {"candid": 5678}])
        self.users_db_connection.session.execute.mock_calls[0] == mock.call(
            """
        WITH positions (ra, dec, oid, candid) AS (
            VALUES (1, 2, 'oid1', '1234'),
(3, 4, 'oid2', '5678')
        )
        SELECT positions.oid, positions.candid, watchlist_target.id FROM watchlist_target, positions
        WHERE ST_DWITHIN(
            ST_SetSRID(ST_MakePoint(positions.ra, positions.dec), 4035) ,
            ST_SetSRID(ST_MakePoint(watchlist_target.ra, watchlist_target.dec), 4035),
            degrees_to_meters(watchlist_target.sr), true);
        """
        )
        step.users_db_connection.session.execute.mock_calls[1] == mock.call(
            """
        INSERT INTO watchlist_match (target, object_id, candid, date) VALUES (1, 'oid1', 1234, 'date'),
(2, 'oid2', 5678, 'date')
        """
        )

    @mock.patch.object(WatchlistStep, "insert_matches")
    def test_should_not_insert_matches_if_no_match_returned(
        self, insert_matches, step_creator
    ):
        self.alerts_db_connection.query().filter().all.return_value = [
            (1, 2, "oid1", "1234"),
            (3, 4, "oid2", "5678"),
        ]
        step = step_creator(
            self.consumer,
            self.alerts_db_connection,
            self.users_db_connection,
            self.config,
        )
        self.users_db_connection.execute().fetchall.return_value = []
        step.execute([{"candid": 1234}, {"candid": 5678}])
        insert_matches.assert_not_called()
