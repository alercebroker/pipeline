from wl_match_step.strategies.sorting_hat import SortingHatStrategy


class TestSortingHatStrategy:
    sorting_hat_strat = SortingHatStrategy()
    alerts = {
        (3, 4): {"ra": 1, "dec": 2, "oid": 3, "candid": 4, "mjd": 10, "mag": 12},
        (7, 8): {"ra": 5, "dec": 6, "oid": 7, "candid": 8, "mjd": 11, "mag": 13},
        (1, 2): {"ra": 7, "dec": 5, "oid": 1, "candid": 2, "mjd": 15, "mag": 7},
    }

    matches = [
        (
            3,
            4,
            100,
            {"fields": {"sorting_hat": ["mag"], "features": ["a", "b"]}, "filters": {}},
        ),
        (
            7,
            8,
            101,
            {
                "fields": {"sorting_hat": ["mjd", "mag"], "features": ["c"]},
                "filters": {},
            },
        ),
        (
            1,
            2,
            102,
            {"fields": {"features": ["a", "b"]}, "filters": {}},
        ),
    ]

    def test_get_coordinates(self):
        result = self.sorting_hat_strat.get_coordinates(self.alerts)

        assert len(result) == len(self.alerts)
        assert all([len(r) == 4 for r in result])
        assert result[0] == (1, 2, 3, 4)
        assert result[1] == (5, 6, 7, 8)

    def test_get_new_values(self):
        result_values, result_filters = self.sorting_hat_strat.get_new_values(
            self.matches, self.alerts
        )

        assert len(result_filters) == 2
        assert result_filters[0] == self.matches[0][3]
        assert result_filters[1] == self.matches[1][3]

        assert len(result_values) == 2
        assert result_values[0] == (3, 4, 100, {"mag": 12})
        assert result_values[1] == (7, 8, 101, {"mag": 13, "mjd": 11})
