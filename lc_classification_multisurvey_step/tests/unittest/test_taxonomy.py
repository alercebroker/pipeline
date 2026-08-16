"""Tests for the two read-only startup queries and the startup assertions.

The session is faked rather than mocked with MagicMock so the assertions read as
"given these DB rows, ...". No database is involved.
"""
from contextlib import contextmanager

import pytest

from lc_classification_multisurvey_step.db import db


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self._rows


class FakeSession:
    """Returns canned rows and records the statements it was asked to execute."""

    def __init__(self, rows_by_call):
        self._rows_by_call = list(rows_by_call)
        self.executed = []

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return FakeResult(self._rows_by_call.pop(0))


class FakeConnection:
    def __init__(self, *rows_by_call):
        self.session_obj = FakeSession(rows_by_call)

    @contextmanager
    def session(self):
        yield self.session_obj


CLASSIFIER_ROWS = [
    {"classifier_id": 50, "classifier_name": "base", "classifier_version": "2.1.0"},
    {"classifier_id": 60, "classifier_name": "base_top", "classifier_version": "2.1.0"},
]


class TestGetClassifierIdsByName:
    def test_maps_name_to_id_and_version(self):
        conn = FakeConnection(CLASSIFIER_ROWS)

        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)

        assert got == {
            "base": {"classifier_id": 50, "classifier_version": "2.1.0"},
            "base_top": {"classifier_id": 60, "classifier_version": "2.1.0"},
        }

    def test_ids_are_not_assumed_to_be_five_through_nine(self):
        conn = FakeConnection(
            [{"classifier_id": 41, "classifier_name": "base", "classifier_version": "2.1.0"}]
        )
        assert db.get_classifier_ids_by_name(["base"], conn)["base"]["classifier_id"] == 41

    def test_rows_returned_out_of_order_still_map_correctly(self):
        conn = FakeConnection(list(reversed(CLASSIFIER_ROWS)))
        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)
        assert got["base"]["classifier_id"] == 50

    def test_names_are_passed_as_a_bound_parameter(self):
        conn = FakeConnection(CLASSIFIER_ROWS)
        db.get_classifier_ids_by_name(["base", "base_top"], conn)
        statement, params = conn.session_obj.executed[0]
        assert params == {"names": ["base", "base_top"]}
        # The names must reach the DB as a bound parameter, never interpolated.
        # An expanding bindparam defers placeholder rendering to execution time,
        # so an unexecuted statement shows SQLAlchemy's POSTCOMPILE token rather
        # than ":names" — its presence, plus the absence of the literal values,
        # is what proves the query is parameterised.
        assert "__[POSTCOMPILE_names]" in statement
        assert "base" not in statement

    def test_missing_name_is_simply_absent(self):
        conn = FakeConnection([CLASSIFIER_ROWS[0]])
        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)
        assert "base_top" not in got

    def test_duplicate_name_raises(self):
        conn = FakeConnection(
            [
                {"classifier_id": 50, "classifier_name": "base", "classifier_version": "2.1.0"},
                {"classifier_id": 51, "classifier_name": "base", "classifier_version": "2.1.0"},
            ]
        )
        with pytest.raises(ValueError, match="base"):
            db.get_classifier_ids_by_name(["base"], conn)

    def test_db_errors_propagate_rather_than_returning_an_empty_map(self):
        class Boom(FakeConnection):
            @contextmanager
            def session(self):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            db.get_classifier_ids_by_name(["base"], Boom())


TAXONOMY_ROWS = [
    {"classifier_id": 50, "class_id": 0, "class_name": "SNIa"},
    {"classifier_id": 50, "class_id": 1, "class_name": "AGN"},
    {"classifier_id": 60, "class_id": 0, "class_name": "Transient"},
]


class TestGetTaxonomyByClassifierId:
    def test_groups_class_names_by_classifier_id(self):
        conn = FakeConnection(TAXONOMY_ROWS)

        got = db.get_taxonomy_by_classifier_id([50, 60], conn)

        assert got == {50: {"SNIa": 0, "AGN": 1}, 60: {"Transient": 0}}

    def test_ids_are_passed_as_a_bound_parameter(self):
        conn = FakeConnection(TAXONOMY_ROWS)
        db.get_taxonomy_by_classifier_id([50, 60], conn)
        statement, params = conn.session_obj.executed[0]
        assert params == {"classifier_ids": [50, 60]}
        # See the note in test_names_are_passed_as_a_bound_parameter.
        assert "__[POSTCOMPILE_classifier_ids]" in statement
        assert "50" not in statement

    def test_classifier_with_no_rows_is_absent(self):
        conn = FakeConnection([TAXONOMY_ROWS[0]])
        assert db.get_taxonomy_by_classifier_id([50, 60], conn) == {50: {"SNIa": 0}}

    def test_db_errors_propagate(self):
        class Boom(FakeConnection):
            @contextmanager
            def session(self):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            db.get_taxonomy_by_classifier_id([50], Boom())


# --- resolve_classifiers ---------------------------------------------------

FIVE_NAMES = ["b", "b_top", "b_transient", "b_stochastic", "b_periodic"]


def classifier_rows(version="2.1.0", names=None, start_id=41):
    """One classifier row per name, with ids that are deliberately not 5-9."""
    names = names if names is not None else FIVE_NAMES
    return [
        {"classifier_id": start_id + i, "classifier_name": n, "classifier_version": version}
        for i, n in enumerate(names)
    ]


def taxonomy_rows(ids):
    return [
        {"classifier_id": cid, "class_id": 0, "class_name": f"class{cid}"} for cid in ids
    ]


ALL_IDS = [41, 42, 43, 44, 45]


class TestResolveClassifiers:
    def test_returns_ids_by_name_and_taxonomy_by_id(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS))

        ids, taxonomy = db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert ids == dict(zip(FIVE_NAMES, ALL_IDS))
        assert taxonomy == {cid: {f"class{cid}": 0} for cid in ALL_IDS}

    def test_taxonomy_is_queried_with_the_resolved_ids(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS))
        db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)
        _statement, params = conn.session_obj.executed[1]
        assert params == {"classifier_ids": ALL_IDS}

    def test_missing_classifier_name_raises_and_names_it(self):
        conn = FakeConnection(classifier_rows(names=FIVE_NAMES[:4]), taxonomy_rows(ALL_IDS))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "b_periodic" in str(exc.value)

    def test_empty_taxonomy_for_one_head_raises_and_names_it(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS[:4]))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "45" in str(exc.value)

    def test_version_mismatch_raises_and_reports_both_versions(self):
        conn = FakeConnection(classifier_rows(version="2.0.0"), taxonomy_rows(ALL_IDS))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "2.0.0" in str(exc.value)
        assert "2.1.0" in str(exc.value)

    def test_duplicate_name_raises(self):
        rows = classifier_rows()
        rows.append(
            {"classifier_id": 99, "classifier_name": "b", "classifier_version": "2.1.0"}
        )
        conn = FakeConnection(rows, taxonomy_rows(ALL_IDS + [99]))

        with pytest.raises(ValueError, match="more than one row"):
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)
