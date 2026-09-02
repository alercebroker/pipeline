"""Message batch -> features frame, lastmjd map, and filtering.

Message shape follows schemas/feature_step/output.avsc: oid is a string carrying
the bigint masterid, detections is an array of records each with an mjd and a
`forced` flag, features is a nullable record (a plain dict here).
"""
import pandas as pd
import pytest

from lc_classification_multisurvey_step import input_dto


def detection(mjd, forced=False):
    return {"mjd": mjd, "forced": forced, "candid": "c", "oid": "1"}


DEFAULT = object()  # sentinel: `features=None` must be able to mean a null record


def message(oid="12345", features=DEFAULT, detections=None):
    return {
        "oid": oid,
        "features": {"feat_a": 1.0, "feat_b": 2.0} if features is DEFAULT else features,
        "detections": detections if detections is not None else [detection(60000.0)],
    }


class TestFilterMessages:
    def test_keeps_messages_with_features(self):
        msgs = [message(oid="1"), message(oid="2")]
        assert len(input_dto.filter_messages(msgs)) == 2

    def test_drops_messages_with_null_features(self):
        msgs = [message(oid="1", features=None), message(oid="2")]
        kept = input_dto.filter_messages(msgs)
        assert [m["oid"] for m in kept] == ["2"]

    def test_drops_messages_with_empty_features(self):
        kept = input_dto.filter_messages([message(oid="1", features={})])
        assert kept == []

    def test_min_detections_unset_keeps_everything(self):
        msgs = [message(detections=[detection(1.0)])]
        assert len(input_dto.filter_messages(msgs, min_detections=None)) == 1

    def test_min_detections_counts_non_forced_only(self):
        msgs = [
            message(
                oid="1",
                detections=[detection(1.0), detection(2.0, forced=True), detection(3.0, forced=True)],
            )
        ]
        assert input_dto.filter_messages(msgs, min_detections=2) == []
        assert len(input_dto.filter_messages(msgs, min_detections=1)) == 1

    def test_missing_forced_key_defaults_to_non_forced(self):
        # A detection dict with no "forced" key at all must still count toward
        # min_detections, the same as an explicit forced=False.
        msgs = [message(oid="1", detections=[{"mjd": 1.0, "candid": "c", "oid": "1"}])]
        assert len(input_dto.filter_messages(msgs, min_detections=1)) == 1


class TestFilterMessagesBadOid:
    # `oid` is a plain Avro "string" — the schema cannot constrain it to
    # digits, so its validity rests on a producer convention, not the wire
    # format. Records with these oids deserialize fine and then crash int().
    # One such message must not take down the whole batch (design §8): drop
    # it here, before it ever reaches collapse_by_oid.
    @pytest.mark.parametrize("bad_oid", ["ZTF21abcdefg", "", "1e5"])
    def test_drops_messages_with_unparseable_oid(self, bad_oid):
        kept = input_dto.filter_messages([message(oid=bad_oid)])
        assert kept == []

    def test_bad_oid_does_not_take_down_the_rest_of_the_batch(self):
        msgs = [message(oid="ZTF21abcdefg"), message(oid="2")]

        kept = input_dto.filter_messages(msgs)

        assert [m["oid"] for m in kept] == ["2"]
        frame = input_dto.build_features_frame(input_dto.collapse_by_oid(kept))
        lastmjd = input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(kept))
        assert list(frame.index) == [2]
        assert 2 in lastmjd

    def test_bad_oids_are_logged_once_per_batch_naming_the_raw_value(self, caplog):
        with caplog.at_level("WARNING"):
            kept = input_dto.filter_messages([message(oid="ZTF21abcdefg"), message(oid="")])

        assert kept == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "ZTF21abcdefg" in warnings[0].message
        assert "''" in warnings[0].message


class TestBuildFeaturesFrame:
    def test_one_row_per_message_indexed_by_int_oid(self):
        msgs = [
            message(oid="12345", features={"a": 1.0, "b": 2.0}),
            message(oid="67890", features={"a": 3.0, "b": 4.0}),
        ]

        frame = input_dto.build_features_frame(input_dto.collapse_by_oid(msgs))

        assert list(frame.index) == [12345, 67890]
        assert frame.index.name == "oid"
        assert list(frame.columns) == ["a", "b"]
        assert frame.loc[67890, "a"] == 3.0

    def test_oid_is_cast_to_int_not_left_as_string(self):
        frame = input_dto.build_features_frame(input_dto.collapse_by_oid([message(oid="12345")]))
        assert frame.index[0] == 12345
        assert not isinstance(frame.index[0], str)

    def test_empty_batch_gives_an_empty_frame(self):
        frame = input_dto.build_features_frame(input_dto.collapse_by_oid([]))
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 0
        assert frame.index.name == "oid"

    def test_duplicate_oids_collapse_keeping_the_last_message(self):
        # Two messages for the same object can land in one consume batch. Left
        # alone they would produce two probability rows colliding on
        # (oid, sid, classifier_id, class_id), and the scribe's highest-lastmjd
        # dedup could not break the tie. The stamp step collapses the same way.
        msgs = [
            message(oid="1", features={"a": 1.0}),
            message(oid="1", features={"a": 2.0}),
            message(oid="2", features={"a": 3.0}),
        ]

        frame = input_dto.build_features_frame(input_dto.collapse_by_oid(msgs))

        assert list(frame.index) == [1, 2]
        assert frame.loc[1, "a"] == 2.0  # last message wins


class TestLastmjdByOid:
    def test_max_mjd_over_detections(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60010.5)])]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs)) == {1: 60010.5}

    def test_forced_photometry_counts_toward_lastmjd(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60020.0, forced=True)])]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs)) == {1: 60020.0}

    def test_no_jd_offset_is_subtracted(self):
        msgs = [message(oid="1", detections=[detection(60000.0)])]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs))[1] == pytest.approx(60000.0)

    def test_message_without_detections_is_absent(self):
        msgs = [message(oid="1", detections=[]), message(oid="2")]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs)) == {2: 60000.0}

    def test_none_mjd_is_skipped(self):
        msgs = [
            message(
                oid="1",
                detections=[detection(60000.0), {"mjd": None, "forced": False, "candid": "c", "oid": "1"}],
            )
        ]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs)) == {1: 60000.0}

    @pytest.mark.parametrize("bad_mjd", [float("nan"), float("inf")])
    def test_nan_mjd_does_not_leak_through_regardless_of_order(self, bad_mjd):
        # max() is order-sensitive with NaN: max(nan, 60000.0) is nan, but
        # max(60000.0, nan) is 60000.0. Neither NaN nor inf may win either way.
        # inf is the more dangerous of the two: a NaN lastmjd gets dropped
        # downstream by probabilities.py's isna() check, but an inf lastmjd is
        # accepted by Postgres double precision and would win the scribe's
        # highest-lastmjd dedup forever.
        msgs = [message(oid="1", detections=[detection(bad_mjd), detection(60000.0)])]
        assert input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs)) == {1: 60000.0}

    def test_oids_with_no_usable_mjd_are_logged_once_per_batch(self, caplog):
        # A batch-wide flood of per-message warnings would bury the signal in
        # exactly the incident the log exists to diagnose (see probabilities.py's
        # equivalent aggregation for the same reasoning).
        msgs = [message(oid="1", detections=[]), message(oid="2", detections=[])]

        with caplog.at_level("WARNING"):
            result = input_dto.lastmjd_by_oid(input_dto.collapse_by_oid(msgs))

        assert result == {}
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "1" in warnings[0].message and "2" in warnings[0].message


class TestDuplicateOidConsistency:
    def test_features_and_lastmjd_derive_from_the_same_winning_message(self):
        # The winning message for oid 7 (last by arrival order) has an empty
        # detections list. Both derive from ONE collapse — the way step.execute
        # calls them — so the emitted row cannot pair the winner's features with
        # the loser's stale lastmjd.
        msgs = [
            message(oid="7", features={"a": 1.0}, detections=[detection(60000.0)]),
            message(oid="7", features={"a": 2.0}, detections=[]),
        ]

        collapsed = input_dto.collapse_by_oid(msgs)
        frame = input_dto.build_features_frame(collapsed)
        lastmjd = input_dto.lastmjd_by_oid(collapsed)

        assert frame.loc[7, "a"] == 2.0  # winner's features
        assert 7 not in lastmjd  # winner has no detections -> absent, not msg1's stale value


class TestCreateInputDto:
    def test_features_only_dto(self):
        pytest.importorskip("alerce_classifiers.base.factories")

        dto = input_dto.create_input_dto(input_dto.collapse_by_oid([message(oid="1", features={"a": 1.0})]))

        assert list(dto.features.index) == [1]
        assert len(dto.detections) == 0
        assert len(dto.non_detections) == 0
        assert len(dto.xmatch) == 0
        assert len(dto.stamps) == 0
