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


class TestBuildFeaturesFrame:
    def test_one_row_per_message_indexed_by_int_oid(self):
        msgs = [
            message(oid="12345", features={"a": 1.0, "b": 2.0}),
            message(oid="67890", features={"a": 3.0, "b": 4.0}),
        ]

        frame = input_dto.build_features_frame(msgs)

        assert list(frame.index) == [12345, 67890]
        assert frame.index.name == "oid"
        assert list(frame.columns) == ["a", "b"]
        assert frame.loc[67890, "a"] == 3.0

    def test_oid_is_cast_to_int_not_left_as_string(self):
        frame = input_dto.build_features_frame([message(oid="12345")])
        assert frame.index[0] == 12345
        assert not isinstance(frame.index[0], str)

    def test_empty_batch_gives_an_empty_frame(self):
        frame = input_dto.build_features_frame([])
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

        frame = input_dto.build_features_frame(msgs)

        assert list(frame.index) == [1, 2]
        assert frame.loc[1, "a"] == 2.0  # last message wins


class TestLastmjdByOid:
    def test_max_mjd_over_detections(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60010.5)])]
        assert input_dto.lastmjd_by_oid(msgs) == {1: 60010.5}

    def test_forced_photometry_counts_toward_lastmjd(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60020.0, forced=True)])]
        assert input_dto.lastmjd_by_oid(msgs) == {1: 60020.0}

    def test_no_jd_offset_is_subtracted(self):
        msgs = [message(oid="1", detections=[detection(60000.0)])]
        assert input_dto.lastmjd_by_oid(msgs)[1] == pytest.approx(60000.0)

    def test_message_without_detections_is_absent(self):
        msgs = [message(oid="1", detections=[]), message(oid="2")]
        assert input_dto.lastmjd_by_oid(msgs) == {2: 60000.0}


class TestCreateInputDto:
    def test_features_only_dto(self):
        pytest.importorskip("alerce_classifiers.base.factories")

        dto = input_dto.create_input_dto([message(oid="1", features={"a": 1.0})])

        assert list(dto.features.index) == [1]
        assert len(dto.detections) == 0
        assert len(dto.non_detections) == 0
        assert len(dto.xmatch) == 0
        assert len(dto.stamps) == 0
