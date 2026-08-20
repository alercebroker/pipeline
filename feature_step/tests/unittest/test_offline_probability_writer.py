"""Unit tests for probability_writer — pure row building + fake-engine write."""
import pandas as pd
import pytest

from alerce_classifiers.base.dto import OutputDTO
from features.offline import probability_writer as pw
from features.offline.classifier_taxonomy_lut import TAXONOMY_LUT

OID = 36028941624528297

# Test oracle: {classifier_id: {class_name: class_id}} derived from the fixture
# (the runtime source is the DB; here we simulate what fetch_taxonomy_maps returns).
TAX_MAPS = {
    cid: {name: idx for idx, name in enumerate(names)}
    for cid, names in TAXONOMY_LUT.items()
}


def _frame(classifier_id, probs=None):
    """Build a 1-oid frame whose columns are the model's class labels for a classifier."""
    names = TAXONOMY_LUT[classifier_id]
    if probs is None:
        probs = [1.0 / (i + 1) for i in range(len(names))]  # strictly decreasing
    return pd.DataFrame([dict(zip(names, probs))], index=[OID])


def _full_dto():
    return OutputDTO(
        _frame(5),
        {"top": _frame(6),
         "children": {"Transient": _frame(7),
                      "Stochastic": _frame(8),
                      "Periodic": _frame(9)}},
    )


def test_version_to_smallint():
    assert pw.classifier_version_to_smallint("2.1.0") == 210
    assert pw.classifier_version_to_smallint("1.0.4") == 104
    assert pw.classifier_version_to_smallint("2.1.0_rc1") == 210  # patch suffix stripped


def test_classifier_version_refuses_an_unparseable_version():
    """Returning 0 silently mislabels every row of the run.

    <schema>.probability has no FK or CHECK on classifier_version, so a 0 is
    accepted and stored. It happened for real: the CLI passed the version the
    model pickle self-reports ("no_version"), which has no 3 parts, so 45 rows
    were written as version 0 while the batch runner wrote 210 for the same
    model. Refusing is what keeps the two paths from disagreeing.
    """
    import pytest
    with pytest.raises(ValueError, match="no_version"):
        pw.classifier_version_to_smallint("no_version")


def test_classifier_ids_constant():
    assert pw.CLASSIFIER_IDS == [5, 6, 7, 8, 9]


def test_build_rows_fans_out_to_45_rows():
    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    assert len(rows) == 45  # 21 + 3 + 6 + 6 + 9
    by_cls = {}
    for r in rows:
        by_cls[r["classifier_id"]] = by_cls.get(r["classifier_id"], 0) + 1
    assert by_cls == {5: 21, 6: 3, 7: 6, 8: 6, 9: 9}


def test_build_rows_field_values():
    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    r = next(r for r in rows if r["classifier_id"] == 7 and r["class_id"] == 0)
    # transient class_id 0 == "SESN" (fixture/DB), highest prob (1/1) -> ranking 1
    assert r["oid"] == OID and isinstance(r["oid"], int)
    assert r["sid"] == 0
    assert r["classifier_version"] == 210
    assert r["probability"] == pytest.approx(1.0)
    assert r["ranking"] == 1
    assert r["lastmjd"] == 60000.5
    # ranking is dense-desc within the classifier: class_id 5 (TDE, prob 1/6) -> rank 6
    last = next(r for r in rows if r["classifier_id"] == 7 and r["class_id"] == 5)
    assert last["ranking"] == 6


def test_build_rows_uses_class_ids_from_the_map_not_position():
    # A taxonomy map whose class_ids are NOT the enumerate position — the builder
    # must use the map's ids verbatim (proves it reads the DB map, not list index).
    maps = {6: {"Periodic": 42, "Stochastic": 43, "Transient": 44}}
    dto = OutputDTO(_frame(5), {"top": _frame(6), "children": {}})
    # only classifier 6 present in the map; probabilities frame (id 5) must also map,
    # so include id 5 too:
    maps[5] = TAX_MAPS[5]
    rows = pw.build_probability_rows(dto, OID, 1.0, maps)
    ids_for_6 = sorted(r["class_id"] for r in rows if r["classifier_id"] == 6)
    assert ids_for_6 == [42, 43, 44]


def test_build_rows_unknown_class_raises():
    bad = OutputDTO(
        pd.DataFrame([{"AGN": 0.5, "SNIbc": 0.5}], index=[OID]),  # SNIbc not in taxonomy
        {"top": pd.DataFrame(), "children": {}},
    )
    with pytest.raises(ValueError, match="SNIbc"):
        pw.build_probability_rows(bad, OID, 1.0, {5: {"AGN": 0}})


def test_build_rows_missing_classifier_in_map_raises():
    # classifier 5 present in output but absent from the taxonomy map -> hard error
    dto = OutputDTO(_frame(5), {"top": pd.DataFrame(), "children": {}})
    with pytest.raises(ValueError, match="classifier_id=5"):
        pw.build_probability_rows(dto, OID, 1.0, {6: {}})


def test_build_rows_empty_output_returns_empty():
    empty = OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})
    assert pw.build_probability_rows(empty, OID, 1.0, TAX_MAPS) == []


def test_build_rows_rejects_multi_oid_frame():
    multi = pd.DataFrame(
        [{n: 0.5 for n in TAXONOMY_LUT[5]}, {n: 0.5 for n in TAXONOMY_LUT[5]}],
        index=[OID, OID + 1],
    )
    dto = OutputDTO(multi, {"top": pd.DataFrame(), "children": {}})
    with pytest.raises(ValueError, match="single-oid"):
        pw.build_probability_rows(dto, OID, 1.0, TAX_MAPS)


def test_build_rows_requires_lastmjd():
    with pytest.raises(ValueError, match="lastmjd"):
        pw.build_probability_rows(_full_dto(), OID, None, TAX_MAPS)


class _RecordingConn:
    def __init__(self):
        self.calls = []

    def execute(self, sql, records):
        self.calls.append((sql, records))


class _FakeEngine:
    def __init__(self):
        self.conn = _RecordingConn()

    def begin(self):
        conn = self.conn

        class _Ctx:
            def __enter__(self_):
                return conn

            def __exit__(self_, *a):
                return False

        return _Ctx()


def test_write_dry_run_does_not_connect(monkeypatch):
    def _boom(_creds):
        raise AssertionError("dry-run must not open a connection")
    monkeypatch.setattr(pw.db, "_make_engine", _boom)

    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    result = pw.write_probabilities(rows, "ignored", execute=False)
    assert result == {"executed": False, "would_write": 45}


def test_write_execute_upserts(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)

    rows = pw.build_probability_rows(_full_dto(), OID, 60000.5, TAX_MAPS)
    result = pw.write_probabilities(rows, "creds", schema="multisurvey_ztf", execute=True)

    assert result == {"executed": True, "written": 45}
    assert len(fake.conn.calls) == 1
    sql, records = fake.conn.calls[0]
    sql_str = str(sql)
    assert "multisurvey_ztf.probability" in sql_str
    assert "ON CONFLICT (oid, sid, classifier_id, class_id)" in sql_str
    assert "DO UPDATE SET" in sql_str
    assert "updated_date" not in sql_str  # probability has no updated_date column
    assert records[0]["oid"] == OID and isinstance(records[0]["oid"], int)


def test_write_default_schema_is_db_schema(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)
    rows = pw.build_probability_rows(_full_dto(), OID, 1.0, TAX_MAPS)
    pw.write_probabilities(rows, "creds", execute=True)  # no schema=
    assert f"{pw.db.SCHEMA}.probability" in str(fake.conn.calls[0][0])


def test_write_empty_rows_execute_no_call(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(pw.db, "_make_engine", lambda _c: fake)
    result = pw.write_probabilities([], "creds", execute=True)
    assert result == {"executed": True, "written": 0}
    assert fake.conn.calls == []
