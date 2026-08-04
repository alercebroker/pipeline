from unittest import mock

from metadata_step.step import MetadataStep


def _make_step():
    # No real consumer/DB needed: _format_detection never touches self.db, and
    # an empty catalog skips the gaia/ps1 catalog-update branches.
    config = {"CONSUMER_CONFIG": {"CLASS": "unittest.mock.MagicMock"}}
    return MetadataStep(config, db_sql=mock.MagicMock())


def _message(band):
    return {
        "oid": "ZTFband",
        "candid": 1,
        "fid": band,
        "extra_fields": {
            # jdstartref/jdendref are read directly by format_reference
            "jdstartref": 2455500.5,
            "jdendref": 2455600.5,
        },
    }


def test_format_detection_maps_all_ztf_bands():
    # ZTF has three bands; the char fid is mapped to its numeric id. An i-band
    # alert (fid='i') used to raise KeyError and poison-pill the consumer.
    step = _make_step()
    empty_catalogs = {"ps1": {}, "gaia": {}}
    for band, expected in (("g", 1), ("r", 2), ("i", 3)):
        out = step._format_detection(_message(band), empty_catalogs)
        assert out["reference"]["fid"] == expected
        assert out["dataquality"]["fid"] == expected
