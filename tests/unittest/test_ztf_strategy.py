import numpy as np

from correction_step.core.strategy import ZTFStrategy

import utils


def test_ztf_strategy_corrected_is_based_on_distance():
    dists = np.linspace(1, 2, 10)
    alerts = [utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=d), candid=f"c{d}") for d in dists]
    corrector = ZTFStrategy(alerts)

    assert (dists >= ZTFStrategy.DISTANCE_THRESHOLD).any()  # Fix test
    assert (corrector.corrected == (dists < ZTFStrategy.DISTANCE_THRESHOLD)).all()


def test_ztf_strategy_first_detection_with_close_source_splits_by_aid_and_fid():
    mjds = np.linspace(2, 3, 3)
    alerts = [utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="fn")]
    alerts.extend([
        utils.generate_alert(
            extra_fields=utils.ztf_extra_fields(distnr=np.random.random() * 2),
            candid=f"fn{mjd}",
            mjd=mjd,
        ) for mjd in mjds])
    alerts.append(utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="fy", fid=2))
    alerts.extend([
        utils.generate_alert(
            extra_fields=utils.ztf_extra_fields(distnr=np.random.random() * 2),
            candid=f"fy{mjd}",
            mjd=mjd,
            fid=2
        ) for mjd in mjds])
    alerts.append(utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="sy", aid="AID2"))
    alerts.extend([
        utils.generate_alert(
            extra_fields=utils.ztf_extra_fields(distnr=np.random.random() * 2),
            candid=f"sy{mjd}",
            mjd=mjd,
            aid="AID2"
        ) for mjd in mjds])
    alerts.append(utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="sn", aid="AID2", fid=2))
    alerts.extend([
        utils.generate_alert(
            extra_fields=utils.ztf_extra_fields(distnr=np.random.random() * 2),
            candid=f"sn{mjd}",
            mjd=mjd,
            fid=2,
            aid="AID2"
        ) for mjd in mjds])
    corrector = ZTFStrategy(alerts)

    assert corrector.first[corrector.first.index.str.startswith("fy")].all()
    assert ~corrector.first[corrector.first.index.str.startswith("fn")].all()
    assert corrector.first[corrector.first.index.str.startswith("sy")].all()
    assert ~corrector.first[corrector.first.index.str.startswith("sn")].all()


def test_ztf_strategy_dubious_for_negative_difference_without_close_source():
    alerts = [utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), isdiffpos=-1)]
    corrector = ZTFStrategy(alerts)
    assert corrector.dubious.all()

    alerts = [utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), isdiffpos=1)]
    corrector = ZTFStrategy(alerts)
    assert ~corrector.dubious.all()


def test_ztf_strategy_dubious_for_first_with_close_source_and_follow_up_without():
    alerts = [
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="first"),
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="second", mjd=2),
    ]
    corrector = ZTFStrategy(alerts)
    assert ~corrector.dubious.loc["first"]
    assert corrector.dubious.loc["second"]

    alerts = [
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="first"),
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="second", mjd=2),
    ]
    corrector = ZTFStrategy(alerts)
    assert ~corrector.dubious.all()


def test_ztf_strategy_dubious_for_follow_up_with_close_source_and_first_without():
    alerts = [
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="first"),
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=1), candid="second", mjd=2),
    ]
    corrector = ZTFStrategy(alerts)
    assert ~corrector.dubious.loc["first"]
    assert corrector.dubious.loc["second"]

    alerts = [
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="first"),
        utils.generate_alert(extra_fields=utils.ztf_extra_fields(distnr=2), candid="second", mjd=2),
    ]
    corrector = ZTFStrategy(alerts)
    assert ~corrector.dubious.all()

#
#
# def test_atlas_strategy_correction_is_all_nan():
#     alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
#     corrector = ATLASStrategy(alerts)
#
#     correction = corrector.corrected_frame()
#     assert (correction["corrected"] == corrector.corrected).all()
#     assert (correction["dubious"] == corrector.dubious).all()
#
#     assert (correction["mag_corr"].isna()).all()
#     assert (correction["e_mag_corr"].isna()).all()
#     assert (correction["e_mag_corr_ext"].isna()).all()
#
#
# def test_atlas_strategy_message_correction_preserves_old_fields():
#     alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
#     corrector = ATLASStrategy(alerts)
#
#     correction = corrector.corrected_message()
#     assert all(alert[f] == corr[f] for alert, corr in zip(alerts, correction) for f in alert)
#
#
# def test_atlas_strategy_message_correction_includes_corrected_fields():
#     alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
#     corrector = ATLASStrategy(alerts)
#     check_fields = ["mag_corr", "e_mag_corr", "e_mag_corr_ext", "candid"]
#
#     correction = corrector.corrected_message()
#     assert all(all(f in corr for f in check_fields) for corr in correction)
