from correction_step.core.strategy import ATLASStrategy

import utils


def test_atlas_strategy_corrected_is_always_false():
    alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
    corrector = ATLASStrategy(alerts)

    assert (~corrector.near_source).all()


def test_atlas_strategy_dubious_is_always_false():
    alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
    corrector = ATLASStrategy(alerts)

    assert (~corrector.dubious).all()


def test_atlas_strategy_correction_is_all_nan():
    alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
    corrector = ATLASStrategy(alerts)

    correction = corrector.corrected_frame()
    assert (correction["corrected"] == corrector.near_source).all()
    assert (correction["dubious"] == corrector.dubious).all()

    assert (correction["mag_corr"].isna()).all()
    assert (correction["e_mag_corr"].isna()).all()
    assert (correction["e_mag_corr_ext"].isna()).all()


def test_atlas_strategy_message_correction_preserves_old_fields():
    alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
    corrector = ATLASStrategy(alerts)

    correction = corrector.corrected_message()
    assert all(alert[f] == corr[f] for alert, corr in zip(alerts, correction) for f in alert)


def test_atlas_strategy_message_correction_includes_corrected_fields():
    alerts = [utils.generate_alert(mag=m, candid=f"c{m}") for m in range(-5, 15)]
    corrector = ATLASStrategy(alerts)
    check_fields = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]

    correction = corrector.corrected_message()
    assert all(all(f in corr for f in check_fields) for corr in correction)
