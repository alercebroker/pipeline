def get_fid(feature: str):
    """
    Gets the band number (fid) of a feature.
    Most features include the fid in the name as a sufix after '_' (underscore) character.
    Some features don't include the fid in their name but are known to be asociated with a specific band or multiband.
    This method considers all of these cases and the possible return values are:

    - 0: for wise features and sgscore
    - 12: for multiband features or power_rate
    - 1: for fid = 1
    - 2 for fid = 2
    - -99 if there is no known band for the feature

    Parameters
    ----------
    feature : str
        name of the feature
    """
    if not isinstance(feature, str):
        raise Exception(
            f"Feature {feature} is not a valid feature. Should be str instance with fid after underscore (_)"
        )
    fid0 = [
        "W1",
        "W1-W2",
        "W2",
        "W2-W3",
        "W3",
        "W4",
        "g-W2",
        "g-W3",
        "g-r_ml",
        "gal_b",
        "gal_l",
        "r-W2",
        "r-W3",
        "rb",
        "sgscore1",
    ]
    fid12 = [
        "Multiband_period",
        "Period_fit",
        "g-r_max",
        "g-r_max_corr",
        "g-r_mean",
        "g-r_mean_corr",
        "PPE",
    ]
    if feature in fid0:
        return 0
    if feature in fid12 or feature.startswith("Power_rate"):
        return 12
    fid = feature.rsplit("_", 1)[-1]
    if fid.isdigit():
        return int(fid)
    return -99


def check_feature_name(name: str):
    fid = name.rsplit("_", 1)[-1]
    if name.startswith("Power_rate"):
        return name
    if fid.isdigit():
        return name.rsplit("_", 1)[0]

    return name
