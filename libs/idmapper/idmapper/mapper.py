import numpy as np


# Constants
SURVEY_IDS = {
    "ZTF": 1,
    "ATLAS": 2,
    "LSST": 3,
    "LS4": 4,
}
SURVEY_PREFIX_LEN_BITS = 8
SURVEY_IDS["MAXSURVEY"] = 2**SURVEY_PREFIX_LEN_BITS - 1

REVERSE_SURVEY_IDS = dict((zip(SURVEY_IDS.values(), SURVEY_IDS.keys())))


def catalog_oid_to_masterid(
    catalog: str, catalog_oid: str | np.int64 | int, validate: bool = False
) -> np.int64:
    """
    Convert a catalog object ID to a master ID.

    Parameters
    ----------
    catalog : str
        The name of the catalog (e.g., "ZTF").
    catalog_oid : str
        The ZTF object ID.
    validate: bool
        If True, validate the ztf_oid before conversion.

    Returns
    -------
    str
        The master ID.
    """
    if catalog not in SURVEY_IDS.keys():
        raise ValueError(f"Unsupported catalog: {catalog}")

    # Add the survey ID to the master ID
    master_id = SURVEY_IDS[catalog] << (63 - SURVEY_PREFIX_LEN_BITS)
    master_id = np.int64(master_id)

    if catalog == "ZTF":
        if not isinstance(catalog_oid, str):
            raise ValueError(f"Invalid ZTF object ID: {catalog_oid}")

        master_id += encode_ztf_to_masterid_without_survey(catalog_oid, validate)
    elif catalog == "LSST":
        # LSST conversion logic
        if isinstance(catalog_oid, np.int64):
            pass
        elif isinstance(catalog_oid, int):
            catalog_oid = np.int64(catalog_oid)
        elif isinstance(catalog_oid, str) and catalog_oid.isdigit():
            catalog_oid = np.int64(catalog_oid)
        else:
            raise ValueError(f"Invalid LSST object ID: {catalog_oid}")

        # LSST master ID is the same as the catalog_oid
        master_id += catalog_oid

    return master_id


def encode_ztf_to_masterid_without_survey(ztf_oid: str, validate: bool) -> np.int64:
    if validate and not is_ztf_oid_valid(ztf_oid):
        raise ValueError(f"Invalid ZTF object ID: {ztf_oid}")

    year = ztf_oid[3:5]
    seq = ztf_oid[5:12]

    # Convert the sequence of letters to a number
    master_id = 0
    for i, char in enumerate(seq):
        master_id += (ord(char) - ord("a")) * (26 ** (6 - i))

    # Convert the year to a number and add it to the master ID
    master_id += int(year) * 26**7
    return master_id


def is_ztf_oid_valid(ztf_oid: str) -> bool:
    """
    Checks that ztf_oid starts with ZTF, then two numbers and
    finally a sequence of 7 lowercase letters between a and z.

    :param ztf_oid: The ZTF object ID to validate.
    :return: True if ztf_oid is valid, False otherwise
    """
    if not isinstance(ztf_oid, str):
        return False
    if len(ztf_oid) != 12:
        return False
    if ztf_oid[0:3] != "ZTF":
        return False
    if not ztf_oid[3:5].isdigit():
        return False
    if not ztf_oid[5:12].isalpha():
        return False
    if not ztf_oid[5:12].islower():
        return False
    return True


def decode_masterid(masterid: np.int64) -> tuple[str, str | np.int64]:
    """
    Decode a master ID into its components.

    Parameters
    ----------
    masterid : np.int64
        The master ID.

    Returns
    -------
    tuple[str, str]
        The survey of the object and the original oid.
    """
    # Extract the survey from the master ID
    survey_id = masterid >> (63 - SURVEY_PREFIX_LEN_BITS)

    if survey_id in REVERSE_SURVEY_IDS.keys():
        survey = REVERSE_SURVEY_IDS[survey_id]
    else:
        raise ValueError(f"Invalid survey ID: {survey_id}")

    masterid_without_survey = np.bitwise_and(
        masterid, ((1 << (63 - SURVEY_PREFIX_LEN_BITS)) - 1)
    )

    if survey == "ZTF":
        oid = decode_masterid_for_ztf(masterid_without_survey)
        return "ZTF", oid

    elif survey == "LSST":
        return "LSST", masterid_without_survey

    else:
        raise ValueError(f"Unsupported survey ID: {survey_id}")


def decode_masterid_for_ztf(masterid_without_survey: np.int64) -> str:
    """
    Decode a master ID into its components.

    Parameters
    ----------
    masterid_without_survey : np.int64
        The master ID without the survey ID.

    Returns
    -------
    str
        The original oid.
    """
    year = (masterid_without_survey // (26**7)) % 100
    seq = masterid_without_survey % (26**7)

    # Convert the sequence of numbers back to letters
    seq_str = ""
    for i in range(6, -1, -1):
        seq_str += chr((seq // (26**i)) + ord("a"))
        seq %= 26**i

    return f"ZTF{year:02d}{seq_str}"
