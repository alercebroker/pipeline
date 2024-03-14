REFERENCE_KEYS = [
    "oid",
    "candid",
    "fid",
    "rfid",
    "rcid",
    "field",
    "magnr",
    "sigmagnr",
    "chinr",
    "sharpnr",
    "chinr",
    "ranr",
    "decnr",
    "nframesref",
    "mjdstartref",
    "mjdendref",
]

GAIA_KEYS = [
    "oid",
    "candid",
    "neargaia",
    "neargaiabright",
    "maggaia",
    "maggaiabright",
    "unique1",
]

SS_KEYS = ["oid", "candid", "ssdistnr", "ssmagnr", "ssnamenr"]

DATAQUALITY_KEYS = [
    "oid",
    "candid",
    "fid",
    "xpos",
    "ypos",
    "chipsf",
    "sky",
    "fwhm",
    "classtar",
    "mindtoedge",
    "seeratio",
    "aimage",
    "bimage",
    "aimagerat",
    "bimagerat",
    "nneg",
    "nbad",
    "sumrat",
    "scorr",
    "magzpsci",
    "magzpsciunc",
    "magzpscirms",
    "clrcoeff",
    "clrcounc",
    "dsnrms",
    "ssnrms",
    "nmatches",
    "zpclrcov",
    "zpmed",
    "clrmed",
    "clrrms",
    "exptime",
]

_PS1_MultKey = [
    "objectidps",
    "sgmag",
    "srmag",
    "simag",
    "szmag",
    "sgscore",
    "distpsnr",
]

PS1_KEYS = ["oid", "candid", "nmtchps"] + [f"{key}{i}" for key in _PS1_MultKey for i in range(1, 4)]
