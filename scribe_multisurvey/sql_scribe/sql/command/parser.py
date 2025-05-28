def multistream_detection_to_ztf(command: dict):
    if "oid" not in command:
        raise ValueError("OID was not found")

    if command["sid"] != "ZTF":
        raise ValueError("Detection not from ZTF survey")

    mapping = {
        "mag": "magpsf",
        "e_mag": "sigmapsf",
        "mag_corr": "magpsf_corr",
        "e_mag_corr": "sigmapsf_corr",
        "e_mag_corr_ext": "sigmapsf_corr_ext",
    }

    fid_map = {"g": 1, "r": 2, "i": 3}

    exclude = [
        "aid",
        "sid",
        "tid",
        "new",
        "pid",
        "e_ra",
        "e_dec",
        "extra_fields",
    ]

    new_command = {k: v for k, v in command.items() if k not in exclude}
    for k, v in mapping.items():
        if k in new_command:
            new_command[v] = new_command.pop(k)

    new_command["fid"] = fid_map[new_command["fid"]]
    new_command["candid"] = int(new_command["candid"])
    new_command["parent_candid"] = (
        int(new_command["parent_candid"])
        if new_command["parent_candid"]
        else None
    )

    return new_command


def parse_ztf_pf(raw_detection: dict, oid: str) -> dict:
    forced_photometry = {
        "oid": oid,
        "measurement_id": raw_detection["measurement_id"],
        "pid": raw_detection["pid"],
        "mag": raw_detection["mag"],
        "e_mag": raw_detection["e_mag"],
        "mag_corr": raw_detection["mag_corr"],
        "e_mag_corr": raw_detection["e_mag_corr"],
        "e_mag_corr_ext": raw_detection["e_mag_corr_ext"],
        "isdiffpos": raw_detection["isdiffpos"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
        "field": raw_detection["extra_fields"]["field"],
        "rcid": raw_detection["extra_fields"]["rcid"],
        "rfid": raw_detection["extra_fields"]["rfid"],
        "sciinpseeing": raw_detection["extra_fields"]["sciinpseeing"],
        "scibckgnd": raw_detection["extra_fields"]["scibckgnd"],
        "scisigpix": raw_detection["extra_fields"]["scisigpix"],
        "magzpsci": raw_detection["extra_fields"]["magzpsci"],
        "magzpsciunc": raw_detection["extra_fields"]["magzpsciunc"],
        "magzpscirms": raw_detection["extra_fields"]["magzpscirms"],
        "clrcoeff": raw_detection["extra_fields"]["clrcoeff"],
        "clrcounc": raw_detection["extra_fields"]["clrcounc"],
        "exptime": raw_detection["extra_fields"]["exptime"],
        "adpctdif1": raw_detection["extra_fields"]["adpctdif1"],
        "adpctdif2": raw_detection["extra_fields"]["adpctdif2"],
        "diffmaglim": raw_detection["extra_fields"]["diffmaglim"],
        "programid": raw_detection["extra_fields"]["programid"],
        "procstatus": raw_detection["extra_fields"]["procstatus"],
        "distnr": raw_detection["extra_fields"]["distnr"],
        "ranr": raw_detection["extra_fields"]["ranr"],
        "decnr": raw_detection["extra_fields"]["decnr"],
        "magnr": raw_detection["extra_fields"]["magnr"],
        "sigmagnr": raw_detection["extra_fields"]["sigmagnr"],
        "chinr": raw_detection["extra_fields"]["chinr"],
        "sharpnr": raw_detection["extra_fields"]["sharpnr"],
    }
    return forced_photometry

def parse_ztf_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "oid": oid,
        "measurement_id": raw_detection["measurement_id"],
        "pid": raw_detection["pid"],
        "diffmaglim": raw_detection["extra_fields"]["diffmaglim"],
        "isdiffpos": raw_detection["isdiffpos"],
        "nid": raw_detection["extra_fields"]["nid"],
        "magpsf": raw_detection["mag"],
        "sigmapsf": raw_detection["e_mag"],
        "magap": raw_detection["extra_fields"]["magap"],
        "sigmagap": raw_detection["extra_fields"]["sigmagap"],
        "distnr": raw_detection["extra_fields"]["distnr"],
        "rb": raw_detection["extra_fields"]["rb"],
        "rbversion": raw_detection["extra_fields"]["rbversion"],
        "drb": raw_detection["extra_fields"]["drb"],
        "drbversion": raw_detection["extra_fields"]["drbversion"],
        "magapbig": raw_detection["extra_fields"]["magapbig"],
        "sigmagapbig": raw_detection["extra_fields"]["sigmagapbig"],
        "rfid": raw_detection["extra_fields"]["rfid"],
        "magpsf_corr": raw_detection["mag_corr"],
        "sigmapsf_corr": raw_detection["e_mag_corr"],
        "sigmapsf_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
    }
    return detection

def parse_ztf_ps1(candidate: dict, oid: str) -> dict:
    ps1 = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "ssdistnr": candidate["extra_fields"]["ssdistnr"],
        "ssmagnr": candidate["extra_fields"]["ssmagnr"],
        "ssnamenr": candidate["extra_fields"]["ssnamenr"],
    }
    return ps1

def parse_ztf_ss(candidate: dict, oid: str) -> dict:
    ss = {
        "oid": oid,
        "measurement_id": candidate["extra_fields"]["measure"],
        "objectidps1": candidate["extra_fields"]["objecti"],
        "sgmag1": candidate["extra_fields"]["sgmag1 "],
        "srmag1": candidate["extra_fields"]["srmag1 "],
        "simag1": candidate["extra_fields"]["simag1 "],
        "szmag1": candidate["extra_fields"]["szmag1 "],
        "sgscore1": candidate["extra_fields"]["sgscore"],
        "distpsnr1": candidate["extra_fields"]["distpsn"],
        "objectidps2": candidate["extra_fields"]["objecti"],
        "sgmag2": candidate["extra_fields"]["sgmag2 "],
        "srmag2": candidate["extra_fields"]["srmag2 "],
        "simag2": candidate["extra_fields"]["simag2 "],
        "szmag2": candidate["extra_fields"]["szmag2 "],
        "sgscore2": candidate["extra_fields"]["sgscore"],
        "distpsnr2": candidate["extra_fields"]["distpsn"],
        "objectidps3": candidate["extra_fields"]["objecti"],
        "sgmag3": candidate["extra_fields"]["sgmag3 "],
        "srmag3": candidate["extra_fields"]["srmag3 "],
        "simag3": candidate["extra_fields"]["simag3 "],
        "szmag3": candidate["extra_fields"]["szmag3 "],
        "sgscore3": candidate["extra_fields"]["sgscore"],
        "distpsnr3": candidate["extra_fields"]["distpsn"],
        "nmtchps": candidate["extra_fields"]["nmtchps"],
        "unique1": candidate["extra_fields"]["unique1"],
        "unique2": candidate["extra_fields"]["unique2"],
        "unique3": candidate["extra_fields"]["unique3"],
    }
    return ss

def parse_ztf_gaia(candidate: dict, oid: str) -> dict:
    gaia = {

    }
    return gaia

def parse_ztf_dq(candidate: dict, oid: str) -> dict:
    dq = {

    }
    return dq
