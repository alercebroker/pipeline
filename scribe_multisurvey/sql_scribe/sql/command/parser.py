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

"""
def parse_ztf_pf(raw_detection: dict, oid: str):
    forced_photometry = {
        "b_oid": oid,
        "b_measurement_id": raw_detection["measurement_id"],
        "mag_corr": raw_detection["mag_corr"],
        "e_mag_corr": raw_detection["e_mag_corr"],
        "e_mag_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        
    }
    return forced_photometry

def parse_ztf_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "b_oid": oid,
        "b_measurement_id": raw_detection["measurement_id"],
        "magpsf_corr": raw_detection["mag_corr"],
        "sigmapsf_corr": raw_detection["e_mag_corr"],
        "sigmapsf_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
    }
    return detection

"""

def parse_fp(raw_detection: dict, oid: str) -> dict:
    forced_photometry = {
        "oid": oid,
        "measurement_id": raw_detection["measurement_id"],
        "mjd": raw_detection["mjd"],
        "ra": raw_detection["ra"],
        "dec": raw_detection["dec"],
        "band": raw_detection["band"],
    }
    return forced_photometry

def parse_ztf_fp(raw_detection: dict, oid: str) -> dict:
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

def parse_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "oid": oid,
        "measurement_id": raw_detection["measurement_id"],
        "mjd": raw_detection["mjd"],
        "ra": raw_detection["ra"],
        "dec": raw_detection["dec"],
        "band": raw_detection["band"],
    }

    return detection

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
        "drb": raw_detection["extra_fields"].get("drb", None),
        "drbversion": raw_detection["extra_fields"].get("drbversion", None),
        "magapbig": raw_detection["extra_fields"]["magapbig"],
        "sigmagapbig": raw_detection["extra_fields"]["sigmagapbig"],
        "rfid": raw_detection["extra_fields"].get("rfid", None),
        "magpsf_corr": raw_detection["mag_corr"],
        "sigmapsf_corr": raw_detection["e_mag_corr"],
        "sigmapsf_corr_ext": raw_detection["e_mag_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
    }
    return detection

    
def parse_ztf_ss(candidate: dict, oid: str) -> dict:
    ss = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "ssdistnr": candidate["extra_fields"]["ssdistnr"],
        "ssmagnr": candidate["extra_fields"]["ssmagnr"],
        "ssnamenr":None if candidate["extra_fields"]["ssnamenr"] == 'null' else  candidate["extra_fields"]["ssnamenr"],
    }
    return ss

def parse_ztf_ps1(candidate: dict, oid: str) -> dict:
    ps1 = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "objectidps1": (
            int(candidate["extra_fields"]["objectidps1"])
            if candidate["extra_fields"]["objectidps1"] != "-999" or candidate["extra_fields"]["objectidps1"] is not None
            else None
        ),
        "sgmag1": candidate["extra_fields"]["sgmag1"],
        "srmag1": candidate["extra_fields"]["srmag1"],
        "simag1": candidate["extra_fields"]["simag1"],
        "szmag1": candidate["extra_fields"]["szmag1"],
        "sgscore1": candidate["extra_fields"]["sgscore1"],
        "distpsnr1": candidate["extra_fields"]["distpsnr1"],
        "objectidps2":(
            int(candidate["extra_fields"]["objectidps2"])
            if candidate["extra_fields"]["objectidps2"] != "-999" or candidate["extra_fields"]["objectidps2"] is not None
            else None
        ),
        "sgmag2": candidate["extra_fields"]["sgmag2"],
        "srmag2": candidate["extra_fields"]["srmag2"],
        "simag2": candidate["extra_fields"]["simag2"],
        "szmag2": candidate["extra_fields"]["szmag2"],
        "sgscore2": candidate["extra_fields"]["sgscore2"],
        "distpsnr2": candidate["extra_fields"]["distpsnr2"],
        "objectidps3":(
            int(candidate["extra_fields"]["objectidps3"])
            if candidate["extra_fields"]["objectidps3"] != "-999" or candidate["extra_fields"]["objectidps3"] is not None
            else None
        ),
        "sgmag3": candidate["extra_fields"]["sgmag3"],
        "srmag3": candidate["extra_fields"]["srmag3"],
        "simag3": candidate["extra_fields"]["simag3"],
        "szmag3": candidate["extra_fields"]["szmag3"],
        "sgscore3": candidate["extra_fields"]["sgscore3"],
        "distpsnr3": candidate["extra_fields"]["distpsnr3"],
        "nmtchps": candidate["extra_fields"]["nmtchps"],
    }
    return ps1

def parse_ztf_gaia(candidate: dict, oid: str) -> dict:
    gaia = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "neargaia": candidate["extra_fields"]["neargaia"],
        "neargaiabright": candidate["extra_fields"]["neargaiabright"],
        "maggaia": candidate["extra_fields"]["maggaia"],
        "maggaiabright": candidate["extra_fields"]["maggaiabright"],
    }
    return gaia

def parse_ztf_dq(candidate: dict, oid: str) -> dict:
    dq = {
        "oid": oid,
        "measurement_id": candidate["measurement_id"],
        "xpos": candidate["extra_fields"]["xpos"],
        "ypos": candidate["extra_fields"]["ypos"],
        "chipsf": candidate["extra_fields"]["chipsf"],
        "sky": candidate["extra_fields"]["sky"],
        "fwhm": candidate["extra_fields"]["fwhm"],
        "classtar": candidate["extra_fields"]["classtar"],
        "mindtoedge": candidate["extra_fields"]["mindtoedge"],
        "seeratio": candidate["extra_fields"]["seeratio"],
        "aimage": candidate["extra_fields"]["aimage"],
        "bimage": candidate["extra_fields"]["bimage"],
        "aimagerat": candidate["extra_fields"]["aimagerat"],
        "bimagerat": candidate["extra_fields"]["bimagerat"],
        "nneg": candidate["extra_fields"]["nneg"],
        "nbad": candidate["extra_fields"]["nbad"],
        "sumrat": candidate["extra_fields"]["sumrat"],
        "scorr": candidate["extra_fields"]["scorr"],
        "dsnrms": candidate["extra_fields"]["dsnrms"],
        "ssnrms": candidate["extra_fields"]["ssnrms"],
        "magzpsci": candidate["extra_fields"]["magzpsci"],
        "magzpsciunc": candidate["extra_fields"]["magzpsciunc"],
        "magzpscirms": candidate["extra_fields"]["magzpscirms"],
        "nmatches": candidate["extra_fields"]["nmatches"],
        "clrcoeff": candidate["extra_fields"]["clrcoeff"],
        "clrcounc": candidate["extra_fields"]["clrcounc"],
        "zpclrcov": candidate["extra_fields"]["zpclrcov"],
        "zpmed": candidate["extra_fields"]["zpmed"],
        "clrmed": candidate["extra_fields"]["clrmed"],
        "clrrms": candidate["extra_fields"]["clrrms"],
        "exptime": candidate["extra_fields"]["exptime"],
    }
    return dq

def parse_ztf_refernece(candidate: dict, oid: str) -> dict:
    reference = {
        "oid": oid,
        "rfid": candidate["extra_fields"]["rfid"],
        "measurement_id": candidate["measurement_id"],
        "band": candidate["band"],
        "rcid": candidate["extra_fields"]["rcid"],
        "field": candidate["extra_fields"]["field"],
        "magnr": candidate["extra_fields"]["magnr"],
        "sigmagnr": candidate["extra_fields"]["sigmagnr"],
        "chinr": candidate["extra_fields"]["chinr"],
        "sharpnr": candidate["extra_fields"]["sharpnr"],
        "ranr": candidate["extra_fields"]["ranr"],
        "decnr": candidate["extra_fields"]["decnr"],
        "mjdstartref": candidate["extra_fields"]["jdstartref"] - 2400000.5,
        "mjdendref": candidate["extra_fields"]["jdendref"] - 2400000.5,
        "nframesref": candidate["extra_fields"]["nframesref"],
    }
    return reference

def parse_obj_stats(raw_magstats, oid: str) -> dict:
    obj = {
        "_oid": oid,
        "oid": oid,
        "meanra": raw_magstats["meanra"],
        "meandec": raw_magstats["meandec"],
        "sigmara": raw_magstats["sigmara"],
        "sigmadec": raw_magstats["sigmadec"],
        "firstmjd": raw_magstats["firstmjd"],
        "lastmjd": raw_magstats["lastmjd"],
        "deltamjd": raw_magstats["deltajd"],
        "n_det": raw_magstats["n_det"],
        "n_forced": raw_magstats["n_fphot"] if raw_magstats["n_fphot"] else 0,
        "n_non_det": raw_magstats["n_ndet"] if raw_magstats["n_ndet"] else 0,
        "corrected": raw_magstats["corrected"],
        "stellar": raw_magstats["stellar"],
    }

    return obj

def parse_magstats(sub_magstats: dict, oid: str) -> dict:
    magstats = {
        "oid": oid,
        "band": sub_magstats["band"], 
        "stellar": sub_magstats["stellar"], 
        "corrected": sub_magstats["corrected"], 
        "ndubious": sub_magstats["ndubious"], 
        "dmdt_first": sub_magstats["dmdt_first"], 
        "dm_first": sub_magstats["dm_first"], 
        "sigmadm_first": sub_magstats["sigmadm_first"], 
        "dt_first": sub_magstats["dt_first"], 
        "magmean": sub_magstats["magmean"], 
        "magmedian": sub_magstats["magmedian"], 
        "magmax": sub_magstats["magmax"], 
        "magmin": sub_magstats["magmin"], 
        "magsigma": sub_magstats["magsigma"], 
        "maglast": sub_magstats["maglast"], 
        "magfirst": sub_magstats["magfirst"], 
        "magmean_corr": sub_magstats["magmean_corr"], 
        "magmedian_corr": sub_magstats["magmedian_corr"], 
        "magmax_corr": sub_magstats["magmax_corr"], 
        "magmin_corr": sub_magstats["magmin_corr"], 
        "magsigma_corr": sub_magstats["magsigma_corr"], 
        "maglast_corr": sub_magstats["maglast_corr"], 
        "magfirst_corr": sub_magstats["magfirst_corr"], 
        "saturation_rate": sub_magstats["saturation_rate"], 
    }

    return magstats