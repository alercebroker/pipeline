def multisurvey_detection_to_ztf(command: dict):
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
        "tid",
        "new",
        "pid",
        "e_ra",
        "e_dec",
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
        "sid": raw_detection["sid"],
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
        "sid": raw_detection["sid"],
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
        "field": raw_detection["field"],
        "rcid": raw_detection["rcid"],
        "rfid": raw_detection["rfid"],
        "sciinpseeing": raw_detection["sciinpseeing"],
        "scibckgnd": raw_detection["scibckgnd"],
        "scisigpix": raw_detection["scisigpix"],
        "magzpsci": raw_detection["magzpsci"],
        "magzpsciunc": raw_detection["magzpsciunc"],
        "magzpscirms": raw_detection["magzpscirms"],
        "clrcoeff": raw_detection["clrcoeff"],
        "clrcounc": raw_detection["clrcounc"],
        "exptime": raw_detection["exptime"],
        "adpctdif1": raw_detection["adpctdif1"],
        "adpctdif2": raw_detection["adpctdif2"],
        "diffmaglim": raw_detection["diffmaglim"],
        "programid": raw_detection["programid"],
        "procstatus": raw_detection["procstatus"],
        "distnr": raw_detection["distnr"],
        "ranr": raw_detection["ranr"],
        "decnr": raw_detection["decnr"],
        "magnr": raw_detection["magnr"],
        "sigmagnr": raw_detection["sigmagnr"],
        "chinr": raw_detection["chinr"],
        "sharpnr": raw_detection["sharpnr"],
    }

    return forced_photometry

def parse_det(raw_detection: dict, oid: str) -> dict:
    detection = {
        "oid": oid,
        "sid": raw_detection["sid"],
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
        "sid": raw_detection["sid"],
        "measurement_id": raw_detection["measurement_id"],
        "pid": raw_detection["pid"],
        "diffmaglim": raw_detection["diffmaglim"],
        "isdiffpos": raw_detection["isdiffpos"],
        "nid": raw_detection["nid"],
        "magpsf": raw_detection["mag"],
        "sigmapsf": raw_detection["e_mag"],
        "magap": raw_detection["magap"],
        "sigmagap": raw_detection["sigmagap"],
        "distnr": raw_detection["distnr"],
        "rb": raw_detection["rb"],
        "rbversion": raw_detection["rbversion"],
        "drb": raw_detection.get("drb", None),
        "drbversion": raw_detection.get("drbversion", None),
        "magapbig": raw_detection["magapbig"],
        "sigmagapbig": raw_detection["sigmagapbig"],
        "rfid": raw_detection.get("rfid", None),
        "magpsf_corr": raw_detection["magpsf_corr"],
        "sigmapsf_corr": raw_detection["sigmapsf_corr"],
        "sigmapsf_corr_ext": raw_detection["sigmapsf_corr_ext"],
        "corrected": raw_detection["corrected"],
        "dubious": raw_detection["dubious"],
        "parent_candid": raw_detection["parent_candid"],
        "has_stamp": raw_detection["has_stamp"],
    }
    return detection

    
def parse_ztf_ss(candidate: dict, oid: str) -> dict:
    ss = {
        "oid": oid,
        "_mjd": candidate["mjd"],
        "measurement_id": candidate["measurement_id"],
        "ssdistnr": candidate["ssdistnr"],
        "ssmagnr": candidate["ssmagnr"],
        "ssnamenr":None if candidate["ssnamenr"] == 'null' else  candidate["ssnamenr"],
    }
    return ss

def parse_ztf_object(candidate: dict, oid: str) -> dict:
    ztfobj = {
        "oid": oid,
       "_oid": oid,
        "_mjd": candidate["mjd"],
        "ndethist": candidate["ndethist"],
        "ncovhist": candidate["ncovhist"],
        "mjdstarthist": candidate["jdstarthist"] - 2400000.5,
        "mjdendhist": candidate["jdendhist"] - 2400000.5
    }
    return ztfobj

def parse_ztf_ps1(candidate: dict, oid: str) -> dict:
    ps1 = {
        "oid": oid,
        "_mjd": candidate["mjd"],
        "measurement_id": candidate["measurement_id"],
        "objectidps1": (
            int(candidate["objectidps1"])
            if candidate["objectidps1"] != "-999" or candidate["objectidps1"] is not None
            else None
        ),
        "sgmag1": candidate["sgmag1"],
        "srmag1": candidate["srmag1"],
        "simag1": candidate["simag1"],
        "szmag1": candidate["szmag1"],
        "sgscore1": candidate["sgscore1"],
        "distpsnr1": candidate["distpsnr1"],
        "objectidps2":(
            int(candidate["objectidps2"])
            if candidate["objectidps2"] != "-999" or candidate["objectidps2"] is not None
            else None
        ),
        "sgmag2": candidate["sgmag2"],
        "srmag2": candidate["srmag2"],
        "simag2": candidate["simag2"],
        "szmag2": candidate["szmag2"],
        "sgscore2": candidate["sgscore2"],
        "distpsnr2": candidate["distpsnr2"],
        "objectidps3":(
            int(candidate["objectidps3"])
            if candidate["objectidps3"] != "-999" or candidate["objectidps3"] is not None
            else None
        ),
        "sgmag3": candidate["sgmag3"],
        "srmag3": candidate["srmag3"],
        "simag3": candidate["simag3"],
        "szmag3": candidate["szmag3"],
        "sgscore3": candidate["sgscore3"],
        "distpsnr3": candidate["distpsnr3"],
        "nmtchps": candidate["nmtchps"],
    }
    return ps1

def parse_ztf_gaia(candidate: dict, oid: str) -> dict:
    gaia = {
        "oid": oid,
        "_mjd": candidate["mjd"],
        "measurement_id": candidate["measurement_id"],
        "neargaia": candidate["neargaia"],
        "neargaiabright": candidate["neargaiabright"],
        "maggaia": candidate["maggaia"],
        "maggaiabright": candidate["maggaiabright"],
    }
    return gaia

def parse_ztf_dq(candidate: dict, oid: str) -> dict:
    dq = {
        "oid": oid,
        "_mjd": candidate["mjd"],
        "measurement_id": candidate["measurement_id"],
        "xpos": candidate["xpos"],
        "ypos": candidate["ypos"],
        "chipsf": candidate["chipsf"],
        "sky": candidate["sky"],
        "fwhm": candidate["fwhm"],
        "classtar": candidate["classtar"],
        "mindtoedge": candidate["mindtoedge"],
        "seeratio": candidate["seeratio"],
        "aimage": candidate["aimage"],
        "bimage": candidate["bimage"],
        "aimagerat": candidate["aimagerat"],
        "bimagerat": candidate["bimagerat"],
        "nneg": candidate["nneg"],
        "nbad": candidate["nbad"],
        "sumrat": candidate["sumrat"],
        "scorr": candidate["scorr"],
        "dsnrms": candidate["dsnrms"],
        "ssnrms": candidate["ssnrms"],
        "magzpsci": candidate["magzpsci"],
        "magzpsciunc": candidate["magzpsciunc"],
        "magzpscirms": candidate["magzpscirms"],
        "nmatches": candidate["nmatches"],
        "clrcoeff": candidate["clrcoeff"],
        "clrcounc": candidate["clrcounc"],
        "zpclrcov": candidate["zpclrcov"],
        "zpmed": candidate["zpmed"],
        "clrmed": candidate["clrmed"],
        "clrrms": candidate["clrrms"],
        "exptime": candidate["exptime"],
    }
    return dq

def parse_ztf_refernece(candidate: dict, oid: str) -> dict:
    reference = {
        "oid": oid,
        "rfid": candidate["rfid"],
        "_mjd": candidate["mjd"],
        "measurement_id": candidate["measurement_id"],
        "band": candidate["band"],
        "rcid": candidate["rcid"],
        "field": candidate["field"],
        "magnr": candidate["magnr"],
        "sigmagnr": candidate["sigmagnr"],
        "chinr": candidate["chinr"],
        "sharpnr": candidate["sharpnr"],
        "ranr": candidate["ranr"],
        "decnr": candidate["decnr"],
        "mjdstartref": candidate["jdstartref"] - 2400000.5,
        "mjdendref": candidate["jdendref"] - 2400000.5,
        "nframesref": candidate["nframesref"],
    }
    return reference

def parse_obj_stats(raw_magstats, oid: str, sid: int) -> dict:
    obj = {
        "_oid": oid,
        "oid": oid,
        "sid": sid,
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

def parse_ztf_objstats(raw_magstats, oid: str, sid: int) -> dict:
    obj = {
         "_oid": oid,
        "oid": oid,
        "sid": sid,
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
        "reference_change": raw_magstats["reference_change"],
        "diffpos": raw_magstats["diffpos"]
    }
    return obj

def parse_ztf_magstats(sub_magstats: dict, oid: str, sid: int):
    ztf_magstats = {
        "oid": oid,
        "sid": sid,
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
        "n_det": sub_magstats["ndet"],
        "firstmjd": sub_magstats["firstmjd"],
        "lastmjd": sub_magstats["lastmjd"]
    }
    return ztf_magstats


def parse_dia_object(object, oid: str) -> dict:

    obj = {
        "_oid": oid,
        "oid": oid,
        "validityStartMjdTai": object["validityStartMjdTai"],
        "ra": object["ra"],
        "raErr": object["raErr"],
        "dec": object["dec"],
        "decErr": object["decErr"],
        "ra_dec_Cov": object["ra_dec_Cov"],
        "u_psfFluxMean": object["u_psfFluxMean"],
        "u_psfFluxMeanErr": object["u_psfFluxMeanErr"],
        "u_psfFluxSigma": object["u_psfFluxSigma"],
        "u_psfFluxNdata": object["u_psfFluxNdata"],
        "u_fpFluxMean": object["u_fpFluxMean"],
        "u_fpFluxMeanErr": object["u_fpFluxMeanErr"],
        "g_psfFluxMean": object["g_psfFluxMean"],
        "g_psfFluxMeanErr": object["g_psfFluxMeanErr"],
        "g_psfFluxSigma": object["g_psfFluxSigma"],
        "g_psfFluxNdata": object["g_psfFluxNdata"],
        "g_fpFluxMean": object["g_fpFluxMean"],
        "g_fpFluxMeanErr": object["g_fpFluxMeanErr"],
        "r_psfFluxMean": object["r_psfFluxMean"],
        "r_psfFluxMeanErr": object["r_psfFluxMeanErr"],
        "r_psfFluxSigma": object["r_psfFluxSigma"],
        "r_psfFluxNdata": object["r_psfFluxNdata"],
        "r_fpFluxMean": object["r_fpFluxMean"],
        "r_fpFluxMeanErr": object["r_fpFluxMeanErr"],
        "i_psfFluxMean": object["i_psfFluxMean"],
        "i_psfFluxMeanErr": object["i_psfFluxMeanErr"],
        "i_psfFluxSigma": object["i_psfFluxSigma"],
        "i_psfFluxNdata": object["i_psfFluxNdata"],
        "i_fpFluxMean": object["i_fpFluxMean"],
        "i_fpFluxMeanErr": object["i_fpFluxMeanErr"],
        "z_psfFluxMean": object["z_psfFluxMean"],
        "z_psfFluxMeanErr": object["z_psfFluxMeanErr"],
        "z_psfFluxSigma": object["z_psfFluxSigma"],
        "z_psfFluxNdata": object["z_psfFluxNdata"],
        "z_fpFluxMean": object["z_fpFluxMean"],
        "z_fpFluxMeanErr": object["z_fpFluxMeanErr"],
        "y_psfFluxMean": object["y_psfFluxMean"],
        "y_psfFluxMeanErr": object["y_psfFluxMeanErr"],
        "y_psfFluxSigma": object["y_psfFluxSigma"],
        "y_psfFluxNdata": object["y_psfFluxNdata"],
        "y_fpFluxMean": object["y_fpFluxMean"],
        "y_fpFluxMeanErr": object["y_fpFluxMeanErr"],
        "u_scienceFluxMean": object["u_scienceFluxMean"],
        "u_scienceFluxMeanErr": object["u_scienceFluxMeanErr"],
        "g_scienceFluxMean": object["g_scienceFluxMean"],
        "g_scienceFluxMeanErr": object["g_scienceFluxMeanErr"],
        "r_scienceFluxMean": object["r_scienceFluxMean"],
        "r_scienceFluxMeanErr": object["r_scienceFluxMeanErr"],
        "i_scienceFluxMean": object["i_scienceFluxMean"],
        "i_scienceFluxMeanErr": object["i_scienceFluxMeanErr"],
        "z_scienceFluxMean": object["z_scienceFluxMean"],
        "z_scienceFluxMeanErr": object["z_scienceFluxMeanErr"],
        "y_scienceFluxMean": object["y_scienceFluxMean"],
        "y_scienceFluxMeanErr": object["y_scienceFluxMeanErr"],
        "u_psfFluxMin": object["u_psfFluxMin"],
        "u_psfFluxMax": object["u_psfFluxMax"],
        "u_psfFluxMaxSlope": object["u_psfFluxMaxSlope"],
        "u_psfFluxErrMean": object["u_psfFluxErrMean"],
        "g_psfFluxMin": object["g_psfFluxMin"],
        "g_psfFluxMax": object["g_psfFluxMax"],
        "g_psfFluxMaxSlope": object["g_psfFluxMaxSlope"],
        "g_psfFluxErrMean": object["g_psfFluxErrMean"],
        "r_psfFluxMin": object["r_psfFluxMin"],
        "r_psfFluxMax": object["r_psfFluxMax"],
        "r_psfFluxMaxSlope": object["r_psfFluxMaxSlope"],
        "r_psfFluxErrMean": object["r_psfFluxErrMean"],
        "i_psfFluxMin": object["i_psfFluxMin"],
        "i_psfFluxMax": object["i_psfFluxMax"],
        "i_psfFluxMaxSlope": object["i_psfFluxMaxSlope"],
        "i_psfFluxErrMean": object["i_psfFluxErrMean"],
        "z_psfFluxMin": object["z_psfFluxMin"],
        "z_psfFluxMax": object["z_psfFluxMax"],
        "z_psfFluxMaxSlope": object["z_psfFluxMaxSlope"],
        "z_psfFluxErrMean": object["z_psfFluxErrMean"],
        "y_psfFluxMin": object["y_psfFluxMin"],
        "y_psfFluxMax": object["y_psfFluxMax"],
        "y_psfFluxMaxSlope": object["y_psfFluxMaxSlope"],
        "y_psfFluxErrMean": object["y_psfFluxErrMean"],
        "firstDiaSourceMjdTai": object["firstDiaSourceMjdTai"],
        "lastDiaSourceMjdTai": object["lastDiaSourceMjdTai"],
        "nDiaSources": object["nDiaSources"],
    }

    return obj

def parse_magstats(sub_magstats: dict, oid: str, sid: int) -> dict:
    magstats = {
        "oid": oid,
        "sid": sid,
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

def parse_mpc_orbits(mpc_orbits: dict) -> dict:
    parsed_mpc_orbit = {
        "ssObjectId": mpc_orbits["ssObjectId"],
        "mjd": mpc_orbits["mjd"],
        "designation": mpc_orbits["designation"],
        "packed_primary_provisional_designation": mpc_orbits["packed_primary_provisional_designation"],
        "unpacked_primary_provisional_designation": mpc_orbits["unpacked_primary_provisional_designation"],
        "mpc_orb_jsonb": mpc_orbits["mpc_orb_jsonb"],
        "created_at": mpc_orbits["created_at"],
        "updated_at": mpc_orbits["updated_at"],
        "orbit_type_int": mpc_orbits["orbit_type_int"],
        "u_param": mpc_orbits["u_param"],
        "nopp": mpc_orbits["nopp"],
        "arc_length_total": mpc_orbits["arc_length_total"],
        "arc_length_sel": mpc_orbits["arc_length_sel"],
        "nobs_total": mpc_orbits["nobs_total"],
        "nobs_total_sel": mpc_orbits["nobs_total_sel"],
        "a": mpc_orbits["a"],
        "q": mpc_orbits["q"],
        "e": mpc_orbits["e"],
        "i": mpc_orbits["i"],
        "node": mpc_orbits["node"],
        "peri_time": mpc_orbits["peri_time"],
        "yarkovsky": mpc_orbits["yarkovsky"],
        "srp": mpc_orbits["srp"],
        "a1": mpc_orbits["a1"],
        "a2": mpc_orbits["a2"],
        "a3": mpc_orbits["a3"],
        "dt": mpc_orbits["dt"],
        "mean_anomaly": mpc_orbits["mean_anomaly"],
        "period": mpc_orbits["period"],
        "mean_motion": mpc_orbits["mean_motion"],
        "a_unc": mpc_orbits["a_unc"],
        "q_unc": mpc_orbits["q_unc"],
        "e_unc": mpc_orbits["e_unc"],
        "i_unc": mpc_orbits["i_unc"],
        "node_unc": mpc_orbits["node_unc"],
        "argperi_unc": mpc_orbits["argperi_unc"],
        "peri_time_unc": mpc_orbits["peri_time_unc"],
        "yarkovsky_unc": mpc_orbits["yarkovsky_unc"],
        "srp_unc": mpc_orbits["srp_unc"],
        "a1_unc": mpc_orbits["a1_unc"],
        "a2_unc": mpc_orbits["a2_unc"],
        "a3_unc": mpc_orbits["a3_unc"],
        "dt_unc": mpc_orbits["dt_unc"],
        "mean_anomaly_unc": mpc_orbits["mean_anomaly_unc"],
        "period_unc": mpc_orbits["period_unc"],
        "mean_motion_unc": mpc_orbits["mean_motion_unc"],
        "epoch_mjd": mpc_orbits["epoch_mjd"],
        "h": mpc_orbits["h"],
        "g": mpc_orbits["g"],
        "not_normalized_rms": mpc_orbits["not_normalized_rms"],
        "normalized_rms": mpc_orbits["normalized_rms"],
        "earth_moid": mpc_orbits["earth_moid"],
        "fitting_datetime": mpc_orbits["fitting_datetime"]
    }
    return parsed_mpc_orbit