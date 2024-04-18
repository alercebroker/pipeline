import logging
import numpy as np
import pandas as pd
from lc_classifier.base import AstroObject, query_ao_table
from features.core.utils.functions import collapse_fid_columns
from typing import List, Dict, Optional


def detections_to_astro_objects(
    detections: List[Dict], xmatches: Optional[Dict]
) -> AstroObject:
    detection_keys = [
        "oid",
        "candid",
        "aid",
        "tid",
        "sid",
        "pid",
        "ra",
        "dec",
        "mjd",
        "mag_corr",
        "e_mag_corr",
        "mag",
        "e_mag",
        "fid",
        "isdiffpos",
        "forced",
        # 'sgscore1'
    ]

    values = []
    for detection in detections:
        values.append([detection[key] for key in detection_keys])

    a = pd.DataFrame(data=values, columns=detection_keys)
    a.fillna(value=np.nan, inplace=True)
    a.rename(
        columns={"mag_corr": "brightness", "e_mag_corr": "e_brightness"}, inplace=True
    )
    a["unit"] = "magnitude"
    a_flux = a.copy()
    a_flux["brightness"] = 10.0 ** (-0.4 * (a["mag"] - 23.9)) * a["isdiffpos"]
    a_flux["e_brightness"] = a_flux["e_mag"] * 0.4 * np.log(10) * np.abs(a_flux["mag"])
    a_flux["unit"] = "diff_flux"
    a = pd.concat([a, a_flux], axis=0)
    a.set_index("aid", inplace=True)

    aid = a.index.values[0]
    oid = a["oid"].iloc[0]

    aid_forced = a[a["forced"]]
    aid_detections = a[~a["forced"]]

    w1 = w2 = w3 = w4 = np.nan
    if xmatches is not None and "allwise" in xmatches.keys():
        w1 = xmatches["allwise"]["W1mag"]
        w2 = xmatches["allwise"]["W2mag"]
        w3 = xmatches["allwise"]["W3mag"]
        w4 = xmatches["allwise"]["W4mag"]

    metadata = pd.DataFrame(
        [
            ["aid", aid],
            ["oid", oid],
            ["W1", w1],
            ["W2", w2],
            ["W3", w3],
            ["W4", w4],
            ["sgscore1", detections[0]["extra_fields"]["sgscore1"]],
            ["sgmag1", detections[0]["extra_fields"]["sgmag1"]],
            ["srmag1", detections[0]["extra_fields"]["srmag1"]],
            ["distpsnr1", detections[0]["extra_fields"]["distpsnr1"]],
        ],
        columns=["name", "value"],
    ).fillna(value=np.nan)

    astro_object = AstroObject(
        detections=aid_detections, forced_photometry=aid_forced, metadata=metadata
    )
    return astro_object


def parse_scribe_payload(
    astro_objects: List[AstroObject], features_version, features_group
):
    """Create the json with the messages for the scribe producer from the
    features dataframe. It adds the fid and correct the name.

    :param astro_objects: a list of AstroObjects with computed features inside.
    :param features_version: a string with the features version used
    :return: a list of json with Alerce Scribe commands
    """

    def get_fid(band: str):
        """
        Parses the number used to reference the fid in the ztf alerts
        to the string value corresponding
        """
        fid_map = {"g": 1, "r": 2, "g,r": 12}
        if band in fid_map:
            return fid_map[band]
        return 0

    # features = features.replace({np.nan: None, np.inf: None, -np.inf: None})
    upsert_features_commands_list = []
    update_object_command_list = []

    for astro_object in astro_objects:
        # for upserting features
        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        ao_features["fid"] = ao_features["fid"].apply(get_fid)
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
        oid = query_ao_table(astro_object.metadata, "oid")

        features_list = ao_features.to_dict("records")

        upsert_features_command = {
            "collection": "object",
            "type": "update_features",
            "criteria": {"_id": oid},
            "data": {
                "features_version": features_version,
                "features_group": features_group,
                "features": features_list,
            },
            "options": {"upsert": True},
        }
        upsert_features_commands_list.append(upsert_features_command)

        # for updating the object
        g_r_max = list(
            filter(lambda x: x["name"] == "g-r_max" and x["fid"] == 12, features_list)
        )
        g_r_max = g_r_max[0] if len(g_r_max) == 1 else None
        g_r_mean = list(
            filter(lambda x: x["name"] == "g-r_mean" and x["fid"] == 12, features_list)
        )
        g_r_mean = g_r_mean[0] if len(g_r_mean) == 1 else None

        if g_r_max and g_r_mean:
            update_object_command = {
                "collection": "object",
                "type": "update",
                "criteria": {"oid": oid},
                "data": {
                    "g_r_max_corr": g_r_max,
                    "g_r_mean_corr": g_r_mean,
                },
                "options": {},
            }
            update_object_command_list.append(update_object_command)

    return {
        "update_object": update_object_command_list,
        "upserting_features": upsert_features_commands_list,
    }


def parse_output(
    astro_objects: List[AstroObject], messages: List[Dict], candids: Dict
) -> list[dict]:
    """
    Parse output of the step. It uses the input data to extend the schema to
    add the features of each object, identified by its oid.
    astro_objects and messages must be in the same order

    :param astro_objects:
    :param messages:
    :param candids:
    :return: a list of dictionaries, each input object with its data and the
        features calculated.
    """

    output_messages = []
    for message, astro_object in zip(messages, astro_objects):
        oid = message["oid"]
        candid = candids[oid]

        ao_features = astro_object.features[["name", "fid", "value"]].copy()
        fid_map = {"g": "_1", "r": "_2", "g,r": "_12", None: ""}
        ao_features["name"] += ao_features["fid"].map(fid_map)
        ao_features = ao_features.sort_values("name")
        ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
        oid_ao = query_ao_table(astro_object.metadata, "oid")
        assert oid_ao == oid
        feature_names = [f.replace("-", "_") for f in ao_features["name"].values]

        features_for_oid = dict(
            zip(feature_names, ao_features["value"].astype(np.double))
        )
        out_message = {
            "oid": oid,
            "candid": candid,
            "detections": message["detections"],
            "non_detections": message["non_detections"],
            "xmatches": message["xmatches"],
            "features": features_for_oid,
        }
        output_messages.append(out_message)

    return output_messages


[
    {"name": "g-r_mean", "fid": 12, "value": -0.010186289276425953},
    {"name": "g-r_max", "fid": 12, "value": -0.08876323699951172},
    {"name": "W1-W2", "fid": 12, "value": "0.0"},
    {"name": "W2-W3", "fid": 12, "value": "0.0"},
    {"name": "W3-W4", "fid": 12, "value": "0.0"},
    {"name": "g-W1", "fid": 12, "value": "2.4591054683778353"},
    {"name": "r-W1", "fid": 12, "value": "2.4692917576542612"},
    {"name": "g-W2", "fid": 12, "value": "2.4591054683778353"},
    {"name": "r-W2", "fid": 12, "value": "2.4692917576542612"},
    {"name": "g-W3", "fid": 12, "value": "2.4591054683778353"},
    {"name": "r-W3", "fid": 12, "value": "2.4692917576542612"},
    {"name": "g-W4", "fid": 12, "value": "2.4591054683778353"},
    {"name": "r-W4", "fid": 12, "value": "2.4692917576542612"},
    {"name": "sg_score", "fid": 0, "value": 0.3548886775970459},
    {"name": "dist_nr", "fid": 0, "value": 0.3251536786556244},
    {"name": "ps_g-r", "fid": 0, "value": 0.14206504821777344},
    {"name": "MHPS_ratio", "fid": 1, "value": 21.157636642456055},
    {"name": "MHPS_low", "fid": 1, "value": 2231.0380859375},
    {"name": "MHPS_high", "fid": 1, "value": 105.44835662841797},
    {"name": "MHPS_non_zero", "fid": 1, "value": 9.0},
    {"name": "MHPS_PN_flag", "fid": 1, "value": 0.0},
    {"name": "MHPS_ratio", "fid": 2, "value": 8.643694877624512},
    {"name": "MHPS_low", "fid": 2, "value": 120.1384048461914},
    {"name": "MHPS_high", "fid": 2, "value": 13.898963928222656},
    {"name": "MHPS_non_zero", "fid": 2, "value": 30.0},
    {"name": "MHPS_PN_flag", "fid": 2, "value": 0.0},
    {"name": "GP_DRW_sigma", "fid": 1, "value": 1989454.7950557263},
    {"name": "GP_DRW_tau", "fid": 1, "value": 0.0024787521766663585},
    {"name": "GP_DRW_sigma", "fid": 2, "value": 1398277.765107033},
    {"name": "GP_DRW_tau", "fid": 2, "value": 0.0024787521766663585},
    {"name": "Multiband_period", "fid": 12, "value": 0.9659870087158232},
    {"name": "PPE", "fid": 12, "value": 0.0014025914207903556},
    {"name": "Period_band", "fid": 1, "value": 0.08064547389029565},
    {"name": "delta_period", "fid": 1, "value": 0.8853415348255276},
    {"name": "Period_band", "fid": 2, "value": 0.08007828086950539},
    {"name": "delta_period", "fid": 2, "value": 0.8859087278463178},
    {"name": "Power_rate_1_4", "fid": 12, "value": 0.16167280077934265},
    {"name": "Power_rate_1_3", "fid": 12, "value": 0.04847700521349907},
    {"name": "Power_rate_1_2", "fid": 12, "value": 0.07137070596218109},
    {"name": "Power_rate_2", "fid": 12, "value": 0.299659788608551},
    {"name": "Power_rate_3", "fid": 12, "value": 0.14928393065929413},
    {"name": "Power_rate_4", "fid": 12, "value": 0.2142982929944992},
    {"name": "Psi_CS", "fid": 1, "value": 0.1953134847979119},
    {"name": "Psi_eta", "fid": 1, "value": 2.159312184740055},
    {"name": "Psi_CS", "fid": 2, "value": 0.15569699113391167},
    {"name": "Psi_eta", "fid": 2, "value": 2.1809582204263354},
    {"name": "Harmonics_mse", "fid": 1, "value": 3.387653030374152},
    {"name": "Harmonics_chi", "fid": 1, "value": 33.765972942840214},
    {"name": "Harmonics_mag_1", "fid": 1, "value": 1.0022284699951352},
    {"name": "Harmonics_mag_2", "fid": 1, "value": 1.1146477234437941},
    {"name": "Harmonics_phase_2", "fid": 1, "value": 1.2425462854136164},
    {"name": "Harmonics_mag_3", "fid": 1, "value": 0.18613573656983887},
    {"name": "Harmonics_phase_3", "fid": 1, "value": 4.934868945119998},
    {"name": "Harmonics_mag_4", "fid": 1, "value": 0.8868749820686674},
    {"name": "Harmonics_phase_4", "fid": 1, "value": 2.671406898643497},
    {"name": "Harmonics_mag_5", "fid": 1, "value": 0.6306703043625401},
    {"name": "Harmonics_phase_5", "fid": 1, "value": 0.4091058226807549},
    {"name": "Harmonics_mag_6", "fid": 1, "value": 0.33044872267828107},
    {"name": "Harmonics_phase_6", "fid": 1, "value": 0.691351809357811},
    {"name": "Harmonics_mag_7", "fid": 1, "value": 0.7331583002781676},
    {"name": "Harmonics_phase_7", "fid": 1, "value": 1.9939922116256543},
    {"name": "Harmonics_mse", "fid": 2, "value": 3.414686845672582},
    {"name": "Harmonics_chi", "fid": 2, "value": 31.71690643531344},
    {"name": "Harmonics_mag_1", "fid": 2, "value": 1.242317118787142},
    {"name": "Harmonics_mag_2", "fid": 2, "value": 0.7327088993804755},
    {"name": "Harmonics_phase_2", "fid": 2, "value": 2.103708320624911},
    {"name": "Harmonics_mag_3", "fid": 2, "value": 0.6818077287193468},
    {"name": "Harmonics_phase_3", "fid": 2, "value": 6.098628326969303},
    {"name": "Harmonics_mag_4", "fid": 2, "value": 0.1556934839272827},
    {"name": "Harmonics_phase_4", "fid": 2, "value": 4.160507036398975},
    {"name": "Harmonics_mag_5", "fid": 2, "value": 0.4555941073728955},
    {"name": "Harmonics_phase_5", "fid": 2, "value": 2.097520816445977},
    {"name": "Harmonics_mag_6", "fid": 2, "value": 0.09111710478902256},
    {"name": "Harmonics_phase_6", "fid": 2, "value": 2.8529736076433556},
    {"name": "Harmonics_mag_7", "fid": 2, "value": 0.3493066734939889},
    {"name": "Harmonics_phase_7", "fid": 2, "value": 4.042603677059521},
    {"name": "Amplitude", "fid": 1, "value": 3101.3867141403543},
    {"name": "AndersonDarling", "fid": 1, "value": 0.9999993786822315},
    {"name": "Autocor_length", "fid": 1, "value": 1.0},
    {"name": "Beyond1Std", "fid": 1, "value": 0.2926829268292683},
    {"name": "Con", "fid": 1, "value": 0.0},
    {"name": "Eta_e", "fid": 1, "value": 0.9609730983688637},
    {"name": "Gskew", "fid": 1, "value": 578.489572448647},
    {"name": "MaxSlope", "fid": 1, "value": 3404.4503456684142},
    {"name": "Mean", "fid": 1, "value": 130.6308088234238},
    {"name": "Meanvariance", "fid": 1, "value": 10.797595767069229},
    {"name": "MedianAbsDev", "fid": 1, "value": 265.3108477600479},
    {"name": "MedianBRP", "fid": 1, "value": 0.5609756097560976},
    {"name": "PairSlopeTrend", "fid": 1, "value": -0.03333333333333333},
    {"name": "PercentAmplitude", "fid": 1, "value": 74.05729587394636},
    {"name": "Q31", "fid": 1, "value": 803.1554602019718},
    {"name": "Rcs", "fid": 1, "value": 0.18450375198511548},
    {"name": "Skew", "fid": 1, "value": 0.06844263358466116},
    {"name": "SmallKurtosis", "fid": 1, "value": 1.1867426189507824},
    {"name": "Std", "fid": 1, "value": 1410.4986684006303},
    {"name": "StetsonK", "fid": 1, "value": 0.47658637120732567},
    {"name": "Pvar", "fid": 1, "value": 1.0},
    {"name": "ExcessVar", "fid": 1, "value": 116.58353486098531},
    {"name": "SF_ML_amplitude", "fid": 1, "value": 15.0},
    {"name": "SF_ML_gamma", "fid": 1, "value": 0.007870252669556047},
    {"name": "IAR_phi", "fid": 1, "value": 0.29300893884955986},
    {"name": "LinearTrend", "fid": 1, "value": 1.1452247671222469},
    {"name": "Amplitude", "fid": 2, "value": 2615.855112281759},
    {"name": "AndersonDarling", "fid": 2, "value": 0.9999872523376002},
    {"name": "Autocor_length", "fid": 2, "value": 1.0},
    {"name": "Beyond1Std", "fid": 2, "value": 0.2962962962962963},
    {"name": "Con", "fid": 2, "value": 0.0},
    {"name": "Eta_e", "fid": 2, "value": 0.38309352524698476},
    {"name": "Gskew", "fid": 2, "value": -1715.2958190110965},
    {"name": "MaxSlope", "fid": 2, "value": 3154.16182454297},
    {"name": "Mean", "fid": 2, "value": -409.84565713031157},
    {"name": "Meanvariance", "fid": 2, "value": -2.885206127875466},
    {"name": "MedianAbsDev", "fid": 2, "value": 497.3123151762545},
    {"name": "MedianBRP", "fid": 2, "value": 0.5185185185185185},
    {"name": "PairSlopeTrend", "fid": 2, "value": 0.03333333333333333},
    {"name": "PercentAmplitude", "fid": 2, "value": -39.561439304329},
    {"name": "Q31", "fid": 2, "value": 1423.2769780163442},
    {"name": "Rcs", "fid": 2, "value": 0.10401978795186653},
    {"name": "Skew", "fid": 2, "value": -0.7793498203263672},
    {"name": "SmallKurtosis", "fid": 2, "value": 0.7576052063153758},
    {"name": "Std", "fid": 2, "value": 1182.489201435522},
    {"name": "StetsonK", "fid": 2, "value": 0.3516544236984472},
    {"name": "Pvar", "fid": 2, "value": 1.0},
    {"name": "ExcessVar", "fid": 2, "value": 8.323923550401151},
    {"name": "SF_ML_amplitude", "fid": 2, "value": 15.0},
    {"name": "SF_ML_gamma", "fid": 2, "value": -0.009853943481483388},
    {"name": "IAR_phi", "fid": 2, "value": 0.3287666466607363},
    {"name": "LinearTrend", "fid": 2, "value": 0.02670077509922052},
    {"name": "SPM_A", "fid": 1, "value": 3181.301252213403},
    {"name": "SPM_t0", "fid": 1, "value": 32.088215398741795},
    {"name": "SPM_gamma", "fid": 1, "value": 1.5619213758835997},
    {"name": "SPM_beta", "fid": 1, "value": 0.7160749747773597},
    {"name": "SPM_tau_rise", "fid": 1, "value": 7.424809190460223},
    {"name": "SPM_tau_fall", "fid": 1, "value": 180.0},
    {"name": "SPM_A", "fid": 2, "value": 4640.236648725893},
    {"name": "SPM_t0", "fid": 2, "value": 164.4121316854273},
    {"name": "SPM_gamma", "fid": 2, "value": 1.0138910587968013},
    {"name": "SPM_beta", "fid": 2, "value": 0.8004204781844635},
    {"name": "SPM_tau_rise", "fid": 2, "value": 2.3238544360447904},
    {"name": "SPM_tau_fall", "fid": 2, "value": 125.82849005494802},
    {"name": "SPM_chi", "fid": 1, "value": 23765.47394151319},
    {"name": "SPM_chi", "fid": 2, "value": 18268.08461329254},
    {"name": "TDE_decay", "fid": 1, "value": 0.0},
    {"name": "TDE_decay_chi", "fid": 1, "value": None},
    {"name": "TDE_decay", "fid": 2, "value": 1.1036857831771842},
    {"name": "TDE_decay_chi", "fid": 2, "value": 124.37409129772661},
    {"name": "fleet_a", "fid": 1, "value": 0.6},
    {"name": "fleet_w", "fid": 1, "value": -0.000654772118500606},
    {"name": "fleet_chi", "fid": 1, "value": 15611.111477949611},
    {"name": "fleet_a", "fid": 2, "value": 0.6},
    {"name": "fleet_w", "fid": 2, "value": -0.0007666465265662293},
    {"name": "fleet_chi", "fid": 2, "value": 12770.955267505216},
    {"name": "color_variation", "fid": 12, "value": None},
    {"name": "positive_fraction", "fid": 1, "value": 0.5365853658536586},
    {"name": "n_forced_phot_band_before", "fid": 1, "value": 0.0},
    {"name": "dbrightness_first_det_band", "fid": 1, "value": None},
    {"name": "dbrightness_forced_phot_band", "fid": 1, "value": None},
    {"name": "last_brightness_before_band", "fid": 1, "value": None},
    {"name": "max_brightness_before_band", "fid": 1, "value": None},
    {"name": "median_brightness_before_band", "fid": 1, "value": None},
    {"name": "n_forced_phot_band_after", "fid": 1, "value": 55.0},
    {"name": "max_brightness_after_band", "fid": 1, "value": 3468.5894040604708},
    {"name": "median_brightness_after_band", "fid": 1, "value": 89.69497164742427},
    {"name": "positive_fraction", "fid": 2, "value": 0.42592592592592593},
    {"name": "n_forced_phot_band_before", "fid": 2, "value": 0.0},
    {"name": "dbrightness_first_det_band", "fid": 2, "value": None},
    {"name": "dbrightness_forced_phot_band", "fid": 2, "value": None},
    {"name": "last_brightness_before_band", "fid": 2, "value": None},
    {"name": "max_brightness_before_band", "fid": 2, "value": None},
    {"name": "median_brightness_before_band", "fid": 2, "value": None},
    {"name": "n_forced_phot_band_after", "fid": 2, "value": 40.0},
    {"name": "max_brightness_after_band", "fid": 2, "value": 3529.661663611551},
    {"name": "median_brightness_after_band", "fid": 2, "value": 97.84546687338607},
    {"name": "ulens_u0", "fid": 1, "value": 0.6},
    {"name": "ulens_tE", "fid": 1, "value": 20.0},
    {"name": "ulens_fs", "fid": 1, "value": 0.5},
    {"name": "ulens_chi", "fid": 1, "value": 225.63180341005017},
    {"name": "ulens_u0", "fid": 2, "value": 0.6},
    {"name": "ulens_tE", "fid": 2, "value": 20.0},
    {"name": "ulens_fs", "fid": 2, "value": 0.5},
    {"name": "ulens_chi", "fid": 2, "value": 307.5268031762322},
    {"name": "Timespan", "fid": 0, "value": 980.5895653760235},
    {"name": "Coordinate_x", "fid": 0, "value": 0.9879223606149642},
    {"name": "Coordinate_y", "fid": 0, "value": 0.12203027447686557},
    {"name": "Coordinate_z", "fid": 0, "value": 0.09548833179010625},
]
