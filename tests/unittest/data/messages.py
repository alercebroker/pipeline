import numpy as np
LC_MESSAGE = [
        {
            'id': 0,
            'detections': [
                {
                "_id" : 0,
                "tid" : 0,
                "aid" : 0,
                "oid" : 0,
                "mjd" : 0,
                "fid" : 0,
                "ra" : 0,
                "e_ra" : 0,
                "dec" : 0,
                "e_dec" : 0,
                "mag" : 15,
                "e_mag" : 0.1,
                "mag_corr" : 0,
                "e_mag_corr" : 0.01,
                "isdiffpos" : False,
                "corrected" : True,
                "dubious" : False,
                "parent_candidate" : None,
                "has_stamp" : False,
                "step_id_corr" : 0,
                "extra_fields" : {
                    "distnr" : 0.5,
                    "distpsnr1" : 2,
                    "sgscore1" : 0.9,
                    "chinr" : 0.5,
                    "sharpnr" : -0.1,
                    "rfid": 0,
                    }
                },
                {
                "_id" : 1,
                "tid" : 0,
                "aid" : 0,
                "oid" : 0,
                "mjd" : 3,
                "fid" : 0,
                "ra" : 0,
                "e_ra" : 0,
                "dec" : 0,
                "e_dec" : 0,
                "mag" : 16,
                "e_mag" : 0.1,
                "mag_corr" : 0,
                "e_mag_corr" : 0.01,
                "isdiffpos" : False,
                "corrected" : True,
                "dubious" : False,
                "parent_candidate" : None,
                "has_stamp" : False,
                "step_id_corr" : 0,
                "extra_fields" : {
                    "distnr" : 0.5,
                    "distpsnr1" : 2,
                    "sgscore1" : 0.9,
                    "chinr" : 0.5,
                    "sharpnr" : -0.1,
                    "rfid": 0,
                    }
                }
            ],
            'non_detections': [
                {
                "_id" : 3,
                "aid" : 0,
                "tid" : 0,
                "oid" : 0,
                "mjd" : 0.3,
                "fid" : 0,
                "diffmaglim" : 0.5,
                },
                {
                "_id" : 4,
                "aid" : 0,
                "tid" : 0,
                "oid" : 0,
                "mjd" : 1.5,
                "fid" : 0,
                "diffmaglim" : 0.3,
                },
                ]
        },
        {
            'id': 1,
            'detections': [

                ],
            'non_detections': [

                ]
        }
    ]

MAGSTATS_RESULT = [{'oid': 0.0, 'fid': 0.0, 'corrected': True, 'nearZTF': True, 'nearPS1': False, 'stellarZTF': True, 'stellarPS1': True, 'stellar': True, 'ndet': 2, 'ndubious': 0, 'nrfid': 1, 'magpsf_mean': 15.5, 'magpsf_median': 15.5, 'magpsf_max': 16.0, 'magpsf_min': 15.0, 'sigmapsf': 0.7071067811865476, 'magpsf_first': 15.0, 'sigmapsf_first': 0.1, 'magpsf_last': 16.0, 'first_mjd': 0.0, 'last_mjd': 3.0, 'saturation_rate': 0.0, 'step_id_corr': 'test', 'new': True, 'close_nondet': False, 'dmdt_first': np.nan, 'dm_first': np.nan, 'sigmadm_first': np.nan, 'dt_first': np.nan}]
