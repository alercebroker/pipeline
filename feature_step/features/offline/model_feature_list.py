"""Authority: the exact 199 band-suffixed feature names the deployed BHRF model
consumes, pinned so the coverage check runs without the 1.72 GB model download.

Provenance: squidward 2.1.0 hierarchical_random_forest_model.pkl, sourced from the
md5-verified deployed artifact
    https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl
    size 1,720,755,396 bytes, md5 95e8e9f18fde62f22025e31a88ad81fa (Last-Modified 2 Jun 2025)
These are the POST-preprocess column names classify_batch selects via
`features[self.feature_list]` (see hierarchical_random_forest.py). The offline
`--smoke` path asserts a freshly loaded model's feature_list equals this list, so
drift is caught rather than silently trusted.
"""

MODEL_VERSION = "2.1.0"
MODEL_MD5 = "95e8e9f18fde62f22025e31a88ad81fa"
MODEL_SIZE_BYTES = 1_720_755_396

# 199 names, verbatim from loaded_data["feature_list"].
MODEL_FEATURE_LIST = [
    'Amplitude_1', 'Amplitude_2', 'AndersonDarling_1', 'AndersonDarling_2',
    'Autocor_length_1', 'Autocor_length_2', 'Beyond1Std_1', 'Beyond1Std_2',
    'Con_1', 'Con_2', 'Coordinate_x', 'Coordinate_y',
    'Coordinate_z', 'Eta_e_1', 'Eta_e_2', 'ExcessVar_1',
    'ExcessVar_2', 'GP_DRW_sigma_1', 'GP_DRW_sigma_2', 'GP_DRW_tau_1',
    'GP_DRW_tau_2', 'Gskew_1', 'Gskew_2', 'Harmonics_chi_1',
    'Harmonics_chi_2', 'Harmonics_mag_1_1', 'Harmonics_mag_1_2', 'Harmonics_mag_2_1',
    'Harmonics_mag_2_2', 'Harmonics_mag_3_1', 'Harmonics_mag_3_2', 'Harmonics_mag_4_1',
    'Harmonics_mag_4_2', 'Harmonics_mag_5_1', 'Harmonics_mag_5_2', 'Harmonics_mag_6_1',
    'Harmonics_mag_6_2', 'Harmonics_mag_7_1', 'Harmonics_mag_7_2', 'Harmonics_mse_1',
    'Harmonics_mse_2', 'Harmonics_phase_2_1', 'Harmonics_phase_2_2', 'Harmonics_phase_3_1',
    'Harmonics_phase_3_2', 'Harmonics_phase_4_1', 'Harmonics_phase_4_2', 'Harmonics_phase_5_1',
    'Harmonics_phase_5_2', 'Harmonics_phase_6_1', 'Harmonics_phase_6_2', 'Harmonics_phase_7_1',
    'Harmonics_phase_7_2', 'IAR_phi_1', 'IAR_phi_2', 'LinearTrend_1',
    'LinearTrend_2', 'MHPS_PN_flag_1', 'MHPS_PN_flag_2', 'MHPS_high_30_1',
    'MHPS_high_30_2', 'MHPS_high_1', 'MHPS_high_2', 'MHPS_low_365_1',
    'MHPS_low_365_2', 'MHPS_low_1', 'MHPS_low_2', 'MHPS_non_zero_1',
    'MHPS_non_zero_2', 'MHPS_ratio_365_30_1', 'MHPS_ratio_365_30_2', 'MHPS_ratio_1',
    'MHPS_ratio_2', 'MaxSlope_1', 'MaxSlope_2', 'Mean_1',
    'Mean_2', 'Meanvariance_1', 'Meanvariance_2', 'MedianAbsDev_1',
    'MedianAbsDev_2', 'MedianBRP_1', 'MedianBRP_2', 'Multiband_period_12',
    'PPE_12', 'PairSlopeTrend_1', 'PairSlopeTrend_2', 'PercentAmplitude_1',
    'PercentAmplitude_2', 'Period_band_1', 'Period_band_2', 'Power_rate_1_2_12',
    'Power_rate_1_3_12', 'Power_rate_1_4_12', 'Power_rate_2_12', 'Power_rate_3_12',
    'Power_rate_4_12', 'Psi_CS_1', 'Psi_CS_2', 'Psi_eta_1',
    'Psi_eta_2', 'Pvar_1', 'Pvar_2', 'Q31_1',
    'Q31_2', 'Rcs_1', 'Rcs_2', 'SF_ML_amplitude_1',
    'SF_ML_amplitude_2', 'SF_ML_gamma_1', 'SF_ML_gamma_2', 'SPM_A_1',
    'SPM_A_2', 'SPM_beta_1', 'SPM_beta_2', 'SPM_chi_1',
    'SPM_chi_2', 'SPM_gamma_1', 'SPM_gamma_2', 'SPM_t0_1',
    'SPM_t0_2', 'SPM_tau_fall_1', 'SPM_tau_fall_2', 'SPM_tau_rise_1',
    'SPM_tau_rise_2', 'Skew_1', 'Skew_2', 'SmallKurtosis_1',
    'SmallKurtosis_2', 'Std_1', 'Std_2', 'StetsonK_1',
    'StetsonK_2', 'TDE_decay_chi_1', 'TDE_decay_chi_2', 'TDE_decay_1',
    'TDE_decay_2', 'Timespan', 'W1_W2', 'W2_W3',
    'W3_W4', 'color_variation_12', 'dbrightness_first_det_band_1', 'dbrightness_first_det_band_2',
    'dbrightness_forced_phot_band_1', 'dbrightness_forced_phot_band_2', 'delta_period_1', 'delta_period_2',
    'distpsnr1', 'fleet_a_1', 'fleet_a_2', 'fleet_chi_1',
    'fleet_chi_2', 'fleet_w_1', 'fleet_w_2', 'g_W1',
    'g_W2', 'g_W3', 'g_W4', 'g_r_max_corr_12',
    'g_r_max_12', 'g_r_mean_corr_12', 'g_r_mean_12', 'last_brightness_before_band_1',
    'last_brightness_before_band_2', 'max_brightness_after_band_1', 'max_brightness_after_band_2', 'max_brightness_before_band_1',
    'max_brightness_before_band_2', 'mean_chinr_12', 'mean_distnr_12', 'mean_sharpnr_12',
    'median_brightness_after_band_1', 'median_brightness_after_band_2', 'median_brightness_before_band_1', 'median_brightness_before_band_2',
    'n_forced_phot_band_after_1', 'n_forced_phot_band_after_2', 'n_forced_phot_band_before_1', 'n_forced_phot_band_before_2',
    'positive_fraction_1', 'positive_fraction_2', 'ps_g_r', 'ps_i_z',
    'ps_r_i', 'r_W1', 'r_W2', 'r_W3',
    'r_W4', 'sgscore1', 'sigma_distnr_12', 'ulens_chi_1',
    'ulens_chi_2', 'ulens_fs_1', 'ulens_fs_2', 'ulens_tE_1',
    'ulens_tE_2', 'ulens_u0_1', 'ulens_u0_2',
]
