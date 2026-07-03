-- BHRF (Squidward 2.1.0) classifier + taxonomy seed for ZTF (tid=0).
-- Generated from feature_step/features/offline/classifier_taxonomy_lut.py
--   (single source of truth). Do not hand-edit; regenerate from the fixture.
-- Run with:  psql "<conn>" -f ztf_classifier_taxonomy_seed.sql
--   (needs INSERT privileges, not readonly_user).
--
-- RE-VERIFY next-free classifier_id against live before applying:
--   SELECT classifier_id, classifier_name, classifier_version, tid
--     FROM multisurvey_ztf.classifier ORDER BY classifier_id;

-- 1. classifier (5 rows: flat + top + 3 branches)
INSERT INTO multisurvey_ztf.classifier (classifier_id, classifier_name, classifier_version, tid) VALUES
(5, 'lc_classifier_BHRF_forced_phot', '2.1.0', 0),
(6, 'lc_classifier_BHRF_forced_phot_top', '2.1.0', 0),
(7, 'lc_classifier_BHRF_forced_phot_transient', '2.1.0', 0),
(8, 'lc_classifier_BHRF_forced_phot_stochastic', '2.1.0', 0),
(9, 'lc_classifier_BHRF_forced_phot_periodic', '2.1.0', 0)
ON CONFLICT (classifier_id) DO NOTHING;

-- 2. taxonomy (45 rows: 21 + 3 + 6 + 6 + 9; class_id per-classifier 0-indexed)
INSERT INTO multisurvey_ztf.taxonomy (class_id, class_name, "order", classifier_id) VALUES
(0, 'AGN', 0, 5),
(1, 'Blazar', 1, 5),
(2, 'CEP', 2, 5),
(3, 'CV/Nova', 3, 5),
(4, 'DSCT', 4, 5),
(5, 'EA', 5, 5),
(6, 'EB/EW', 6, 5),
(7, 'LPV', 7, 5),
(8, 'Microlensing', 8, 5),
(9, 'Periodic-Other', 9, 5),
(10, 'QSO', 10, 5),
(11, 'RRLab', 11, 5),
(12, 'RRLc', 12, 5),
(13, 'RSCVn', 13, 5),
(14, 'SESN', 14, 5),
(15, 'SLSN', 15, 5),
(16, 'SNII', 16, 5),
(17, 'SNIIn', 17, 5),
(18, 'SNIa', 18, 5),
(19, 'TDE', 19, 5),
(20, 'YSO', 20, 5),
(0, 'Periodic', 0, 6),
(1, 'Stochastic', 1, 6),
(2, 'Transient', 2, 6),
(0, 'SESN', 0, 7),
(1, 'SLSN', 1, 7),
(2, 'SNII', 2, 7),
(3, 'SNIIn', 3, 7),
(4, 'SNIa', 4, 7),
(5, 'TDE', 5, 7),
(0, 'AGN', 0, 8),
(1, 'Blazar', 1, 8),
(2, 'CV/Nova', 2, 8),
(3, 'Microlensing', 3, 8),
(4, 'QSO', 4, 8),
(5, 'YSO', 5, 8),
(0, 'CEP', 0, 9),
(1, 'DSCT', 1, 9),
(2, 'EA', 2, 9),
(3, 'EB/EW', 3, 9),
(4, 'LPV', 4, 9),
(5, 'Periodic-Other', 5, 9),
(6, 'RRLab', 6, 9),
(7, 'RRLc', 7, 9),
(8, 'RSCVn', 8, 9)
ON CONFLICT (class_id, classifier_id) DO NOTHING;
