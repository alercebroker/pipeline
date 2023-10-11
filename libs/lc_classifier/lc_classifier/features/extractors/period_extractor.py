from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
import logging
from P4J import MultiBandPeriodogram
from typing import List, Optional, Tuple, Dict


class PeriodExtractor(FeatureExtractor):
    def __init__(
            self,
            bands: List[str],
            smallest_period: float,
            largest_period: float,
            trim_lightcurve_to_n_days: Optional[float],
            min_length: int,
            use_forced_photo: bool
    ):
        self.version = '1.0.0'
        self.bands = bands

        self.periodogram_computer = MultiBandPeriodogram(method='MHAOV')
        self.smallest_period = smallest_period
        self.largest_period = largest_period
        self.trim_lightcurve_to_n_days = trim_lightcurve_to_n_days
        self.min_length = min_length
        self.use_forced_photo = use_forced_photo

    def _trim_lightcurve(self, observations: pd.DataFrame):
        if self.trim_lightcurve_to_n_days is None:
            return observations

        times = observations['mjd'].values

        # indexes of the best subsequence so far
        best_starting = 0
        best_ending = 0  # index of the last obs (included)

        # subsequence being examined
        # invariant: subsequence timespan under max allowed
        starting = 0
        ending = 0  # included

        while True:
            # subsequence len
            current_n = ending - starting + 1

            # best subsequence len
            len_best_subsequence = best_ending - best_starting + 1

            if current_n > len_best_subsequence:
                best_starting = starting
                best_ending = ending

            current_timespan = times[ending] - times[starting]
            if current_timespan < self.trim_lightcurve_to_n_days:
                # try to extend the subsequence
                ending += 1

                # nothing else to do
                if ending >= len(times):
                    break

                # restore invariant
                current_timespan = times[ending] - times[starting]
                while current_timespan > self.trim_lightcurve_to_n_days:
                    starting += 1
                    current_timespan = times[ending] - times[starting]
            else:
                starting += 1

        return observations.iloc[best_starting:(best_ending + 1)]

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if self.use_forced_photo:
            if astro_object.forced_photometry is not None:
                observations = pd.concat([
                    observations,
                    astro_object.forced_photometry], axis=0)
        observations = observations[observations['brightness'].notna()]
        observations = self._trim_lightcurve(observations)
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = self.get_observations(astro_object)

        period_candidate = np.nan
        significance = np.nan

        band_periods = dict()
        for band in self.bands:
            band_periods[band] = (np.nan, np.nan)

        if len(detections) < self.min_length:
            # don't compute the period if the lightcurve
            # is too short
            self._add_period_to_astroobject(
                astro_object, period_candidate, significance, band_periods)
            return

        aid = astro_object.metadata[astro_object.metadata['field'] == 'aid']
        aid = aid['value'].values[0]
        self.periodogram_computer.set_data(
            mjds=detections['mjd'].values,
            mags=detections['brightness'].values,
            errs=detections['e_brightness'].values,
            fids=detections['fid'].values)

        try:
            self.periodogram_computer.optimal_frequency_grid_evaluation(
                smallest_period=self.smallest_period,
                largest_period=self.largest_period,
                shift=0.2
            )
            self.periodogram_computer.optimal_finetune_best_frequencies(
                times_finer=10.0, n_local_optima=10)

            best_freq, best_per = self.periodogram_computer.get_best_frequencies()
            if len(best_freq) == 0:
                logging.error(
                    f'[PeriodExtractor] best frequencies has len 0: '
                    f'aid {aid}')
                self._add_period_to_astroobject(
                    astro_object, period_candidate, significance, band_periods)
                return

        except TypeError as e:
            logging.error(
                f'TypeError exception in PeriodExtractor: '
                f'oid {aid}\n{e}')
            self._add_period_to_astroobject(
                astro_object, period_candidate, significance, band_periods)
            return

        freq, per = self.periodogram_computer.get_periodogram()
        period_candidate = 1.0 / best_freq[0]

        available_bands = np.unique(detections['fid'].values)
        for band in self.bands:
            if band not in available_bands:
                continue
            best_freq_band = self.periodogram_computer.get_best_frequency(band)

            # Getting best period
            best_period_band = 1.0 / best_freq_band

            # Calculating delta period
            delta_period_band = np.abs(period_candidate - best_period_band)
            band_periods[band] = (best_period_band, delta_period_band)

        # Significance estimation
        entropy_best_n = 100
        top_values = np.sort(per)[-entropy_best_n:]
        normalized_top_values = top_values + 1e-2
        normalized_top_values = normalized_top_values / np.sum(normalized_top_values)
        entropy = (-normalized_top_values * np.log(normalized_top_values)).sum()
        significance = 1 - entropy / np.log(entropy_best_n)
        self._add_period_to_astroobject(
            astro_object, period_candidate, significance, band_periods)

    def _add_period_to_astroobject(
            self,
            astro_object: AstroObject,
            period_candidate: float,
            significance: float,
            band_periods: Dict[str, Tuple[float, float]]
    ):
        features = []

        all_bands = ','.join(self.bands)

        features.append(('Multiband_period', period_candidate, all_bands))
        features.append(('PPE', significance, all_bands))

        for band in self.bands:
            period_band, delta_period_band = band_periods[band]

            features.append((f'Period_band_{band}', period_band, band))
            features.append((f'delta_period_{band}', delta_period_band, band))

        features_df = pd.DataFrame(
            data=features,
            columns=['name', 'value', 'fid']
        )

        sids = astro_object.detections['sid'].unique()
        sids = np.sort(sids)
        sid = ','.join(sids)

        features_df['sid'] = sid
        features_df['version'] = self.version

        astro_object.features = pd.concat(
            [astro_object.features, features_df],
            axis=0
        )
