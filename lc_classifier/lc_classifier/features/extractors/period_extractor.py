from ..core.base import FeatureExtractor, AstroObject
from lc_classifier.utils import is_sorted
import numpy as np
import pandas as pd
import logging
from P4J import MultiBandPeriodogram
from typing import List, Optional, Tuple, Dict


class PeriodExtractor(FeatureExtractor):
    def __init__(
        self,
        bands: List[str],
        unit: str,
        smallest_period: float,
        largest_period: float,
        trim_lightcurve_to_n_days: Optional[float],
        min_length: int,
        use_forced_photo: bool,
        return_power_rates: bool,
        shift: float = 0.1,
    ):
        self.version = "1.0.0"
        self.bands = bands
        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit
        self.periodogram_computer = MultiBandPeriodogram(method="MHAOV")
        self.smallest_period = smallest_period
        self.largest_period = largest_period
        self.trim_lightcurve_to_n_days = trim_lightcurve_to_n_days
        self.min_length = min_length
        self.shift = shift
        self.use_forced_photo = use_forced_photo
        self.return_power_rates = return_power_rates

        self.factors = [0.25, 1 / 3, 0.5, 2.0, 3.0, 4.0]
        pr_names = ["1_4", "1_3", "1_2", "2", "3", "4"]
        self.pr_names = ["Power_rate_" + n for n in pr_names]

    def _trim_lightcurve(self, observations: pd.DataFrame):
        if self.trim_lightcurve_to_n_days is None or len(observations) == 0:
            return observations

        times = observations["mjd"].values

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

        return observations.iloc[best_starting : (best_ending + 1)]

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if self.use_forced_photo:
            if astro_object.forced_photometry is not None:
                observations = pd.concat(
                    [observations, astro_object.forced_photometry], axis=0
                )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations.sort_values("mjd")
        observations = self._trim_lightcurve(observations)
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        period_candidate = np.nan
        significance = np.nan
        power_rates = None  # was successfully computed?

        band_periods = dict()
        useful_bands = []
        for band in self.bands:
            band_periods[band] = (np.nan, np.nan)
            n_obs_band = len(observations[observations["fid"] == band])
            if n_obs_band >= self.min_length:
                useful_bands.append(band)

        observations = observations[observations["fid"].isin(useful_bands)]

        if len(observations) < self.min_length:
            # don't compute the period if the lightcurve
            # is too short
            self._add_period_to_astroobject(
                astro_object, period_candidate, significance, band_periods, power_rates
            )
            return

        aid = astro_object.metadata[astro_object.metadata["name"] == "aid"]
        aid = aid["value"].values[0]
        self.periodogram_computer.set_data(
            mjds=observations["mjd"].values,
            mags=observations["brightness"].values,
            errs=observations["e_brightness"].values,
            fids=observations["fid"].values,
        )

        try:
            self.periodogram_computer.optimal_frequency_grid_evaluation(
                smallest_period=self.smallest_period,
                largest_period=self.largest_period,
                shift=self.shift,
            )
            self.periodogram_computer.optimal_finetune_best_frequencies(
                times_finer=10.0, n_local_optima=10
            )

            best_freq, best_per = self.periodogram_computer.get_best_frequencies()
            if len(best_freq) == 0:
                logging.error(
                    f"[PeriodExtractor] best frequencies has len 0: " f"aid {aid}"
                )
                self._add_period_to_astroobject(
                    astro_object,
                    period_candidate,
                    significance,
                    band_periods,
                    power_rates,
                )
                return

        except TypeError as e:
            logging.error(f"TypeError exception in PeriodExtractor: " f"oid {aid}\n{e}")
            self._add_period_to_astroobject(
                astro_object, period_candidate, significance, band_periods, power_rates
            )
            return

        freq, per = self.periodogram_computer.get_periodogram()
        if self.return_power_rates:
            power_rates = self.compute_power_rates(freq, per)

        period_candidate = 1.0 / best_freq[0]

        available_bands = np.unique(observations["fid"].values)
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
            astro_object, period_candidate, significance, band_periods, power_rates
        )

    def _add_period_to_astroobject(
        self,
        astro_object: AstroObject,
        period_candidate: float,
        significance: float,
        band_periods: Dict[str, Tuple[float, float]],
        power_rates: Optional[List[Tuple[str, float]]],
    ):
        features = []

        all_bands = ",".join(self.bands)

        features.append(("Multiband_period", period_candidate, all_bands))
        features.append(("PPE", significance, all_bands))

        for band in self.bands:
            period_band, delta_period_band = band_periods[band]

            features.append((f"Period_band", period_band, band))
            features.append((f"delta_period", delta_period_band, band))

        if self.return_power_rates:
            if power_rates is not None:
                for name, pr in power_rates:
                    features.append((name, pr, all_bands))
            else:
                for name in self.pr_names:
                    features.append((name, np.nan, all_bands))

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )

    def compute_power_rates(self, freq, per):
        if not is_sorted(freq):
            order = freq.argsort()
            freq = freq[order]
            per = per[order]

        rate_features = []
        for name, factor in zip(self.pr_names, self.factors):
            power_rate = self._get_power_ratio(freq, per, factor)
            rate_features.append((name, power_rate))
        return rate_features

    def _get_power_ratio(self, frequencies, periodogram, period_factor):
        max_index = periodogram.argmax()
        period = 1.0 / frequencies[max_index]
        period_power = periodogram[max_index]

        desired_period = period * period_factor
        desired_frequency = 1.0 / desired_period
        i = np.searchsorted(frequencies, desired_frequency)

        if i == 0:
            i = 0
        elif desired_frequency > frequencies[-1]:
            i = len(frequencies) - 1
        else:
            left = frequencies[i - 1]
            right = frequencies[i]
            mean = (left + right) / 2.0
            if desired_frequency > mean:
                i = i
            else:
                i = i - 1
        return periodogram[i] / period_power
