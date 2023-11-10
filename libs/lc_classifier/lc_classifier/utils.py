import numba
import numpy as np
from lc_classifier.base import AstroObject
import matplotlib.pyplot as plt


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


def mag_to_flux(mag: np.ndarray):
    """Converts a list of magnitudes into flux."""
    return 10 ** (-(mag + 48.6) / 2.5 + 26.0)


def plot_astro_object(astro_object: AstroObject, unit: str, use_forced_phot: bool, period=None):
    detections = astro_object.detections
    detections = detections[detections['unit'] == unit]

    available_bands = np.unique(detections['fid'])
    available_bands = set(available_bands)

    if use_forced_phot:
        forced_phot = astro_object.forced_photometry
        forced_phot = forced_phot[forced_phot['unit'] == unit]

        forced_phot_bands = np.unique(forced_phot['fid'])
        forced_phot_bands = set(forced_phot_bands)

        available_bands = available_bands.union(forced_phot_bands)

    color_map = {'g': 'tab:green', 'r': "tab:red", 'i': "tab:purple", 'z': "tab:brown"}

    for band in available_bands:
        band_detections = detections[detections['fid'] == band]
        band_time = band_detections['mjd']
        if period is not None:
            band_time = band_time % period

        plt.errorbar(
            band_time,
            band_detections['brightness'],
            yerr=band_detections['e_brightness'],
            fmt='*',
            label=band,
            color=color_map[band]
        )

        if use_forced_phot:
            band_forced = forced_phot[forced_phot['fid'] == band]
            band_forced_time = band_forced['mjd']
            if period is not None:
                band_forced_time = band_forced_time % period

            plt.errorbar(
                band_forced_time,
                band_forced['brightness'],
                yerr=band_forced['e_brightness'],
                fmt='.',
                label=band+' forced',
                color=color_map[band]
            )

    aid = astro_object.metadata[astro_object.metadata['name'] == 'aid']['value'].values[0]
    plt.title(aid)
    plt.xlabel('Time [mjd]')
    plt.ylabel(f'Brightness [{unit}]')
    if unit == 'magnitude':
        plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
