import numpy as np


def multi_band(time, flux, band, preferred):
    for fid in preferred:
        if fid in band:
            ref = fid
            break
    else:
        raise ValueError(f"None of bands {preferred} found in the provided bands")

    tol = 1e-2
    mask = band == ref
    t0_bounds = [-50, np.max(time)]
    t0_guess = time[mask][np.argmax(flux[mask])] - 10
    # Order: t0, gamma, beta, t_rise, t_fall
    static_bounds = [t0_bounds, [1, 120], [0, 1], [1, 100], [1, 180]]
    static_guess = [
        np.clip(t0_guess, t0_bounds[0] + tol, t0_bounds[1] - tol),
        14,
        0.5,
        7,
        28,
    ]

    guess, bounds = [], []
    for fid in np.unique(band):
        mask = band == fid

        fmax = np.max(flux[mask])
        ampl_bounds = [np.abs(fmax) / 10.0, np.abs(fmax) * 10.0]
        ampl_guess = np.clip(1.2 * fmax, ampl_bounds[0] * 1.1, ampl_bounds[1] * 0.9)

        bounds += [ampl_bounds] + static_bounds
        guess += [ampl_guess] + static_guess

    return np.array(guess, dtype=np.float32), np.array(bounds, dtype=np.float32)


def single_band(time, flux, alt=False, bug=False):
    imax = np.argmax(flux)

    fmax = flux[imax]
    # order for bounds/guess: amplitude, t0, gamma, beta, t_rise, t_fall (lower first, upper second)
    if alt:
        bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 70, 100, 1, 100, 100]]
        guess = np.clip(
            [1.2 * fmax, time[imax] * 2 / 3, time[imax], 0.5, time[imax] / 2, 50],
            *bounds,
        )
    else:
        bounds = [[fmax / 3, -50, 1, 0, 1, 1], [fmax * 3, 50, 100, 1, 100, 100]]
        ampl = 3 * fmax if bug else 1.2 * fmax
        guess = np.clip([ampl, -5, np.max(time), 0.5, time[imax] / 2, 40], *bounds)

    return guess.astype(np.float32), np.array(bounds, dtype=np.float32)
