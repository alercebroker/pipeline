#!/usr/bin/env python
""" Fast algorithm for spectral analysis of unevenly sampled data

The Lomb-Scargle method performs spectral analysis on unevenly sampled
data and is known to be a powerful way to find, and test the
significance of, weak periodic signals. The method has previously been
thought to be 'slow', requiring of order 10(2)N(2) operations to analyze
N data points. We show that Fast Fourier Transforms (FFTs) can be used
in a novel way to make the computation of order 10(2)N log N. Despite
its use of the FFT, the algorithm is in no way equivalent to
conventional FFT periodogram analysis.

Keywords:
  DATA SAMPLING, FAST FOURIER TRANSFORMATIONS,
  SPECTRUM ANALYSIS, SIGNAL  PROCESSING

Example:
  > import numpy
  > import lomb
  > x = numpy.arange(10)
  > y = numpy.sin(x)
  > fx,fy, nout, jmax, prob = lomb.fasper(x,y, 6., 6.)

Reference:
  Press, W. H. & Rybicki, G. B. 1989
  ApJ vol. 338, p. 277-280.
  Fast algorithm for spectral analysis of unevenly sampled data
  bib code: 1989ApJ...338..277P

"""
import numpy as np
import P4J


def fasper(x, y, err, ofac, hifac, MACC=4, fmin=0.0, fmax=5.0):
    """ function fasper
    Given abscissas x (which need not be equally spaced) and ordinates
    y, and given a desired oversampling factor ofac (a typical value
    being 4 or larger). this routine creates an array wk1 with a
    sequence of nout increasing frequencies (not angular frequencies)
    up to hifac times the "average" Nyquist frequency, and creates
    an array wk2 with the values of the Lomb normalized periodogram at
    those frequencies. The arrays x and y are not altered. This
    routine also returns jmax such that wk2(jmax) is the maximum
    element in wk2, and prob, an estimate of the significance of that
    maximum against the hypothesis of random noise. A small value of prob
    indicates that a significant periodic signal is present.

    Reference:
    Press, W. H. & Rybicki, G. B. 1989
    ApJ vol. 338, p. 277-280.
    Fast algorithm for spectral analysis of unevenly sampled data
    (1989ApJ...338..277P)

    Arguments:
      X   : Abscissas array, (e.g. an array of times).
      Y   : Ordinates array, (e.g. corresponding counts).
      Ofac : Oversampling factor.
      Hifac : Hifac * "average" Nyquist frequency = highest frequency
           for which values of the Lomb normalized periodogram will
           be calculated.

    Returns:
      Wk1 : An array of Lomb periodogram frequencies.
      Wk2 : An array of corresponding values of the Lomb periodogram.
      Nout : Wk1 & Wk2 dimensions (number of calculated frequencies)
      Jmax : The array index corresponding to the MAX( Wk2 ).
      Prob : False Alarm Probability of the largest Periodogram value
      MACC : Number of interpolation points per 1/4 cycle
            of highest frequency

    History:
    02/23/2009, v1.0, MF
      Translation of IDL code (orig. Numerical recipies)
    """

    my_per = P4J.periodogram(method='MHAOV')

    my_per.set_data(x, y, err)

    # Course Frequency Evaluation
    my_per.frequency_grid_evaluation(fmin=fmin, fmax=fmax, fresolution=1e-3)

    # Finetine with smaller frequency steps
    n_best = 100
    my_per.finetune_best_frequencies(fresolution=1e-4, n_local_optima=10)

    # wk1: frequency grid, wk2: value of the periodgram at particular frequency
    wk1, wk2 = my_per.get_periodogram()
    fbest, pbest = my_per.get_best_frequencies()
    period_candidate = 1.0 / fbest[0]

    # Significance estimation
    use_entropy = True
    if use_entropy:
        top_values = np.sort(wk2)[-n_best:]
        normalized_top_values = top_values + 1e-2
        normalized_top_values = normalized_top_values / np.sum(normalized_top_values)
        entropy = (-normalized_top_values * np.log(normalized_top_values)).sum()
        prob = 1 - entropy / np.log(n_best)
    else:
        top_values = np.sort(wk2)[-n_best:]
        print(top_values)
        mean_value = wk2.mean()
        exp_parameter = 1 / mean_value  # biased estimator for exponential distribution
        cdf = 1 - np.exp(-exp_parameter * wk2.max())
        prob = cdf

    return wk1, wk2, period_candidate, prob
