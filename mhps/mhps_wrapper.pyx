import cython

import numpy as np
cimport numpy as np

from libcpp cimport bool


cdef extern from "functions.h":
    float _conv (
        int N, float* a, float* time1, float* b1, float* b2, 
        float* mask, float c1, float c2, long d, float dtmin);
    void _statistics(
        int N, float *flux, float* mag, float *magerr, float* time, float *mask,
        float mag0, float t1, float t2, float epsilon, float dt, float mean,
        float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag);
    void _flux_statistics(
        int N, float *flux, float* fluxerr, float* time, float *mask,
        float t1,float t2, float epsilon, float dt, float mean,
        float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag);
    long _sigma_clip(
        float *flux, float *flag, float mean, float var, float *new_mean, float *new_var, long num);


def sigma_clip(np.ndarray[float, ndim=1, mode="c"] flux):
    N = flux.shape[0]
    cdef float mean = np.mean(flux)
    cdef float var = np.var(flux)
    cdef float new_mean
    cdef float new_var
    cdef np.ndarray[float, ndim=1] flag = np.zeros(N, dtype=np.float32)

    new_N = _sigma_clip(<float *> &flux[0], <float *> &flag[0], mean, var, &new_mean, &new_var, N)
    return flag < 0


def conv(
    np.ndarray[float, ndim=1, mode="c"] a,
    np.ndarray[float, ndim=1, mode="c"] time1,
    np.ndarray[float, ndim=1, mode="c"] mask,
    float c1,
    float c2,
    int d,
    float dtmin):
    
        N = a.shape[0]
        cdef np.ndarray[float, ndim=1] b1 = np.zeros(N, dtype=np.float32)
        cdef np.ndarray[float, ndim=1] b2 = np.zeros(N, dtype=np.float32)

        value = _conv(
            N, <float *> &a[0], <float *> &time1[0], <float *> &b1[0], <float *> &b2[0], 
            <float *> &mask[0], c1, c2, d, dtmin)
        return value, b1, b2


def statistics(
    mag,
    magerr,
    time,
    float t1,
    float t2,
    float mag0 = 19.0,
    float epsilon = 1.0,
    float dt = 5.0,
    float threshold_sigma=0.003):

    mag = np.array(mag, dtype=np.float32)
    magerr = np.array(magerr, dtype=np.float32)
    time = np.array(time, dtype=np.float32)

    if len(mag) != len(magerr) or len(magerr) != len(time):
       raise ValueError("mag, magerr and time array should be the same size")

    try:
        return _statistics32(mag, magerr, time, t1, t2, mag0, epsilon, dt, threshold_sigma)
    except ZeroDivisionError:
        return np.nan, np.nan, np.nan, np.nan, np.nan 
    
    
def _statistics32(
    np.ndarray[float, ndim=1, mode="c"] mag,
    np.ndarray[float, ndim=1, mode="c"] magerr,
    np.ndarray[float, ndim=1, mode="c"] time,
    float t1,
    float t2,
    float mag0 = 19.0,
    float epsilon = 1.0,
    float dt = 5.0,
    float threshold_sigma=0.003):

    cdef int N = mag.shape[0]
    cdef np.ndarray[float, ndim=1] flux = np.power(10, (-mag + mag0)*0.4)
    cdef float mean = np.mean(flux)
    cdef float var = np.var(flux)
    cdef float Ik2_low_freq
    cdef float Ik2_high_freq
    cdef int nonzero
    cdef int PN_flag


    cdef np.ndarray[float,ndim=1] mask = np.ones(N,dtype=np.float32)

    flux = flux - mean

    _statistics(
        N, <float*> &flux[0], <float*> &mag[0], <float*> &magerr[0], 
        <float*> &time[0], <float*> &mask[0], mag0, t1, t2, 
        epsilon, dt, mean, &Ik2_low_freq, &Ik2_high_freq, &nonzero, &PN_flag)
    return Ik2_low_freq/Ik2_high_freq, Ik2_low_freq, Ik2_high_freq, nonzero, PN_flag


def flux_statistics(
    flux,
    fluxerr,
    time,
    float t1,
    float t2,
    float epsilon = 1.0,
    float dt = 5.0):

    flux = np.array(flux, dtype=np.float32)
    fluxerr = np.array(fluxerr, dtype=np.float32)
    time = np.array(time, dtype=np.float32)

    if len(flux) != len(fluxerr) or len(fluxerr) != len(time):
       raise ValueError("flux, fluxerr and time array should be the same size")

    try:
        return _flux_statistics32(flux, fluxerr, time, t1, t2, epsilon, dt)
    except ZeroDivisionError:
        return np.nan, np.nan, np.nan, np.nan, np.nan 
    
    
def _flux_statistics32(
    np.ndarray[float, ndim=1, mode="c"] flux,
    np.ndarray[float, ndim=1, mode="c"] fluxerr,
    np.ndarray[float, ndim=1, mode="c"] time,
    float t1,
    float t2,
    float epsilon = 1.0,
    float dt = 5.0):

    cdef int N = flux.shape[0]
    cdef float mean = np.mean(flux)
    cdef float var = np.var(flux)
    cdef float Ik2_low_freq
    cdef float Ik2_high_freq
    cdef int nonzero
    cdef int PN_flag


    cdef np.ndarray[float, ndim=1] mask = np.ones(N, dtype=np.float32)

    flux = flux - mean

    _flux_statistics(
        N, <float*> &flux[0], <float*> &fluxerr[0], <float*> &time[0], 
        <float*> &mask[0], t1, t2, epsilon, dt, mean, 
        &Ik2_low_freq, &Ik2_high_freq, &nonzero, &PN_flag)

    return Ik2_low_freq/Ik2_high_freq, Ik2_low_freq, Ik2_high_freq, nonzero, PN_flag
