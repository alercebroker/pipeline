#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport cython
import numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from .utilities cimport weighted_mean

ctypedef float DTYPE_t
ctypedef int ITYPE_t
ctypedef double TIME_t   # double precision for time, phase folding and accumulators

cdef extern from "math.h" nogil:
    DTYPE_t sqrtf(DTYPE_t)
    DTYPE_t powf(DTYPE_t, DTYPE_t)
    DTYPE_t cosf(DTYPE_t)
    DTYPE_t sinf(DTYPE_t)
    TIME_t floor(TIME_t)   # double floor: phase reduction must happen in double


"""

Multiharmonic weighted AoV Periodogram
Ref: http://adsabs.harvard.edu/abs/1996ApJ...460L.107S

This implementation follows the c code found at
http://users.camk.edu.pl/alex/soft/aovgui.tgz

Please cite the paper above if using this code
"""

cdef DTYPE_t M_PI = 3.1415926535897

cdef DTYPE_t* allocate_and_verify(Py_ssize_t N):
    cdef DTYPE_t* array = <DTYPE_t*>PyMem_Malloc(N*sizeof(DTYPE_t))
    if not array:
        raise MemoryError()
    return array

cdef TIME_t* allocate_time(Py_ssize_t N):
    cdef TIME_t* array = <TIME_t*>PyMem_Malloc(N*sizeof(TIME_t))
    if not array:
        raise MemoryError()
    return array

cdef class MHAOV:
    cdef Py_ssize_t N
    cdef TIME_t* mjd
    cdef DTYPE_t* mag_minus_wmean
    cdef DTYPE_t* err
    cdef DTYPE_t* inv_err          # 1/err, precomputed (frequency-invariant)
    cdef DTYPE_t* mag_over_err     # (mag-wmean)/err, precomputed (frequency-invariant)
    cdef ITYPE_t Nharmonics
    cdef ITYPE_t mode
    cdef DTYPE_t d1, d2
    cdef public DTYPE_t wmean
    cdef public TIME_t wvar
    cdef DTYPE_t* zr 
    cdef DTYPE_t* zi
    cdef DTYPE_t* znr
    cdef DTYPE_t* zni
    cdef DTYPE_t* pr
    cdef DTYPE_t* pi
    cdef DTYPE_t* cfr
    cdef DTYPE_t* cfi

    def __init__(self, TIME_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err, ITYPE_t Nharmonics=1, ITYPE_t mode=1):
        cdef Py_ssize_t i
        if Nharmonics < 1:
            raise ValueError("Number of harmonics has to be greater or equal to 1")
        self.Nharmonics = Nharmonics
        self.N = mag.shape[0]
        self.mjd = allocate_time(self.N)
        self.mag_minus_wmean = allocate_and_verify(self.N)
        self.err = allocate_and_verify(self.N)
        self.inv_err = allocate_and_verify(self.N)
        self.mag_over_err = allocate_and_verify(self.N)
        self.zr = allocate_and_verify(self.N)
        self.zi = allocate_and_verify(self.N)
        self.znr = allocate_and_verify(self.N)
        self.zni = allocate_and_verify(self.N)
        self.pr = allocate_and_verify(self.N)
        self.pi = allocate_and_verify(self.N)
        self.cfr = allocate_and_verify(self.N)
        self.cfi = allocate_and_verify(self.N)
        
        self.wmean = weighted_mean(&mag[0], &err[0], self.N)

        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag_minus_wmean[i] = mag[i] - self.wmean
            self.err[i] = err[i]
            self.inv_err[i] = 1.0/err[i]
            self.mag_over_err[i] = (mag[i] - self.wmean)/err[i]

        self.d1 = Nharmonics*2.0
        self.d2 = self.N - Nharmonics*2 - 1
        self.wvar = 0.0
        for i in range(self.N):
            self.wvar += (<TIME_t>powf(self.mag_minus_wmean[i], 2.))/(<TIME_t>powf(self.err[i], 2.))
            
        self.mode = mode # 0: RAW, 1: F
        

    # Core single-frequency evaluation, shared by the scalar and batch entry
    # points. nogil so the batch loop can drop the GIL. The arithmetic is a
    # 1:1 transcription of the original eval_frequency body (max() replaced by
    # equivalent C branches so it can run nogil).
    cdef TIME_t _eval_one(self, TIME_t freq) noexcept nogil:
        cdef Py_ssize_t i, j
        # Cross-point reductions stay double (correctness: wvar - aov cancellation).
        cdef TIME_t sn, alr, ali, scr, sci
        cdef TIME_t aov=0.0
        cdef TIME_t arg, frac, denom
        # Per-point basis recurrence stays float32 (matches original precision, keeps
        # the hot update loop SIMD-vectorizable).
        cdef DTYPE_t argf, hargf, sr, si, tmp, alrf, alif
        cdef TIME_t two_pi = 2.0*M_PI

        for i in range(self.N):
            # Phase folding in double precision: mjd (~6e4) * freq is computed and
            # reduced mod 1 in float64, avoiding the ~0.1-cycle error float32 incurs.
            arg = self.mjd[i]*freq
            frac = arg - floor(arg)
            argf = <DTYPE_t>(two_pi*frac)                      # angle in [0, 2pi)

            # z = exp(j*arg), complex exp with f=freq eval at times mjd[i]
            self.zr[i] = cosf(argf)
            self.zi[i] = sinf(argf)

            # zn = 1, bias basis?
            self.znr[i] = 1.
            self.zni[i] = 0.

            # p = 1/err  (precomputed)
            self.pr[i] = self.inv_err[i]
            self.pi[i] = 0.

            # cf = (mag-wmean)*exp(j*n_harmonics*arg)/err.
            # For Nharmonics==1 (production) the harmonic basis equals z, so reuse
            # zr/zi instead of recomputing cosf/sinf (bit-identical, halves trig).
            if self.Nharmonics == 1:
                self.cfr[i] = self.mag_over_err[i]*self.zr[i]
                self.cfi[i] = self.mag_over_err[i]*self.zi[i]
            else:
                hargf = <DTYPE_t>(two_pi*frac*self.Nharmonics)
                self.cfr[i] = self.mag_over_err[i]*cosf(hargf)
                self.cfi[i] = self.mag_over_err[i]*sinf(hargf)
        for j in range(2*self.Nharmonics+1):
            sn = alr = ali = scr = sci = 0.0
            for i in range(self.N):
                # += |p|^2  (accumulated in double)
                sn += self.pr[i]**2 + self.pi[i]**2

                # al += z*p/err  (multiply by precomputed 1/err)
                alr += (self.zr[i]*self.pr[i] - self.zi[i]*self.pi[i])*self.inv_err[i]
                ali += (self.zr[i]*self.pi[i] + self.zi[i]*self.pr[i])*self.inv_err[i]

                # sc += conj(p)*cr
                scr += self.pr[i]*self.cfr[i] + self.pi[i]*self.cfi[i]
                sci += self.pr[i]*self.cfi[i] - self.pi[i]*self.cfr[i]
            if sn < 1e-9:
                sn = 1e-9

            # al = al/sn
            alr = alr/sn
            ali = ali/sn

            # aov += |sc|^2 / sn
            aov += (scr**2 + sci**2)/sn
            alrf = <DTYPE_t>alr
            alif = <DTYPE_t>ali
            for i in range(self.N):
                # s = al*zn
                sr = alrf*self.znr[i] - alif*self.zni[i]
                si = alrf*self.zni[i] + alif*self.znr[i]


                # updating p = p*z - s*conj(p)
                # tmp = re(p*z)-re(s*conj(p))
                tmp = self.pr[i]*self.zr[i] - self.pi[i]*self.zi[i] - sr*self.pr[i] - si*self.pi[i]
                # im(p) = im(p*z)-im(s*conj(p))
                self.pi[i] = self.pr[i]*self.zi[i] + self.pi[i]*self.zr[i] + sr*self.pi[i] - si*self.pr[i]
                self.pr[i] = tmp

                # updating zn = zn * z
                tmp = self.znr[i]*self.zr[i] - self.zni[i]*self.zi[i]
                self.zni[i] = self.zni[i]*self.zr[i] + self.znr[i]*self.zi[i]
                self.znr[i] = tmp
        if self.mode == 0:
            return aov
        else:  # mode == 1: F-statistic
            denom = self.wvar - aov
            if denom < 1e-9:
                denom = 1e-9
            return (self.d2/self.d1)*aov/denom

    def eval_frequency(self, TIME_t freq):
        return self._eval_one(freq)

    # Batch evaluation: the whole frequency grid is looped inside Cython with the
    # GIL released, instead of a Python list comprehension calling eval_frequency
    # once per frequency. Returns a float64 ndarray (the raw aov / F-statistic is
    # accumulated in double, so the output is kept in double too).
    def eval_frequencies(self, TIME_t [::1] freqs):
        cdef Py_ssize_t k, M = freqs.shape[0]
        cdef TIME_t [::1] out_view
        out = np.empty(M, dtype=np.float64)
        out_view = out
        with nogil:
            for k in range(M):
                out_view[k] = self._eval_one(freqs[k])
        return out

    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag_minus_wmean)
        PyMem_Free(self.err)
        PyMem_Free(self.inv_err)
        PyMem_Free(self.mag_over_err)
        PyMem_Free(self.zr)
        PyMem_Free(self.zi)
        PyMem_Free(self.znr)
        PyMem_Free(self.zni)
        PyMem_Free(self.pr)
        PyMem_Free(self.pi)
        PyMem_Free(self.cfr)
        PyMem_Free(self.cfi)



