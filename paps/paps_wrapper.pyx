import cython

import numpy as np
cimport numpy as np

from libcpp cimport bool


cdef extern from "functions.h":
  double _conv (int N, double* a,double* time1,double* b1,double* b2,double* mask,double c1,double c2,long d,double dtmin)
  void _statistics(int N, double *flux, double* mag, double *magerr, double* time, double *mask,
                   double mag0, double t1,double t2, double epsilon, double dt, double mean,
                   double *Ik2_low_freq,double *Ik2_high_freq,int *nonzero,int *PN_flag);
  long _sigma_clip(double *flux, double *flag, double mean, double var, double *new_mean, double *new_var, long num);


def sigma_clip(np.ndarray[double, ndim=1, mode="c"] flux):

               N = flux.shape[0]
               cdef double mean = np.mean(flux)
               cdef double var = np.var(flux)
               cdef double new_mean
               cdef double new_var
               cdef np.ndarray[double, ndim=1] flag = np.zeros(N,dtype=np.float64)

               new_N = _sigma_clip(<double *> &flux[0], <double *> &flag[0], mean, var, &new_mean, &new_var, N)
               return flag < 0


def conv(np.ndarray[double, ndim=1, mode="c"] a,
         np.ndarray[double, ndim=1, mode="c"] time1,
         np.ndarray[double, ndim=1, mode="c"] mask,
         double c1,
         double c2,
         int d,
         double dtmin):

         N = a.shape[0]
         cdef np.ndarray[double, ndim=1] b1 = np.zeros(N,dtype=np.float64)
         cdef np.ndarray[double, ndim=1] b2 = np.zeros(N,dtype=np.float64)

         value = _conv(N,<double *> &a[0],<double *> &time1[0],<double *> &b1[0],<double *> &b2[0],<double *> &mask[0],c1,c2,d,dtmin)
         return value,b1,b2


def statistics(np.ndarray[double, ndim=1, mode="c"] mag,
  np.ndarray[double, ndim=1, mode="c"] magerr,
  np.ndarray[double, ndim=1, mode="c"] time,
  double t1,
  double t2,
  double mag0 = 19.0,
  double epsilon = 1.0,
  double dt = 5.0,
  double threshold_sigma=0.003):

  cdef int N = mag.shape[0]
  cdef np.ndarray[double, ndim=1] flux = np.power(10, (-mag + mag0)*0.4)
  cdef double mean = np.mean(flux)
  cdef double var = np.var(flux)
  cdef double Ik2_low_freq
  cdef double Ik2_high_freq
  cdef int nonzero
  cdef int PN_flag


  cdef np.ndarray[double,ndim=1] mask = np.ones(N,dtype=np.float64)

  flux = (flux - mean)/np.sqrt(var)

  _statistics(N,<double*> &flux[0],<double*> &mag[0],<double*> &magerr[0], <double*> &time[0], <double*> &mask[0],
            mag0, t1, t2, epsilon, dt, mean,
            &Ik2_low_freq, &Ik2_high_freq,&nonzero,&PN_flag)
  return (Ik2_low_freq/Ik2_high_freq, Ik2_low_freq, Ik2_high_freq,nonzero,PN_flag)
