#ifndef EXAMPLES_H
#define EXAMPLES_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


# define pi 3.1415
# define E 2.718281828

double _conv (int N, double *a,double *time1, double *b1,double *b2,double *mask,double c1,double c2,long d,double dtmin);

long _sigma_clip(double *flux, double *flag, double mean, double var, double *new_mean, double *new_var, long num);

void _statistics(int N, double *flux, double* mag, double *magerr, double* time, double *mask,
                 double mag0, double t1,double t2, double epsilon, double dt, double mean,
                 double *Ik2_low_freq,double *Ik2_high_freq,int *nonzero,int *PN_flag);
#endif
