#ifndef EXAMPLES_H
#define EXAMPLES_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


# define pi 3.141592f
# define E 2.718281828f

float _conv(
    int N, float *a, float *time1, float *b1, float *b2,
    float *mask, float c1, float c2, long d, float dtmin);

long _sigma_clip(
    float *flux, float *flag, float mean, float var, 
    float *new_mean, float *new_var, long num);

void _statistics(
    int N, float *flux, float* mag, float *magerr, float* time, float *mask,
    float mag0, float t1, float t2, float epsilon, float dt, float mean,
    float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag);

void _flux_statistics(
    int N, float *flux, float* fluxerr, float* time, float *mask,
    float t1,float t2, float epsilon, float dt, float mean,
    float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag);
#endif
