#include "functions.h"


float _conv(int N, float *a, float *time1, float *b1, float *b2,
	    float *mask, float c1, float c2, long d, float dtmin){

  int ci, cj, k;
  float norm1, norm2, normf, t, dt1, min, max;

  for (ci=0; ci<d; ci++){
    b1[ci] = 0.0f;
    b2[ci] = 0.0f;
  }

  norm1 = 0.0f;
  norm2 = 0.0f;

  min = -6.0f*c1;
  max = 6.0f*c1;
  
  if (max < 6.0f*dtmin){
    min = -6.0f*dtmin;
    max = 6.0f*dtmin;
  }
  dt1 = (max-min)/1000.0f;

  norm1 = 0.0f;
  norm2 = 0.0f;
  for (t=min; t<=max; t=t+dt1){
    norm1 += powf(E, -t*t/(2*c1*c1));
    norm2 += powf(E, -t*t/(2*c2*c2));
  }
  norm1 = norm1*dt1;
  norm2 = norm2*dt1;

  normf = 0.0f;
  for (t=min; t<=max; t=t+dt1){
    normf += powf(
        powf(E, -t*t/(2*c1*c1))/norm1 
        - powf(E, -t*t/(2*c2*c2))/norm2, 
        2.0f);
  }
  normf = normf*dt1;

  for (ci=0; ci<d; ci++){
    k = 0;
    for (cj=0; cj<d; cj++){
      dt1 = time1[cj] - time1[ci];
      if (dt1>min && dt1<max && mask[cj]>0.0f){
          b1[ci] += a[cj]*powf(E, -dt1*dt1/(2*c1*c1))/norm1;
          b2[ci] += a[cj]*powf(E, -dt1*dt1/(2*c2*c2))/norm2;
          if (dt1>=min/3.0f*2.0f && dt1<=max/3.0f*2.0f){
              k++;
          }
      }
    }
    b1[ci] *= mask[ci];
    b2[ci] *= mask[ci];
    if (k<=3){
      b1[ci] = -1.0f;
      b2[ci] = -1.0f;
    }
  }
  return (normf);
}


void _statistics(int N, float *flux, float* mag, float *magerr, float* time, float *mask,
                 float mag0, float t1,float t2, float epsilon, float dt, float mean,
                 float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag){
  float rateG1[N], rateG2[N], maskG2[N], maskG1[N];
  float kr, sigma1, sigma2, norm, PN, Ik, fluxerr, Ik2;
  int imax, k;

  kr = 1.0f/t1;
  sigma1 = 0.225f/kr/sqrtf(1.0f+epsilon);
  sigma2 = (1.0f+epsilon)*sigma1;

  imax = N;
  norm = _conv(N, mask, time, maskG1, maskG2, mask, sigma1, sigma2, imax, dt);
  norm = _conv(N, flux, time, rateG1, rateG2, mask, sigma1, sigma2, imax, dt);

  k = 0;
  Ik2 = 0.0f;
  *nonzero = 0;
  PN = 0.0f;

  while (k<imax){
      if (maskG1[k] > 0.0f){
          Ik = rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2 += Ik*Ik;
          (*nonzero)++;
          fluxerr = logf(10.0f)*powf(10.0f, -0.4f*(mag[k]-mag0))*0.4f*magerr[k];
          PN += fluxerr;
          }
      k++;
  }
  Ik2 /= (float)*nonzero;
  Ik2 /= norm;
  Ik2 /= (mean*mean);
  PN /= (float)*nonzero;
  PN = PN*PN/(mean*mean);
  Ik2 -= PN;
  *Ik2_low_freq = Ik2;

  kr = 1.0f/t2;
  sigma1 = 0.225f/kr/sqrtf(1.0f+epsilon);
  sigma2 = (1.0f+epsilon)*sigma1;


  norm = _conv(N, flux, time, rateG1, rateG2, mask, sigma1, sigma2, imax, dt);
  norm = _conv(N, mask, time, maskG1, maskG2, mask, sigma1, sigma2, imax, dt);

  k = 0;
  Ik2 = 0.0f;
  *nonzero = 0;
  while (k < imax){
      if (maskG1[k] > 0.0f){
          Ik = rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2 += Ik*Ik;
          (*nonzero)++;
      }
      k++;
  }
  Ik2 /= (float)*nonzero;
  Ik2 /= norm;
  Ik2 /= (mean*mean);
  if (Ik2>PN){
      Ik2 -= PN;
      *PN_flag = 0;
  }
  else *PN_flag = 1;
  *Ik2_high_freq = Ik2;
}


void _flux_statistics(
    int N, float *flux, float* fluxerr, float* time, float *mask,
    float t1,float t2, float epsilon, float dt, float mean,
    float *Ik2_low_freq, float *Ik2_high_freq, int *nonzero, int *PN_flag){

  float rateG1[N], rateG2[N], maskG2[N], maskG1[N];
  float kr, sigma1, sigma2, norm, PN, Ik, Ik2;
  int imax, k;

  kr = 1.0f/t1;
  sigma1 = 0.225f/kr/sqrtf(1.0f+epsilon);
  sigma2 = (1.0+epsilon)*sigma1;

  imax = N;
  norm = _conv(N, mask, time, maskG1, maskG2, mask, sigma1, sigma2, imax, dt);
  norm = _conv(N, flux, time, rateG1, rateG2, mask, sigma1, sigma2, imax, dt);

  k = 0;
  Ik2 = 0.0f;
  *nonzero = 0;
  PN = 0.0f;

  while (k < imax){
      if (maskG1[k] > 0.0f){
          Ik = rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2 += Ik*Ik;
          (*nonzero)++;
          PN += fluxerr[k];
          }
      k++;
  }
  Ik2 /= (float)*nonzero;
  Ik2 /= norm;
  Ik2 /= (mean*mean);
  PN /= (float)*nonzero;
  PN = PN*PN/(mean*mean);
  Ik2 -= PN;
  *Ik2_low_freq = Ik2;

  kr = 1.0f/t2;
  sigma1 = 0.225f/kr/sqrtf(1.0f+epsilon);
  sigma2 = (1.0f+epsilon)*sigma1;


  norm = _conv(N, flux, time, rateG1, rateG2, mask, sigma1, sigma2, imax, dt);
  norm = _conv(N, mask, time, maskG1, maskG2, mask, sigma1, sigma2, imax, dt);

  k = 0;
  Ik2 = 0.0f;
  *nonzero = 0;
  while (k < imax){
      if (maskG1[k] > 0.0f){
          Ik = rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2 += Ik*Ik;
          (*nonzero)++;
      }
      k++;
  }
  Ik2 /= (float)*nonzero;
  Ik2 /= norm;
  Ik2 /= (mean*mean);
  if (Ik2>PN){
      Ik2 -= PN;
      *PN_flag = 0;
  }
  else *PN_flag = 1;
  *Ik2_high_freq = Ik2;
}


long _sigma_clip(float *flux, float *flag, float mean, float var, float *new_mean, float *new_var, long num){

  long i, new_num, new_new_num;
  float sigma, flux2;
  float new_mean_value, new_var_value, new_new_mean_value, new_sigma;

  new_num = 0;
  flux2 = new_mean_value = 0.0f;
  sigma = sqrtf(var);

  for(i=0;i<num;i++){
    if(flux[i]>(mean+3.0f*sigma) ||flux[i]<(mean-3.0f*sigma))
      flag[i]=10.0f;
  }
  for(i=0;i<num;i++){
    if (flag[i]<1.0f){
      new_mean_value+=flux[i];
      flux2+=flux[i]*flux[i];
      new_num++;
    }
  }
  if (new_num>0){
    new_mean_value/=(float)new_num;
    flux2/=(float)new_num;
    new_var_value=flux2-new_mean_value*new_mean_value;
  }

  if((new_var_value-var)*(new_var_value-var)>(new_var_value/10.0f)*(new_var_value/10.0f)){
      new_sigma=sqrtf(new_var_value);
      new_new_num=0;
      flux2=0.0f;
      new_new_mean_value=0.0f;

      for(i=0; i<num; i++){
	    if(flux[i]>(new_mean_value+3.0f*new_sigma) ||flux[i]<(new_mean_value-3.0f*new_sigma))
	    flag[i]=10.0f;
     }
      for(i=0;i<num;i++){
	if (flag[i]<1.0f){
	  new_new_mean_value+=flux[i];
	  flux2+=flux[i]*flux[i];
	  new_new_num++;
	}
      }
      new_num=new_new_num;
      if (new_num>0){
	new_mean_value = new_new_mean_value/(float)new_num;
	flux2 /= (float)new_num;
	new_var_value = flux2-new_mean_value*new_mean_value;
      }
    }
    *new_mean = new_mean_value;
    *new_var = new_var_value;


  return(new_num);
}
