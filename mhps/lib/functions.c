#include "functions.h"

float _conv (int N, float *a,float *time1,float *b1,float *b2,float *mask,float c1,float c2,long d,float dtmin){

  int ci,cj,ck,csize,k,imax1,kmax;
  float norm1,norm2,normf,num[N],t,dt1,min,max;


  for (ci=0;ci<d;ci++){
    b1[ci]=0.0;
    b2[ci]=0.0;
    num[ci]=0.0;
  }

  norm1=0.0;
  norm2=0.0;

  min=-6.0*c1;
  max=6.0*c1;
  if (max<6.0*dtmin){
    min=-6.0*dtmin;
    max=6.0*dtmin;
  }
  dt1=(max-min)/1000.0;

  norm1=0.0;
  norm2=0.0;
  for (t=min; t<=max; t=t+dt1){
    norm1+=pow(E,-t*t/(2*c1*c1));
    norm2+=pow(E,-t*t/(2*c2*c2));
  }
  norm1=norm1*dt1;
  norm2=norm2*dt1;

  normf=0.0;
  for (t=min; t<=max; t=t+dt1){
    normf+=pow(pow(E,-t*t/(2*c1*c1))/norm1-pow(E,-t*t/(2*c2*c2))/norm2,2.0);
  }
  normf=normf*dt1;


  kmax=0;
  for (ci=0;ci<d;ci++){
    k=0;
    for (cj=0;cj<d;cj++){
      dt1=time1[cj]-time1[ci];
      if (dt1>min && dt1<max &&mask[cj]>0.){
          b1[ci]+=a[cj]*pow(E,-dt1*dt1/(2*c1*c1))/norm1;
          b2[ci]+=a[cj]*pow(E,-dt1*dt1/(2*c2*c2))/norm2;
          if (dt1>=min/3.0*2.0 && dt1<=max/3.0*2.0){
              k++;
          }

      }
    }
    b1[ci]*=mask[ci];
    b2[ci]*=mask[ci];


      if (k<=3){
      b1[ci]=-1.0;
      b2[ci]=-1.0;
    }
  }
  return (normf);
}


void _statistics(int N, float *flux, float* mag, float *magerr, float* time, float *mask,
                 float mag0, float t1,float t2, float epsilon, float dt, float mean,
                 float *Ik2_low_freq,float *Ik2_high_freq,int *nonzero,int *PN_flag){
  float rateG1[N],rateG2[N],maskG2[N],maskG1[N];
  float kr,sigma1,sigma2,norm,PN, Ik, fluxerr,Ik2;
  int imax,k;

  kr=1.0/t1;
  sigma1=0.225/kr/sqrt(1.0+epsilon);
  sigma2=(1.0+epsilon)*sigma1;

  imax=N;
  norm=_conv(N,mask,time,maskG1,maskG2,mask,sigma1,sigma2,imax,dt);
  norm=_conv(N,flux,time,rateG1,rateG2,mask,sigma1,sigma2,imax,dt);

  k=0;
  Ik2=0.0;
  *nonzero=0;
  PN=0.0;

  while(k<imax){
      if (maskG1[k]>0.0){
          Ik=rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2+=Ik*Ik;
          (*nonzero)++;
          fluxerr=log(10.0)*pow(10.0,-0.4*(mag[k]-mag0))*0.4*magerr[k];
          PN+=fluxerr;
          }
      k++;
  }
  Ik2/=(float)*nonzero;
  Ik2/=norm;
  Ik2/=(mean*mean);
  PN/=(float)*nonzero;
  PN=PN*PN/(mean*mean);
  Ik2-=PN;
  *Ik2_low_freq=Ik2;

  kr=1.0/t2;
  sigma1=0.225/kr/sqrt(1.0+epsilon);
  sigma2=(1.0+epsilon)*sigma1;


  norm=_conv(N,flux,time,rateG1,rateG2,mask,sigma1,sigma2,imax,dt);
  norm=_conv(N,mask,time,maskG1,maskG2,mask,sigma1,sigma2,imax,dt);

  k=0;
  Ik2=0.0;
  *nonzero=0;
  while(k<imax){
      if (maskG1[k]>0.0){
          Ik=rateG1[k]/maskG1[k]-rateG2[k]/maskG2[k];
          Ik2+=Ik*Ik;
          (*nonzero)++;
      }
      k++;
  }
  Ik2/=(float)*nonzero;
  Ik2/=norm;
  Ik2/=(mean*mean);
  if(Ik2>PN){
      Ik2-=PN;
      *PN_flag=0;
  }
  else *PN_flag=1;
  *Ik2_high_freq=Ik2;


}


long _sigma_clip(float *flux, float *flag, float mean, float var, float *new_mean, float *new_var, long num){

  long i,new_num,new_new_num;
  float sigma,flux2;
  float new_mean_value, new_var_value, new_new_mean_value,new_sigma;

  new_num=0;
  flux2=new_mean_value=0.0;
  sigma=sqrt(var);

  for(i=0;i<num;i++){
    if(flux[i]>(mean+3.0*sigma) ||flux[i]<(mean-3.0*sigma))
      flag[i]=10.0;
  }
  for(i=0;i<num;i++){
    if (flag[i]<1.0){
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

  if((new_var_value-var)*(new_var_value-var)>(new_var_value/10.0)*(new_var_value/10.0)){
      new_sigma=sqrt(new_var_value);
      new_new_num=0;
      flux2=0.0;
      new_new_mean_value=0.0;

      for(i=0;i<num;i++){
	if(flux[i]>(new_mean_value+3.0*new_sigma) ||flux[i]<(new_mean_value-3.0*new_sigma))
	  flag[i]=10.0;
     }
      for(i=0;i<num;i++){
	if (flag[i]<1.0){
	  new_new_mean_value+=flux[i];
	  flux2+=flux[i]*flux[i];
	  new_new_num++;
	}
      }
      new_num=new_new_num;
      if (new_num>0){
	new_mean_value=new_new_mean_value/(float)new_num;
	flux2/=(float)new_num;
	new_var_value=flux2-new_mean_value*new_mean_value;
      }
    }
    *new_mean=new_mean_value;
    *new_var=new_var_value;


  return(new_num);
}
