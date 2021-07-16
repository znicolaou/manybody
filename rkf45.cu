//Zachary G. Nicolaou 7/5/2021
//Runge Kutta Feldberg 4/5 stepper on the GPU
#include "rkf45.h"

//Rkf45 coefficients
//We could make A, B, C, and K matrices rather than vectors, but we'd still need to calculate serially
//The cublas operations do not seem to improve runtime, since additional copies are needed
const double a1[1] = {1.0/4};
const double a2[2] = {3.0/32, 9.0/32};
const double a3[3] = {1932.0/2197, -7200.0/2197, 7296.0/2197};
const double a4[4] = {439.0/216, -8.0, 3680.0/513, -845.0/4104};
const double a5[5] = {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40};
const double b1[6] = {16.0/135, 0.0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
const double b2[6] = {25.0/216, 0.0, 1408.0/2565, 2197.0/4104, -1.0/5, 0.0};
const double c[6] = {0.0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2};

static double *y, *ytemp, *yerr, *k1, *k2, *k3, *k4, *k5, *k6, *normd;
static int N;
static void (*dydt)(double, double*, double*) = NULL;
static double atl, rtl;

//Steps for the RK stepper
__global__ void step1 (double* y, double* ytemp, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i];
  }
}
__global__ void step2 (double* y, double* k1, double* ytemp, const double a10, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+a10*k1[i];
  }
}
__global__ void step3 (double* y, double* k1, double* k2, double* ytemp, const double a20, const double a21, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+a20*k1[i]+a21*k2[i];
  }
}
__global__ void step4 (double* y, double* k1, double* k2, double* k3, double* ytemp, const double a30, const double a31, const double a32, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+a30*k1[i]+a31*k2[i]+a32*k3[i];
  }
}
__global__ void step5 (double* y, double* k1, double* k2, double* k3, double* k4, double* ytemp, const double a40, const double a41, const double a42, const double a43, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+a40*k1[i]+a41*k2[i]+a42*k3[i]+a43*k4[i];
  }
}
__global__ void step6 (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* ytemp, const double a50, const double a51, const double a52, const double a53, const double a54, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    ytemp[i]=y[i]+a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i];
  }
}
//Error estimate for the RK stepper
__global__ void error (double *norm, double *y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* yerr, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55, const double atl, const double rtl, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N) {
    yerr[i]=(a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i]+a55*k6[i])/(atl+rtl*y[i]);
    atomicAdd(norm,yerr[i]*yerr[i]);
  }
}

//Accept the RK step
__global__ void accept (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55, const int N) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    y[i]=y[i]+a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i]+a55*k6[i];
  }
}
//Attempt a RK step
void rkf45_step (double *t, double *h){
  double norm=0;

  //Calculate the intermediate steps and error estimates using the CUDA kernels
  step1<<<(N+255)/256, 256>>>(y, ytemp, N);
  (*dydt)((*t)+(*h)*c[0],y,k1);

  step2<<<(N+255)/256, 256>>>(y, k1, ytemp, *h*a1[0], N);
  (*dydt)((*t)+(*h)*c[1],ytemp,k2);

  step3<<<(N+255)/256, 256>>>(y, k1, k2, ytemp, *h*a2[0], *h*a2[1], N);
  (*dydt)((*t)+(*h)*c[2],ytemp,k3);

  step4<<<(N+255)/256, 256>>>(y, k1, k2, k3, ytemp, *h*a3[0], *h*a3[1], *h*a3[2], N);
  (*dydt)((*t)+(*h)*c[3],ytemp,k4);

  step5<<<(N+255)/256, 256>>>(y, k1, k2, k3, k4, ytemp, *h*a4[0], *h*a4[1], *h*a4[2], *h*a4[3], N);
  (*dydt)((*t)+(*h)*c[4],ytemp,k5);

  step6<<<(N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, ytemp, *h*a5[0], *h*a5[1], *h*a5[2], *h*a5[3], *h*a5[4], N);
  (*dydt)((*t)+(*h)*c[5],ytemp,k6);

  cudaMemcpy (normd, &norm, 1*sizeof(double), cudaMemcpyHostToDevice);
  error<<<(N+255)/256, 256>>>(normd, y, k1, k2, k3, k4, k5, k6, yerr, *h*(b1[0]-b2[0]), *h*(b1[1]-b2[1]), *h*(b1[2]-b2[2]), *h*(b1[3]-b2[3]), *h*(b1[4]-b2[4]), *h*(b1[5]-b2[5]), atl, rtl, N);
  cudaMemcpy (&norm, normd, 1*sizeof(double), cudaMemcpyDeviceToHost);
  norm=sqrt(norm)/N;
  double factor=0.9*pow(norm,-0.2);
  if (factor<0.2)
    factor=0.2;
  if (factor>10)
    factor=10;

  //Accept or reject the step and update the step size
  if(norm<1){
    accept<<<(N+255)/256, 256>>>(y,  k1, k2, k3, k4, k5, k6, *h*b1[0], *h*b1[1], *h*b1[2], *h*b1[3], *h*b1[4], *h*b1[5], N);

    (*t)=(*t)+(*h);
    (*h)*=factor;
  }
  else if (factor<1){
    (*h)*=factor;
  }
}

double* rkf45_init(int n, double atol, double rtol, double *yloc, void (*func)(double, double*, double*)){
  N=n;
  rtl=rtol;
  atl=atol;
  dydt=func;

  cudaMalloc ((void**)&y, N*sizeof(double));
  // cudaMalloc ((void**)&f, N*sizeof(double));
  cudaMalloc ((void**)&yerr, N*sizeof(double));
  cudaMalloc ((void**)&ytemp, N*sizeof(double));
  cudaMalloc ((void**)&k1, N*sizeof(double));
  cudaMalloc ((void**)&k2, N*sizeof(double));
  cudaMalloc ((void**)&k3, N*sizeof(double));
  cudaMalloc ((void**)&k4, N*sizeof(double));
  cudaMalloc ((void**)&k5, N*sizeof(double));
  cudaMalloc ((void**)&k6, N*sizeof(double));
  cudaMalloc ((void**)&normd, 1*sizeof(double));
  cudaMemcpy (y, yloc, N*sizeof(double), cudaMemcpyHostToDevice);
  return y;
}

void rkf45_destroy(){
  cudaFree(y);
  cudaFree(yerr);
  cudaFree(ytemp);
  cudaFree(k1);
  cudaFree(k2);
  cudaFree(k3);
  cudaFree(k4);
  cudaFree(k5);
  cudaFree(k6);
  cudaFree(normd);
}
