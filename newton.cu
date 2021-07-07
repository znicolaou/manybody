//Zachary G. Nicolaou 7/5/2021
//Simulate Newtons equations with periodic boundaries on a gpu
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

//Rkf45 coefficients
const double a1[1] = {1.0/4};
const double a2[2] = {3.0/32, 9.0/32};
const double a3[3] = {1932.0/2197, -7200.0/2197, 7296.0/2197};
const double a4[4] = {439.0/216, -8.0, 3680.0/513, -845.0/4104};
const double a5[5] = {-8.0/27, 2.0, -3544.0/2565, 1859.0/4104, -11.0/40};
const double b1[6] = {16.0/135, 0.0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55};
const double b2[6] = {25.0/216, 0.0, 1408.0/2565, 2197.0/4104, -1.0/5, 0.0};
const double c[6] = {0.0, 1.0/4, 3.0/8, 12.0/13, 1.0, 1.0/2};

//Parameters
int M,N,dim;
double L, R, V, H;

//It would be better to fuse all these kernels if we can

//Set the derivative for the position variables
__global__ void dydt0 (double t, double* y, double* f, int *p1, int *p2, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    for (int k=0; k<dim; k++){
      f[(2*dim)*i+k]=y[(2*dim)*i+k+dim];
      f[(2*dim)*i+dim+k]=0;
    }
  }
}
//Set the derivative for the velocity variables
__global__ void dydt (double t, double* y, double* f, int *p1, int *p2, int M, double L, double R, double H, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M){
    double norm=0;
    for (int k=0; k<dim; k++){
      //Find the smallest periodic distance in each dimension, and calculate the norm
      double d1=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k];
      double d2=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]-L;
      double d3=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]+L;
      double d=d1;
      if(fabs(d)>fabs(d2))
        d=d2;
      if(fabs(d)>fabs(d3))
        d=d3;
      norm=norm+d*d;
    }
    //Calculate the soft interaction if the particles are in contact
    if(sqrt(norm) < 2*R){
      for (int k=0; k<dim; k++){
        double d1=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k];
        double d2=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]-L;
        double d3=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]+L;
        double d=d1;
        if(fabs(d)>fabs(d2))
          d=d2;
        if(fabs(d)>fabs(d3))
          d=d3;
        atomicAdd(&(f[(2*dim)*p1[i]+dim+k]), -H*(d/sqrt(norm))*pow(2*R-sqrt(norm),1.5));
        atomicAdd(&(f[(2*dim)*p2[i]+dim+k]), H*(d/sqrt(norm))*pow(2*R-sqrt(norm),1.5));
      }
    }
  }
}
//Steps for the RK stepper
__global__ void step3 (double* y, double* k1, double* k2, double* ytemp, const double a20, const double a21, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N) {
    ytemp[i]=y[i]+a20*k1[i]+a21*k2[i];
  }
}
__global__ void step4 (double* y, double* k1, double* k2, double* k3, double* ytemp, const double a30, const double a31, const double a32, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N) {
    ytemp[i]=y[i]+a30*k1[i]+a31*k2[i]+a32*k3[i];
  }
}
__global__ void step5 (double* y, double* k1, double* k2, double* k3, double* k4, double* ytemp, const double a40, const double a41, const double a42, const double a43, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N) {
    ytemp[i]=y[i]+a40*k1[i]+a41*k2[i]+a42*k3[i]+a43*k4[i];
  }
}
__global__ void step6 (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* ytemp, const double a50, const double a51, const double a52, const double a53, const double a54, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N) {
    ytemp[i]=y[i]+a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i];
  }
}
//Error estimate for the RK stepper
__global__ void error (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, double* yerr, double* ytemp, const double atl, const double rtl, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N) {
    yerr[i]=(a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i]+a55*k6[i])/(atl+rtl*y[i]);
  }
}
//Accept the RK step (enforcing periodicity here)
__global__ void accept (double* y, double* k1, double* k2, double* k3, double* k4, double* k5, double* k6, const double a50, const double a51, const double a52, const double a53, const double a54, const double a55, int N, double L, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<(2*dim)*N){
    if((i%(2*dim))<dim)
      y[i]=fmod(L+y[i]+a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i]+a55*k6[i],L);
    else
      y[i]=y[i]+a50*k1[i]+a51*k2[i]+a52*k3[i]+a53*k4[i]+a54*k5[i]+a55*k6[i];
  }
}

//Attempt a RK step
void rkf45 (cublasHandle_t handle, double *t, double *h, double *y, double *ytemp, int *p1, int *p2, double *k1, double *k2, double *k3, double *k4, double *k5, double *k6, double atl, double *yerr, double rtl){
  double norm;
  double A10=*h*a1[0];

  //Calculate the intermediate steps and error estimates using the CUDA kernels
  cublasDcopy(handle, (2*dim)*N, y, 1, ytemp, 1);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[0],ytemp,k1,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[0],ytemp,k1,p1,p2, M, L, R, H, dim);
  cublasDaxpy(handle, (2*dim)*N, &A10, k1, 1, ytemp, 1);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[1],ytemp,k2,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[1],ytemp,k2,p1,p2, M, L, R, H, dim);
  step3<<<((2*dim)*N+255)/256, 256>>>(y, k1, k2, ytemp, *h*a2[0], *h*a2[1], N, dim);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[2],ytemp,k3,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[2],ytemp,k3,p1,p2, M, L, R, H, dim);
  step4<<<((2*dim)*N+255)/256, 256>>>(y, k1, k2, k3, ytemp, *h*a3[0], *h*a3[1], *h*a3[2], N, dim);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[3],ytemp,k4,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[3],ytemp,k4,p1,p2, M, L, R, H, dim);
  step5<<<((2*dim)*N+255)/256, 256>>>(y, k1, k2, k3, k4, ytemp, *h*a4[0], *h*a4[1], *h*a4[2], *h*a4[3], N, dim);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[4],ytemp,k5,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[4],ytemp,k5,p1,p2, M, L, R, H, dim);
  step6<<<((2*dim)*N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, ytemp, *h*a5[0], *h*a5[1], *h*a5[2], *h*a5[3], *h*a5[4], N, dim);
  dydt0<<<(N+255)/256, 256>>>((*t)+(*h)*c[5],ytemp,k6,p1,p2, N, dim);
  dydt<<<(M+255)/256, 256>>>((*t)+(*h)*c[5],ytemp,k6,p1,p2, M, L, R, H, dim);
  error<<<((2*dim)*N+255)/256, 256>>>(y, k1, k2, k3, k4, k5, k6, yerr, ytemp, atl, rtl, *h*(b1[0]-b2[0]), *h*(b1[1]-b2[1]), *h*(b1[2]-b2[2]), *h*(b1[3]-b2[3]), *h*(b1[4]-b2[4]), *h*(b1[5]-b2[5]), N, dim);

  //Determine the error norm and step update factor
  cublasDnrm2(handle, (2*dim)*N, yerr, 1, &norm);
  norm/=N;
  double factor=0.9*pow(norm,-0.2);
  if (factor<0.2)
    factor=0.2;
  if (factor>10)
    factor=10;

  //Accept or reject the step and update the step size
  if(norm<1){
    accept<<<((2*dim)*N+255)/256, 256>>>(y,  k1, k2, k3, k4, k5, k6, *h*b1[0], *h*b1[1], *h*b1[2], *h*b1[3], *h*b1[4], *h*b1[5], N, L, dim);
    (*t)=(*t)+(*h);
    (*h)*=factor;
  }
  else if (factor<1){
    (*h)*=factor;
  }
}

//Main function
int main (int argc, char* argv[]) {
    //Command line arguments and defaults
    struct timeval start,end;
    double atl, rtl;
    int gpu, seed;
    double t1, t3, dt;
    char* filebase;
    N=2048;
    L=32;
    R=0.5;
    V=0.1;
    H=10;
    dim=2;
    t1=1e2;
    t3=0;
    dt=1e-1;
    gpu=0;
    seed=1;
    int verbose=0;
    rtl=1e-6;
    atl=1e-6;
    char ch;
    int help=1;
    while (optind < argc) {
      if ((ch = getopt(argc, argv, "N:L:D:R:V:H:g:t:A:d:s:p:r:a:hv")) != -1) {
        switch (ch) {
          case 'N':
              N = (int)atoi(optarg);
              break;
          case 'L':
              L = (double)atof(optarg);
              break;
          case 'D':
              dim = (int)atoi(optarg);
              break;
          case 'V':
              V = (double)atof(optarg);
              break;
          case 'H':
              H = (double)atof(optarg);
              break;
          case 'g':
              gpu = (double)atof(optarg);
              break;
          case 't':
              t1 = (double)atof(optarg);
              break;
          case 'A':
              t3 = (double)atof(optarg);
              break;
          case 'd':
              dt = (double)atof(optarg);
              break;
          case 's':
              seed = (int)atoi(optarg);
              break;
          case 'r':
              rtl = (double)atof(optarg);
              break;
          case 'a':
              atl = (double)atof(optarg);
              break;
          case 'h':
              help=1;
              break;
          case 'v':
              verbose=1;
              break;
        }
      }
      else {
        filebase=argv[optind];
        optind++;
        help=0;
      }
    }
    if (help) {
      printf("usage:\t2dcgle [-h] [-v] [-N N] [-L L] [-R R] [-V V] [-H H]\n");
      printf("\t[-t t1] [-A t3] [-d dt] [-s seed] \n");
      printf("\t[-r rtol] [-a atol] [-g gpu] filebase \n\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("N is number of particles. Default 2048. \n");
      printf("L is linear system size. Default 32. \n");
      printf("R is particle radius. Default 0.5. \n");
      printf("V is initial velocity scale. Default 0.1. \n");
      printf("H is hardness scale. Default 10. \n");
      printf("t1 is total integration time. Default 1e2. \n");
      printf("t3 is time stop outputting dense timestep data. Default 0. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("diff is 0 for finite diff, 1 for pseudospectral. Default 1.\n");
      printf("rtol is relative error tolerance. Default 1e-6.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("filebase is base file name for output. \n");
      exit(0);
    }

    //Initialization
    double t=0,h,t0;
    int i,j,k,steps=0;
    FILE *outlast, *outstates,*outtimes, *out, *in;
    char file[256];
    strcpy(file,filebase);
    strcat(file, "states.dat");
    outstates = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,".out");
    out = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file, "times.dat");
    outtimes = fopen(file,"w");
    double *yloc;
    int *p1loc, *p2loc;
    double *y, *f, *ytemp, *yerr, *k1, *k2, *k3, *k4, *k5, *k6;
    int *p1, *p2;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaSetDevice(gpu);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    M=N*(N-1)/2; //Number of pairwise interactions
    yloc = (double*)calloc((2*dim)*N,sizeof(double));
    p1loc = (int*)calloc(M,sizeof(int));
    p2loc = (int*)calloc(M,sizeof(int));
    size_t fr, total;
    cudaMemGetInfo (&fr, &total);
    printf("GPU Memory: %li %li\n", fr, total);
    if(fr < (100*(2*dim)*N+M)*sizeof(double)) {
      printf("GPU Memory low! \n");
      return 0;
    }
    cudaMalloc ((void**)&y, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&f, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&yerr, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&ytemp, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&p1, M*sizeof(int));
    cudaMalloc ((void**)&p2, M*sizeof(int));
    cudaMalloc ((void**)&k1, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&k2, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&k3, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&k4, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&k5, (2*dim)*N*sizeof(double));
    cudaMalloc ((void**)&k6, (2*dim)*N*sizeof(double));
    fprintf(out, "%i %i %f %f\n", N, dim, L, R);
    fflush(out);

    //Initial conditions
    strcpy(file,filebase);
    strcat(file, "ic.dat");
    if ((in = fopen(file,"r")))
    {
        printf("Using initial conditions from file\n");
        fprintf(out, "Using initial conditions from file\n");
        size_t read=fread(yloc,sizeof(double),(2*dim)*N,in);
        fclose(in);
        if (read!=(2*dim)*N){
          printf("initial conditions file not compatible with N!");
          return 0;
        }
    }
    else {
        printf("Using random initial conditions\n");
        fprintf(out, "Using random initial conditions\n");
        srand(seed);

        for(j=0; j<N; j++){
          for(k=0; k<dim; k++){
            yloc[2*dim*j+k]=L/RAND_MAX*rand();
            yloc[2*dim*j+dim+k]=V*(2.0/RAND_MAX*rand()-1);
          }
        }
    }
    //Particle indices for each pairwise interaction
    int ind=0;
    for(i=0; i<N; i++){
      for(j=0; j<i; j++){
        p1loc[ind]=i;
        p2loc[ind]=j;
        ind++;
      }
    }
    gettimeofday(&start,NULL);
    h = dt/100;
    double tlast=-1;
    cublasSetVector ((2*dim)*N, sizeof(double), yloc, 1, y, 1);
    cublasSetVector (M, sizeof(int), p1loc, 1, p1, 1);
    cublasSetVector (M, sizeof(int), p2loc, 1, p2, 1);

    //Main integration loop
    while(t<t1+dt){
      t0=t;
      if(t>=t3){ //Output
        cublasGetVector ((2*dim)*N, sizeof(double), y, 1, yloc, 1);
        fwrite(yloc,sizeof(double),(2*dim)*N,outstates);
        fflush(outstates);
      }

      while(t<t0+dt){ //Take steps until next output
        steps++;
        if(verbose) {
          gettimeofday(&end,NULL);
          printf("%.3f\t%1.3e\t%1.3e\t%f\t%i\n",t/t1, end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec), (end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec))/((t+h)/t1)*(1-t/t1), h, steps);
          fprintf(out,"%.3f\t%1.3e\t%1.3e\t%f\t%i\n",t/t1, end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec), (end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec))/((t+h)/t1)*(1-t/t1), h, steps);
          fflush(stdout);
          fflush(out);
        }
        if(t>tlast){
          fwrite(&t,sizeof(double),1,outtimes);
          tlast=t;
        }
        fflush(outtimes);
        if(t+h>t0+dt)
          h=t0+dt-t;
        rkf45 (handle, &t, &h, y, ytemp, p1, p2, k1, k2, k3, k4, k5, k6, atl, yerr, rtl);
      }

    }

    //Output final state and summary
    cublasGetVector ((2*dim)*N, sizeof(double), y, 1, yloc, 1);
    strcpy(file,filebase);
    strcat(file,"fs.dat");
    outlast=fopen(file,"w");
    fwrite(yloc,sizeof(double),(2*dim)*N,outlast);
    fflush(outlast);
    fclose(outlast);
    gettimeofday(&end,NULL);
    printf("\nruntime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fprintf(out,"\nsteps: %i\n",steps);
    fprintf(out,"runtime: %f\n",end.tv_sec-start.tv_sec + 1e-6*(end.tv_usec-start.tv_usec));
    fclose(outstates);
    fclose(outtimes);
    fclose(out);


    //Deallocate
    free(yloc);
    free(p1loc);
    free(p2loc);
    cudaFree(y);
    cudaFree(yerr);
    cudaFree(ytemp);
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
    cudaFree(k5);
    cudaFree(k6);
    cublasDestroy(handle);

    return 0;
}
