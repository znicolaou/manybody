//Zachary G. Nicolaou 7/5/2021
//Simulate Newtons equations with periodic boundaries on a gpu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "rkf45.h"

//Parameters
int M,N,dim,bins;
double t1, t2, t3, dt, L, R0, R, V, H, dmax;
int *p1, *p2, *orders;

//The interaction force give separation sqrt(norm) and distance d along a specified axis
//For friction, we'd need to change this to include all directions
__device__ double force(double d, double norm, double H, double Rt){
  if(sqrt(norm) < 2*Rt)
    return -H*(d/sqrt(norm))*pow(1-sqrt(norm)/(2*Rt),1.5);
  return 0;
}
//Set the derivative for the position variables
//Can add repulsive walls or external forces here as well
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
__global__ void dydt1 (double t, double* y, double* f, int *p1, int *p2, int M, double L, double R, double H, double R0, double t2, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M){
    double Rt=R;
    if(t<t2){
      Rt=R0+t/t2*(R-R0);
    }
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
    //Calculate the interaction force
    for (int k=0; k<dim; k++){
      double d1=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k];
      double d2=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]-L;
      double d3=y[(2*dim)*p2[i]+k]-y[(2*dim)*p1[i]+k]+L;
      double d=d1;
      if(fabs(d)>fabs(d2))
        d=d2;
      if(fabs(d)>fabs(d3))
        d=d3;
      atomicAdd(&(f[(2*dim)*p1[i]+dim+k]), force(d, norm, H, Rt));
      atomicAdd(&(f[(2*dim)*p2[i]+dim+k]), force(-d, norm, H, Rt));
    }
  }
}
//enforce periodic boundary conditions
__global__ void periodic (double* y, double L, int N, int dim) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    for (int k=0; k<dim; k++){
      y[(2*dim)*i+k]=fmod(L+y[(2*dim)*i+k],L);
    }
  }
}

__global__ void order (double t, double t2, double* y, int *p1, int *p2, int *orders, int M, double L, double dmax, int dim, int bins) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<M){
    int ind=0, ind2=0;
    int include=1;
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

      if (fabs(d)>dmax){
        include=0;
      }
      ind=ind+(int)pow((double)bins,(double)k)*int((d+dmax)*bins/(2*dmax));
      ind2=ind2+(int)pow((double)bins,(double)k)*int((-d+dmax)*bins/(2*dmax));
    }
    if(include){
      atomicAdd(&(orders[ind]),1);
      atomicAdd(&(orders[ind2]),1);
    }
  }
}

//function for rkf45
void dydt(double t, double *y, double *f){
  dydt0<<<(N+255)/256, 256>>>(t,y,f,p1,p2, N, dim);
  dydt1<<<(M+255)/256, 256>>>(t,y,f,p1,p2, M, L, R, H, R0, t2, dim);
}

//Main function
int main (int argc, char* argv[]) {
    //Command line arguments and defaults
    struct timeval start,end;
    double atl, rtl;
    int gpu, seed;
    char* filebase;
    N=512;
    L=32;
    dmax=5;
    bins=100;
    R0=0.5;
    R=0.5;
    V=0.1;
    H=100;
    dim=2;
    t1=1e1;
    t2=0;
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
      if ((ch = getopt(argc, argv, "N:b:B:L:q:D:R:V:H:g:t:T:A:d:s:r:a:hv")) != -1) {
        switch (ch) {
          case 'N':
              N = (int)atoi(optarg);
              break;
          case 'b':
              bins = (int)atoi(optarg);
              break;
          case 'B':
              dmax = (double)atof(optarg);
              break;
          case 'L':
              L = (double)atof(optarg);
              break;
          case 'q':
              R0 = (double)atof(optarg);
              break;
          case 'D':
              dim = (int)atoi(optarg);
              break;
          case 'R':
              R = (double)atof(optarg);
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
          case 'T':
              t2 = (double)atof(optarg);
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
      printf("usage:\tmanybody  [-N N] [-b b] [-B dmax] [-L L] [-q R0] [-D dim] [-R R]  \n");
      printf("\t [-V V] [-H H] [-g gpu] [-t t1] [-T t2] [-A t3] [-d dt] [-s seed]\n");
      printf("\t [-r rtol] [-a atol] [-h] [-v]  FILEBASE  \n\n");
      printf("N is number of particles. Default 2048. \n");
      printf("b is number of bins for pair correlations. Default 100. \n");
      printf("dmax is maximum distance for pair correlations. Default 5. \n");
      printf("L is linear system size. Default 32. \n");
      printf("R0 is initial particle radius. Default 0.5. \n");
      printf("dim is the dimension. Default 2. \n");
      printf("R is the final particle radius. Default 0.5. \n");
      printf("V is initial velocity scale. Default 0.1. \n");
      printf("H is hardness scale. Default 10. \n");
      printf("gpu is index of the gpu. Default 0.\n");
      printf("t1 is total integration time. Default 1e2. \n");
      printf("t2 is time to quasistatically vary the radius from R0 to R. Default 0. \n");
      printf("t3 is time start outputting dense state data. Default 0. \n");
      printf("dt is the time between outputs. Default 1e0. \n");
      printf("seed is random seed. Default 1. \n");
      printf("rtol is relative error tolerance. Default 1e-6.\n");
      printf("atol is absolute error tolerance. Default 1e-6.\n");
      printf("-h for help \n");
      printf("-v for verbose \n");
      printf("FILEBASE is base file name for output. \n");
      exit(0);
    }

    //Initialization
    double t=0,h,t0;
    int i,j,k,steps=0;
    FILE *outlast, *outstates,*outtimes, *outorder, *out, *in;
    char file[256];
    strcpy(file,filebase);
    strcat(file, "states.dat");
    outstates = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file,"out.dat");
    out = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file, "times.dat");
    outtimes = fopen(file,"w");
    strcpy(file,filebase);
    strcat(file, "orders.dat");
    outorder = fopen(file,"w");
    double *yloc;
    int *p1loc, *p2loc, *ordersloc;
    cudaSetDevice(gpu);

    M=N*(N-1)/2; //Number of pairwise interactions
    yloc = (double*)calloc((2*dim)*N,sizeof(double));
    p1loc = (int*)calloc(M,sizeof(int));
    p2loc = (int*)calloc(M,sizeof(int));
    ordersloc = (int*)calloc((int)pow((double)bins,(double)dim),sizeof(int));
    size_t fr, total;
    cudaMemGetInfo (&fr, &total);
    printf("GPU Memory: %li %li\n", fr, total);
    if(fr < (100*(2*dim)*N+M)*sizeof(double)) {
      printf("GPU Memory low! \n");
      return 0;
    }

    cudaMalloc ((void**)&p1, M*sizeof(int));
    cudaMalloc ((void**)&p2, M*sizeof(int));
    cudaMalloc ((void**)&orders, (int)pow((double)bins,(double)dim)*sizeof(int));

    //Initial conditions
    fprintf(out, "%i %i %f %f %f %f %f %f %f %f %f %f\n", N, dim, t1, t2, t3, dt, L, R0, R, V, H, dmax);
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
    cudaMemcpy (p1, p1loc, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (p2, p2loc, M*sizeof(int), cudaMemcpyHostToDevice);
    double* y=rkf45_init(2*dim*N, atl, rtl, yloc, &dydt);

    for(k=0; k<(int)pow((double)bins,(double)dim); k++){
      ordersloc[k]=0;
    }
    cudaMemcpy (orders, ordersloc, (int)pow((double)bins,(double)dim)*sizeof(int), cudaMemcpyHostToDevice);


    //Main integration loop
    while(t<t1+dt){
      t0=t;
      if(t>=t3){ //Output
        order<<<(M+255)/256, 256>>>(t, t2, y, p1, p2, orders, M, L, dmax, dim, bins);
        cudaMemcpy (yloc, y, (2*dim)*N*sizeof(double), cudaMemcpyDeviceToHost);
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
        rkf45_step (&t, &h);
        //if periodic
        periodic<<<(2*dim*N+255)/256,256>>>(y,L,N,dim);
      }

    }

    //Output final state and summary
    for (i=0; i<argc; i++){
      fprintf(out, "%s ", argv[i]);
    }
    fprintf(out, "\n");
    fflush(out);
    cudaMemcpy (yloc, y, (2*dim)*N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy (ordersloc, orders, (int)pow((double)bins,(double)dim)*sizeof(int), cudaMemcpyDeviceToHost);
    fwrite(ordersloc,sizeof(int),(int)pow((double)bins,(double)dim),outorder);


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
    fclose(outorder);
    fclose(out);

    //Deallocate
    free(yloc);
    free(p1loc);
    free(p2loc);
    free(ordersloc);
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(orders);
    rkf45_destroy();

    return 0;
}
