//Zachary G. Nicolaou 7/5/2021
//Simulate Newtons equations with periodic boundaries on a gpu
#include <stdio.h>
#include <cuda_runtime.h>

void rkf45_step (double *t, double *h);
double* rkf45_init(int n, double atl, double rtl, double *yloc, void (*dydt)(double, double*, double*));
void rkf45_destroy();
