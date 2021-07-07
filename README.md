# newton
Integrate Newton's equations on a GPU

# Installation
Create a conda environment with required packages:
`conda create -n newton_env -c conda-forge numpy scipy jupyter matplotlib`.  Activate this environment to run the oscillators.py script: `conda activate newton_env`. 

The newton.cu program can be compiled on a computer with the nvidia cuda compiler with `nvcc -lcublas -lcuda -arch=sm_61 -o newton newton.cu`.

# Usage
Running `./newton -h` will produce the following help message:
```
usage:	2dcgle [-h] [-v] [-N N] [-L L] [-R R] [-V V] [-H H]
	[-t t1] [-A t3] [-d dt] [-s seed] 
	[-r rtol] [-a atol] [-g gpu] filebase 

-h for help 
-v for verbose 
N is number of particles. Default 2048. 
L is linear system size. Default 32. 
R is particle radius. Default 0.5. 
V is initial velocity scale. Default 0.1. 
H is hardness scale. Default 10. 
t1 is total integration time. Default 1e2. 
t3 is time stop outputting dense timestep data. Default 0. 
dt is the time between outputs. Default 1e0. 
seed is random seed. Default 1. 
diff is 0 for finite diff, 1 for pseudospectral. Default 1.
rtol is relative error tolerance. Default 1e-6.
atol is absolute error tolerance. Default 1e-6.
gpu is index of the gpu. Default 0.
filebase is base file name for output. 
```

# Output files
filebasefs.dat contains the final states of the system.
filebasestates.dat contains dense output of the states and at each time step for t>t3.

# Input files
If filebaseic.dat exists, the C array is used as initial conditions. Final states from previous runs can be copied for initial conditions.
