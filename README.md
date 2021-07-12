# newton
Integrate Newton's equations on a GPU. The interaction is a soft particle potential, and the integrator is a embedding 4/5 Runge-Kutta stepper with adaptive timestepping.

# Installation
Create a conda environment with required packages:
`conda create -n newton_env -c conda-forge numpy scipy jupyter matplotlib`.  Activate this environment to use the plot jupyter notebook: `conda activate newton_env`. 

The newton.cu program can be compiled on a computer with the nvidia cuda compiler with `nvcc -lcublas -lcuda -arch=sm_61 -o newton newton.cu`.

# Usage
Running `./newton -h` will produce the following help message:
```
usage:	newton [-h] [-v] [-N N] [-t t1] [-T t2] [-A t3]
	 [-d dt] [-L L] [-R R] [-q R0] [-V V] [-H H] [-s seed]
	 [-r rtol] [-a atol] [-g gpu] filebase  

-h for help 
-v for verbose 
N is number of particles. Default 2048. 
t1 is total integration time. Default 1e2. 
t2 is time to quasistatically vary the radius from R0 to R. Default 0. 
t3 is time start outputting dense state data. Default 0. 
dt is the time between outputs. Default 1e0. 
L is linear system size. Default 32. 
R is the final particle radius. Default 0.5. 
R0 is initial particle radius. Default 0.5. 
V is initial velocity scale. Default 0.1. 
H is hardness scale. Default 10. 
seed is random seed. Default 1. 
rtol is relative error tolerance. Default 1e-6.
atol is absolute error tolerance. Default 1e-6.
gpu is index of the gpu. Default 0.
filebase is base file name for output. 
```

The required positional argument filebase is a string which specifies input and output file locations. Other arguments are optional and modify the simulation parameters.

# Output files
The newton script creates output files filebase.out, filebasefs.dat, filebasetimes.dat, and filebasestates.dat.

filebase.out contains parameters that ran the files and runtime output.

filebasefs.dat contains the final states of the system.

filebasetimes.dat contains the timesteps chosen by the integrator.

filebasestates.dat contains dense output of the states and at each time step for t>t3.

filebaseorders.dat contains dense output of the particles making contact at each time step for t>t3.

# Input files
If filebaseic.dat exists, the C array is used as initial conditions. Final states from previous runs can be copied for initial conditions.
