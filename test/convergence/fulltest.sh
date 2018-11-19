#!/bin/bash

########################################################################################
########################################################################################
# (REQUIRED) Edit the following variables to reflect your local environment
########################################################################################

# Target run directory (must include the mhd_wave_3D.ini.skel file)
TARGETDIR=/home/pgrete/src/ppkMHD/test/convergence/test

# Full path to ppkMHD executable 
BIN=/home/pgrete/src/ppkMHD/build_mpi-omp/src/ppkMHD 

# Base mpicommand (e.g., mpirun for OpenMPI)
MPICMD=mpirun

# MPI number of processes parameter (e.g., -np for OpenMPI)
MPINP="-np"

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=2

########################################################################################
########################################################################################
# (Optional) Edit the following variables to change usage of mpi and resolution of tests
########################################################################################

# Array indices are the linear resolution in x-direction
# TODO fix plotting script to work with parallel output. For now set everything to 1.
declare -A NMPIS=( [16]=1 [32]=1 [64]=1 [128]=1 [256]=1 [512]=1)

# Array indices are the wave types
declare -A TENDS=( [0]=0.5 [1]=1.0 [2]=2.0 [3]=1.0)


########################################################################################
########################################################################################

cd $TARGETDIR 

for WAVETYPE in "${!TENDS[@]}"; do
  mkdir $WAVETYPE
  cd $WAVETYPE 

  for NX in "${!NMPIS[@]}"; do
    mkdir $NX
    cd $NX
    cp ../../mhd_wave_3D.ini.skel mhd_wave_3D.ini
    
    
    sed -i "s/WAVETYPE/$WAVETYPE/" mhd_wave_3D.ini 
    sed -i "s/TEND/${TENDS[$WAVETYPE]}/" mhd_wave_3D.ini

    MX=${NMPIS[$NX]}
    echo $MX
    if [ $MX -eq 1 ]; then
      MY=1
      MZ=1

      NY=$((NX/2))
      NZ=$((NX/2))

    else
      MY=$((MX/2))
      MZ=$((MX/2))

      NY=$((NX/MY/2))
      NZ=$((NX/MZ/2))
      NX=$((NX/MX))
    fi

    sed -i "s/NX/$NX/" mhd_wave_3D.ini
    sed -i "s/NY/$NY/" mhd_wave_3D.ini
    sed -i "s/NZ/$NZ/" mhd_wave_3D.ini

    sed -i "s/MX/$MX/" mhd_wave_3D.ini
    sed -i "s/MY/$MY/" mhd_wave_3D.ini
    sed -i "s/MZ/$MZ/" mhd_wave_3D.ini

    $MPICMD $MPINP $((MX*MY*MZ)) $BIN mhd_wave_3D.ini | tee wave.out

    cd ..
  done
  
  cd ..
done
