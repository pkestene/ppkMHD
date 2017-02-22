# ppkMHD

## What is it ?

ppkMHD stands for Performance Portable Kokkos for Magnetohydrodynamics solvers.

## Dependencies

* [Kokkos](https://github.com/kokkos/kokkos): for now (Feb 2017) it is required to use a version of kokkos that comes with kokkos.cmake (e.g. https://github.com/pkestene/kokkos branch develop_cmake). We assume you know what you are doing, and have build kokkos previous with the right options (architecture, flags, ...).
 
   
## Build

A few example builds

### Build without MPI / With Kokkos-openmp

* Create a build directory, configure and make
```shell
mkdir build; cd build
cmake -DUSE_MPI=OFF ..
make -j 4
```

Add variable CXX on the cmake command line to change the compiler (clang++, icpc, pgcc, ....)

### Build without MPI / With Kokkos-cuda

* Create a build directory, configure and make
```shell
mkdir build; cd build
CXX=/path/to/nvcc_wrapper cmake -DUSE_MPI=OFF ..
make -j 4
```

### Build with MPI / With Kokkos-cuda

* Make sure MPI compiler wrapper will use `nvcc_wrapper` from Kokkos
```shell
export OMPI_CXX=/path/to/nvcc_wrapper
```

* Create a build directory, configure and make
```shell
mkdir build; cd build
CXX=mpicxx cmake ..
make -j 4
```

