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

### Additionnal requirements

The MOOD numerical scheme require some linear algebra (QR decomposition) on the host (not device). This is done using a Blas/Lapack implementation using the C language interface named Lapacke.

Please note that Atlas doesn't provide Lapackage.
Currently (March 2017), on Ubuntu 16.04, package libatlas-dev is not compatible with package Lapacke (generate errors at link time). So please either Netlib or OpenBLAS implementation.

If you want to enforce the use of OpenBLAS, just use a recent CMake (>=3.6) and add "-DBLA_VENDOR" on the cmake command line. This will tell the cmake system (through the call to find_package(BLAS) ) to only look for OpenBLAS implementation.

On a recent Ubuntu, if atlas is not installed, but OpenBLAS is, you don't need to have a bleeding edge CMake, current cmake will find OpenBLAS.

### Developping with vim and youcomplete plugin

Assuming you are using vim (or neovim) text editor and have installed the youcomplete plugin, you can have
semantic autocompletion in a C++ project.

Make sure to have CMake variable CMAKE_EXPORT_COMPILE_COMMANDS set to ON, and symlink the generated file to the top level
source directory.

