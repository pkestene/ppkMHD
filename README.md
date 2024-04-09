[![DOI](https://zenodo.org/badge/209853926.svg)](https://zenodo.org/badge/latestdoi/209853926) ![C/C++ CI](https://github.com/pkestene/ppkMHD/workflows/C/C++%20CI/badge.svg)

# ppkMHD

## What is it ?

ppkMHD stands for Performance Portable Kokkos for Magneto-HydroDynamics (MHD) solvers.

Here a small list of numerical schemes implementations:

- second order MUSCL-HANCOCK scheme for hydrodynamics and MHD
- high-order MOOD (hydrodynamics only)
- high-order Spectral Difference Method schemes: hydrodynamics only

All scheme are available in 2D and 3D using Kokkos+MPI implementation, and support at least the OpenMP and CUDA kokkos backends (other backends may be used, but not tested).

## Dependencies

* [Kokkos](https://github.com/kokkos/kokkos) library will either be built by ppkMHD by using cmake option `-DPPKMHD_BUILD_KOKKOS=ON`, either be detected if you already have it installed.
* [cmake](https://cmake.org/) with version >= 3.X (3.X is chosen to meet Kokkos own requirement for cmake; i.e. it might increase in the future)


For beginner user, wu suggest that you build ppkMHD with `-DPPKMHD_BUILD_KOKKOS=ON` so that Kokkos is built together, with the same flags as the main application.

## Build

A few example builds, with minimal configuration options.

### If you already have Kokkos installed

If you build kokkos yourself, we advise you to always active HWLOC TPLS.

Just make sure that your env variable `CMAKE_PREFIX_PATH` point to the location where Kokkos where installed. More precisely if Kokkos is installed in `KOKKOS_ROOT`, you add `$KOKKOS_ROOT/lib/cmake` to your `CMAKE_PREFIX_PATH`; this way kokkos will be found automagically by cmake, and the right Kokkos hardware backend will be selected.

```shell
# cd into ppkMHD toplevel sources
cmake -S . -B _build/default -DPPKMHD_BUILD_KOKKOS=OFF
cmake --build _build/default -j 6
```

### Build ppkMHD and kokkos without MPI activated for Kokkos-openmp backend

* Create a build directory, configure and make

```shell
# cd into ppkMHD toplevel sources
cmake -S . -B _build/openmp -DPPKMHD_BUILD_KOKKOS=ON -DPPKMHD_USE_MPI=OFF -DPPKMHD_BACKEND=OpenMP -DKokkos_ENABLE_HWLOC=ON
cmake --build _build/openmp -j 6
```

Add variable CXX on the cmake command line to change the compiler (clang++, icpc, pgcc, ....).

### Build ppkMHD and kokkos without MPI activated for Kokkos-cuda backend

* Create a build directory, configure and make

```shell
# cd into ppkMHD toplevel sources
cmake -S . -B _build/cuda -DPPKMHD_BUILD_KOKKOS=ON -DPPKMHD_USE_MPI=OFF -DPPKMHD_BACKEND=Cuda
# note: GPU architecture will be detected when building on a host with a GPU; if you're building on a host, and running on another machine
# you'll need to tell kokkos what is the target architecture, e.g. add flag like '-DKokkos_ARCH_TURING75=ON' on the cmake configure line
cmake --build _build/cuda -j 6
```

### Build ppkMHD with MPI activated for Kokkos-cuda backend

Please make sure to use a **CUDA-aware MPI implementation** (OpenMPI or MVAPICH2) built with the proper flags for activating CUDA support.

It may happen that eventhough your MPI implementation is actually cuda-aware, cmake MPI detection (through a call to `find_package(MPI)` macro) may fails to detect if your MPI implementation is cuda aware. In that case, you can enforce cuda awareness by turning option `PPKMHD_USE_MPI_CUDA_AWARE_ENFORCED` to ON.

You don't need to use mpi compiler wrapper mpicxx, cmake *should* be able to correctly populate `MPI_CXX_INCLUDE_PATH`, `MPI_CXX_LIBRARIES` which are passed to all final targets by using the alias library `MPI::MPI_CXX`.

* Create a build directory, configure and make

```shell
mkdir build; cd build
cmake -DPPKMHD_USE_MPI=ON -DPPKMHD_BACKEND=Cuda -DKokkos_ARCH_TURING75=ON ..
make -j 4
```

Example command line to run the application (1 GPU used per MPI task)

```shell
mpirun -np 4 ./ppkMHD ./test_implode_2D_mpi.ini
```

### Additionnal features

In order to activate building SDM (Spectral Difference Method) schemes, use cmake option `-DPPKMHD_USE_SDM=ON`.

The MOOD numerical scheme require some linear algebra (QR decomposition) on the host (not device). This is done using a Blas/Lapack implementation using the C language interface named [Lapacke](https://netlib.org/lapack/lapacke.html).

Please note that Atlas doesn't provide Lapackage.
Currently (March 2017), on Ubuntu 16.04, package libatlas-dev is not compatible with package Lapacke (generate errors at link time). So please either [Netlib/Lapack](https://netlib.org/lapack/) or [OpenBLAS](https://www.openblas.net/) implementation.

If you want to enforce the use of OpenBLAS, just use a recent cmake (>=3.6) and add `-DBLA_VENDOR` on the cmake command line. This will tell the cmake build system (through the call to find_package(BLAS) ) to only look for the OpenBLAS implementation.

On a recent Ubuntu, if atlas is not installed, but OpenBLAS is, you don't need to have a bleeding edge cmake, current cmake will find OpenBLAS.

## See also

* [ppkMHD Wiki](https://github.com/pkestene/ppkMHD/wiki)
* [Implementing Spectral Difference Methods (SDM) for Compressible Euler flow simulations using performance portable library kokkos](https://www.researchgate.net/publication/326400645_Implementing_Spectral_Difference_Methods_SDM_for_Compressible_Euler_flow_simulations_using_performance_portable_library_kokkos)
* [Implementing Spectral Difference Methods (SDM) for Compressible Euler flow simulations using performance portable library kokkos (astrosim 2018)](https://www.researchgate.net/publication/328175816_Implementing_Spectral_Difference_Methods_SDM_for_Compressible_Euler_flow_simulations_using_performance_portable_library_kokkos)
