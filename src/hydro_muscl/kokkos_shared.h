#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include "real_type.h"

#ifdef CUDA
# define DEVICE Kokkos::Cuda
#include <cuda.h>
#endif

#ifdef OPENMP
# define DEVICE Kokkos::OpenMP
#endif

#ifndef DEVICE
# define DEVICE Kokkos::OpenMP
#endif

// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)
typedef Kokkos::View<real_t**, DEVICE>   DataArray2d;
typedef DataArray2d::HostMirror          DataArray2dHost;

typedef Kokkos::View<real_t***, DEVICE>  DataArray3d;
typedef DataArray3d::HostMirror          DataArray3dHost;

/**
 * Retrieve cartesian coordinate from index, using memory layout information.
 *
 * for each execution space define a prefered layout.
 * Prefer left layout  for CUDA execution space.
 * Prefer right layout for OpenMP execution space.
 */
KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny) {
#ifdef CUDA
  j = index / Nx;
  i = index - j*Nx;
#else
  i = index / Ny;
  j = index - i*Ny;
#endif
}

KOKKOS_INLINE_FUNCTION
int coord2index(int i, int j, int Nx, int Ny) {
#ifdef CUDA
  return i + Nx*j; // left layout
#else
  return j + Ny*i; // right layout
#endif
}


#endif // KOKKOS_SHARED_H_
