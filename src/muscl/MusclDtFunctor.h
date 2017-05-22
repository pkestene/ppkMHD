#ifndef MUSCL_DT_FUNCTOR_H_
#define MUSCL_DT_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
class ComputeDtFunctor3D : public HydroBaseFunctor<3> {

public:
  
  ComputeDtFunctor3D(HydroParams params,
		     DataArray3d Udata) :
    HydroBaseFunctor<3>(params),
    Udata(Udata)  {};

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  /* this is a reduce (max) functor */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &index, real_t &invDt) const
  {
    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int ghostWidth = params.ghostWidth;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;
    
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth &&
       j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy, vz;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,k,ID);
      uLoc[IP] = Udata(i,j,k,IP);
      uLoc[IU] = Udata(i,j,k,IU);
      uLoc[IV] = Udata(i,j,k,IV);
      uLoc[IW] = Udata(i,j,k,IW);

      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc);
      vx = c+FABS(qLoc[IU]);
      vy = c+FABS(qLoc[IV]);
      vz = c+FABS(qLoc[IW]);

      invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
      
    }
	    
  } // operator ()


  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  DataArray3d Udata;
  
}; // ComputeDtFunctor3D



} // namespace muscl

#endif // MUSCL_DT_FUNCTOR_H_
