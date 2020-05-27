#ifndef MOOD_DT_FUNCTOR_H_
#define MOOD_DT_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/mood_shared.h"
#include "mood/Polynomial.h"
#include "mood/MoodBaseFunctor.h"
#include "mood/QuadratureRules.h"

namespace mood {

// =======================================================================
// =======================================================================
// template<int dim,
// 	 int degree>
// class ComputeDtFunctor : public MoodBaseFunctor<dim,degree>
// {

// public:
//   using typename MoodBaseFunctor<dim,degree>::DataArray;
//   using typename MoodBaseFunctor<dim,degree>::HydroState;

//   /**
//    * Constructor for 2D/3D.
//    */
//   ComputeDtFunctor(HydroParams params,
// 		   DataArray Udata) :
//     MoodBaseFunctor<dim,degree>(params),
//     Udata(Udata)
//   {};

//   // Tell each thread how to initialize its reduction result.
//   KOKKOS_INLINE_FUNCTION
//   void init (real_t& dst) const
//   {
//     // The identity under max is -Inf.
//     // Kokkos does not come with a portable way to access
//     // floating-point Inf and NaN. 
// #ifdef __CUDA_ARCH__
//     dst = -CUDART_INF;
// #else
//     dst = std::numeric_limits<real_t>::min();
// #endif // __CUDA_ARCH__
//   } // init

//   /* this is a reduce (max) functor  for 2d data */
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename
// 		  std::enable_if<dim_==2, int>::type& index,
// 		  real_t &invDt) const
//   {
//     const int isize = this->params.isize;
//     const int jsize = this->params.jsize;
//     const int ghostWidth = this->params.ghostWidth;
//     //const int nbvar = this->params.nbvar;
//     const real_t dx = this->params.dx;
//     const real_t dy = this->params.dy;
    
//     int i,j;
//     index2coord(index,i,j,isize,jsize);

//     if(j >= ghostWidth && j < jsize - ghostWidth &&
//        i >= ghostWidth && i < isize - ghostWidth) {
      
//       HydroState uLoc; // conservative    variables in current cell
//       HydroState qLoc; // primitive    variables in current cell
//       real_t c=0.0;
//       real_t vx, vy;
      
//       // get local conservative variable
//       uLoc[ID] = Udata(i,j,ID);
//       uLoc[IP] = Udata(i,j,IP);
//       uLoc[IU] = Udata(i,j,IU);
//       uLoc[IV] = Udata(i,j,IV);

//       // get primitive variables in current cell
//       this->computePrimitives(uLoc, &c, qLoc);
//       vx = c+FABS(qLoc[IU]);
//       vy = c+FABS(qLoc[IV]);

//       invDt = FMAX(invDt, vx/dx + vy/dy);
      
//     }
	    
//   } // operator () for 2d

//   /* this is a reduce (max) functor for 3d data */
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename
// 		  std::enable_if<dim_==3, int>::type& index,
// 		  real_t &invDt) const
//   {
//     const int isize = this->params.isize;
//     const int jsize = this->params.jsize;
//     const int ksize = this->params.ksize;
//     const int ghostWidth = this->params.ghostWidth;

//     const real_t dx = this->params.dx;
//     const real_t dy = this->params.dy;
//     const real_t dz = this->params.dz;
    
//     int i,j,k;
//     index2coord(index,i,j,k,isize,jsize,ksize);

//     if(k >= ghostWidth && k < ksize - ghostWidth &&
//        j >= ghostWidth && j < jsize - ghostWidth &&
//        i >= ghostWidth && i < isize - ghostWidth) {
      
//       HydroState uLoc; // conservative    variables in current cell
//       HydroState qLoc; // primitive    variables in current cell
//       real_t c=0.0;
//       real_t vx, vy, vz;
      
//       // get local conservative variable
//       uLoc[ID] = Udata(i,j,k,ID);
//       uLoc[IP] = Udata(i,j,k,IP);
//       uLoc[IU] = Udata(i,j,k,IU);
//       uLoc[IV] = Udata(i,j,k,IV);
//       uLoc[IW] = Udata(i,j,k,IW);

//       // get primitive variables in current cell
//       this->computePrimitives(uLoc, &c, qLoc);
//       vx = c+FABS(qLoc[IU]);
//       vy = c+FABS(qLoc[IV]);
//       vz = c+FABS(qLoc[IW]);

//       invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
      
//     }
	    
//   } // operator () for 3d

//   // "Join" intermediate results from different threads.
//   // This should normally implement the same reduction
//   // operation as operator() above. Note that both input
//   // arguments MUST be declared volatile.
//   KOKKOS_INLINE_FUNCTION
//   void join (volatile real_t& dst,
// 	     const volatile real_t& src) const
//   {
//     // max reduce
//     if (dst < src) {
//       dst = src;
//     }
//   } // join
  
//   DataArray Udata;
  
// }; // ComputeDtFunctor

// =======================================================================
// =======================================================================
template<int degree>
class ComputeDtFunctor2d : public MoodBaseFunctor<2,degree>
{

public:
  using typename MoodBaseFunctor<2,degree>::DataArray;
  using typename MoodBaseFunctor<2,degree>::HydroState;
  using MonomMap = typename mood::MonomialMap<2,degree>::MonomMap;

  /**
   * Constructor for 2D
   */
  ComputeDtFunctor2d(HydroParams params,
		     MonomMap    monomMap,
		     DataArray   Udata) :
    MoodBaseFunctor<2,degree>(params,monomMap),
    Udata(Udata)
  {};

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

  /* this is a reduce (max) functor  for 2d data */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index,
		  real_t &invDt) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;
    //const int nbvar = this->params.nbvar;
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive    variables in current cell
      real_t c=0.0;
      real_t vx, vy;
      
      // get local conservative variable
      uLoc[ID] = Udata(i,j,ID);
      uLoc[IP] = Udata(i,j,IP);
      uLoc[IU] = Udata(i,j,IU);
      uLoc[IV] = Udata(i,j,IV);

      // get primitive variables in current cell
      this->computePrimitives(uLoc, &c, qLoc);
      vx = c+FABS(qLoc[IU]);
      vy = c+FABS(qLoc[IV]);

      invDt = FMAX(invDt, vx/dx + vy/dy);
      
    }
	    
  } // operator () for 2d

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
  
  DataArray Udata;
  
}; // ComputeDtFunctor2d

// =======================================================================
// =======================================================================
template<int degree>
class ComputeDtFunctor3d : public MoodBaseFunctor<3,degree>
{

public:
  using typename MoodBaseFunctor<3,degree>::DataArray;
  using typename MoodBaseFunctor<3,degree>::HydroState;
  using MonomMap = typename mood::MonomialMap<3,degree>::MonomMap;

  /**
   * Constructor for 3D.
   */
  ComputeDtFunctor3d(HydroParams params,
		     MonomMap    monomMap,
		     DataArray   Udata) :
    MoodBaseFunctor<3,degree>(params,monomMap),
    Udata(Udata)
  {};

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

  /* this is a reduce (max) functor for 3d data */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index,
		  real_t &invDt) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;
    
    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    const real_t dz = this->params.dz;
    
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
      this->computePrimitives(uLoc, &c, qLoc);
      vx = c+FABS(qLoc[IU]);
      vy = c+FABS(qLoc[IV]);
      vz = c+FABS(qLoc[IW]);

      invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
      
    }
	    
  } // operator () for 3d

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
  
  DataArray Udata;
  
}; // ComputeDtFunctor3d

} // namespace

#endif // MOOD_DT_FUNCTOR_H_
