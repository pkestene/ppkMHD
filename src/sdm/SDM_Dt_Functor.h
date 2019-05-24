#ifndef SDM_DT_FUNCTOR_H_
#define SDM_DT_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/EulerEquations.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute CFL time-step constraint
 */
// template<int dim, int N>
// class ComputeDt_Functor : public SDMBaseFunctor<dim,N> {

// public:
//   using typename SDMBaseFunctor<dim,N>::DataArray;
//   using typename SDMBaseFunctor<dim,N>::HydroState;
//   //using typename ppkMHD::EulerEquations<dim>;
  
//   //! intra-cell degrees of freedom mapping at solution points
//   static constexpr auto dofMap = DofMap<dim,N>;
  
//   ComputeDt_Functor(HydroParams         params,
// 		    SDM_Geometry<dim,N> sdm_geom,
// 		    ppkMHD::EulerEquations<dim> euler,
// 		    DataArray           Udata) :
//     SDMBaseFunctor<dim,N>(params,sdm_geom),
//     euler(euler),
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

//   // ================================================
//   //
//   // 2D version.
//   //
//   // ================================================
//   //! functor for 2d - CFL constraint
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index,
// 		  real_t &invDt) const
//   {
//     const int isize = this->params.isize;
//     const int jsize = this->params.jsize;
//     const int ghostWidth = this->params.ghostWidth;

//     const int nbvar = this->params.nbvar;

//     const real_t dx = this->params.dx;
//     const real_t dy = this->params.dy;

//     // local cell index
//     int i,j;
//     index2coord(index,i,j,isize,jsize);

//     if(j >= ghostWidth && j < jsize - ghostWidth &&
//        i >= ghostWidth && i < isize - ghostWidth) {

//       HydroState uLoc; // conservative    variables in current cell
//       HydroState qLoc; // primitive       variables in current cell
//       real_t c=0.0;
//       real_t vx, vy;

//       // loop over current cell DoF solution points
//       for (int idy=0; idy<N; ++idy) {
// 	for (int idx=0; idx<N; ++idx) {

// 	  // get local conservative variable
// 	  uLoc[ID] = Udata(i,j, dofMap(idx,idy,0,ID));
// 	  uLoc[IE] = Udata(i,j, dofMap(idx,idy,0,IE));
// 	  uLoc[IU] = Udata(i,j, dofMap(idx,idy,0,IU));
// 	  uLoc[IV] = Udata(i,j, dofMap(idx,idy,0,IV));

// 	  // get primitive variables in current cell
// 	  euler.convert_to_primitive(uLoc,qLoc,this->params.settings.gamma0);

//	  c = euler.compute_speed_of_sound(qLoc,this->params.settings.gamma0);
// 	  vx = c+FABS(qLoc[IU]);
// 	  vy = c+FABS(qLoc[IV]);
	  
// 	  invDt = FMAX(invDt, vx/dx + vy/dy);
	  
// 	} // end for idx
//       } // end for idy
      
//     } // end guard - ghostcells

//   } // end operator () - 2d
  
//   // ================================================
//   //
//   // 3D version.
//   //
//   // ================================================
//   //! functor for 3d 
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index,
// 		  real_t &invDt) const
//   {

//     const int isize = this->params.isize;
//     const int jsize = this->params.jsize;
//     const int ksize = this->params.ksize;
//     const int ghostWidth = this->params.ghostWidth;

//     const int nbvar = this->params.nbvar;

//     const real_t dx = this->params.dx;
//     const real_t dy = this->params.dy;
//     const real_t dz = this->params.dz;
    
//     // local cell index
//     int i,j,k;
//     index2coord(index,i,j,k,isize,jsize,ksize);

//     if(k >= ghostWidth && k < ksize - ghostWidth &&
//        j >= ghostWidth && j < jsize - ghostWidth &&
//        i >= ghostWidth && i < isize - ghostWidth) {
      
//       HydroState uLoc; // conservative    variables in current cell
//       HydroState qLoc; // primitive       variables in current cell
//       real_t c=0.0;
//       real_t vx, vy, vz;
      
//       // loop over current cell DoF solution points
//       for (int idz=0; idz<N; ++idz) {
// 	for (int idy=0; idy<N; ++idy) {
// 	  for (int idx=0; idx<N; ++idx) {
	  
// 	  // get local conservative variable
// 	  uLoc[ID] = Udata(i,j,k, dofMap(idx,idy,idz,ID));
// 	  uLoc[IE] = Udata(i,j,k, dofMap(idx,idy,idz,IE));
// 	  uLoc[IU] = Udata(i,j,k, dofMap(idx,idy,idz,IU));
// 	  uLoc[IV] = Udata(i,j,k, dofMap(idx,idy,idz,IV));
// 	  uLoc[IW] = Udata(i,j,k, dofMap(idx,idy,idz,IW));

// 	  // get primitive variables in current cell
// 	  euler.convert_to_primitive(uLoc,qLoc,this->params.settings.gamma0);

//	  c = euler.compute_speed_of_sound(qLoc,this->params.settings.gamma0);
// 	  vx = c+FABS(qLoc[IU]);
// 	  vy = c+FABS(qLoc[IV]);
// 	  vz = c+FABS(qLoc[IW]);
	  
// 	  invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
	  
// 	  } // end for idx
// 	} // end for idy
//       } // end for idz
      
//     } // end guard - ghostcells
    
//   } // end operator () - 3d

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

//   ppkMHD::EulerEquations<dim> euler;
//   DataArray Udata;

// }; // class ComputeDt_Functor 

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute CFL time-step constraint
 */
template<int N>
class ComputeDt_Functor_2d : public SDMBaseFunctor<2,N> {

public:
  using typename SDMBaseFunctor<2,N>::DataArray;
  using typename SDMBaseFunctor<2,N>::HydroState;
  //using typename ppkMHD::EulerEquations<2>;
  
  //! intra-cell degrees of freedom mapping at solution points
  static constexpr auto dofMap = DofMap<2,N>;
  
  ComputeDt_Functor_2d(HydroParams         params,
		       SDM_Geometry<2,N> sdm_geom,
		       ppkMHD::EulerEquations<2> euler,
		       DataArray           Udata) :
    SDMBaseFunctor<2,N>(params,sdm_geom),
    euler(euler),
    Udata(Udata)
  {};

  // static method which does it all: create and execute functor
  static real_t apply(HydroParams               params,
                      SDM_Geometry<2,N>         sdm_geom,
                      ppkMHD::EulerEquations<2> euler,
                      DataArray                 Udata)
  {
    int64_t nbCells = params.isize * params.jsize;

    real_t invDt = 0;
    ComputeDt_Functor_2d<N> functor(params, sdm_geom, euler, Udata);
    Kokkos::parallel_reduce(nbCells, functor, invDt);
    return invDt;
  }

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

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d - CFL constraint
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index,
		  real_t &invDt) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    //const int nbvar = this->params.nbvar;

    // to take into account the N DoF per direction per cell
    // we divide dx,dy by N
    const real_t dx = this->params.dx/N;
    const real_t dy = this->params.dy/N;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    if(j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {

      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive       variables in current cell
      real_t c=0.0;
      real_t vx, vy;

      // loop over current cell DoF solution points
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {

	  // get local conservative variable
	  uLoc[ID] = Udata(i,j, dofMap(idx,idy,0,ID));
	  uLoc[IE] = Udata(i,j, dofMap(idx,idy,0,IE));
	  uLoc[IU] = Udata(i,j, dofMap(idx,idy,0,IU));
	  uLoc[IV] = Udata(i,j, dofMap(idx,idy,0,IV));

	  // get primitive variables in current cell
	  euler.convert_to_primitive(uLoc,qLoc,this->params.settings.gamma0);

	  c = euler.compute_speed_of_sound(qLoc,this->params.settings.gamma0);
	  
	  vx = c+FABS(qLoc[IU]);
	  vy = c+FABS(qLoc[IV]);
	  
	  invDt = FMAX(invDt, vx/dx + vy/dy);
	  
	} // end for idx
      } // end for idy
      
    } // end guard - ghostcells

  } // end operator () - 2d
  
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

  ppkMHD::EulerEquations<2> euler;
  DataArray Udata;

}; // class ComputeDt_Functor_2d

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute CFL time-step constraint
 */
template<int N>
class ComputeDt_Functor_3d : public SDMBaseFunctor<3,N> {

public:
  using typename SDMBaseFunctor<3,N>::DataArray;
  using typename SDMBaseFunctor<3,N>::HydroState;
  //using typename ppkMHD::EulerEquations<3>;
  
  //! intra-cell degrees of freedom mapping at solution points
  static constexpr auto dofMap = DofMap<3,N>;
  
  ComputeDt_Functor_3d(HydroParams         params,
		       SDM_Geometry<3,N> sdm_geom,
		       ppkMHD::EulerEquations<3> euler,
		       DataArray           Udata) :
    SDMBaseFunctor<3,N>(params,sdm_geom),
    euler(euler),
    Udata(Udata)
  {};

  // static method which does it all: create and execute functor
  static real_t apply(HydroParams               params,
                      SDM_Geometry<3,N>         sdm_geom,
                      ppkMHD::EulerEquations<3> euler,
                      DataArray                 Udata)
  {
    int64_t nbCells = params.isize * params.jsize * params.ksize;

    real_t invDt = 0;
    ComputeDt_Functor_3d<N> functor(params, sdm_geom, euler, Udata);
    Kokkos::parallel_reduce(nbCells, functor, invDt);
    return invDt;
  }

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

  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d 
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index,
		  real_t &invDt) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    //const int nbvar = this->params.nbvar;

    // to take into account the N DoF per direction per cell
    // we divide dx,dy,dz by N
    const real_t dx = this->params.dx/N;
    const real_t dy = this->params.dy/N;
    const real_t dz = this->params.dz/N;
    
    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth &&
       j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
      
      HydroState uLoc; // conservative    variables in current cell
      HydroState qLoc; // primitive       variables in current cell
      real_t c=0.0;
      real_t vx, vy, vz;
      
      // loop over current cell DoF solution points
      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	  
	    // get local conservative variable
	    uLoc[ID] = Udata(i,j,k, dofMap(idx,idy,idz,ID));
	    uLoc[IE] = Udata(i,j,k, dofMap(idx,idy,idz,IE));
	    uLoc[IU] = Udata(i,j,k, dofMap(idx,idy,idz,IU));
	    uLoc[IV] = Udata(i,j,k, dofMap(idx,idy,idz,IV));
	    uLoc[IW] = Udata(i,j,k, dofMap(idx,idy,idz,IW));
	    
	    // get primitive variables in current cell
	    euler.convert_to_primitive(uLoc,qLoc,this->params.settings.gamma0);
	    
	    c = euler.compute_speed_of_sound(qLoc,this->params.settings.gamma0);
	    
	    vx = c+FABS(qLoc[IU]);
	    vy = c+FABS(qLoc[IV]);
	    vz = c+FABS(qLoc[IW]);
	    
	    invDt = FMAX(invDt, vx/dx + vy/dy + vz/dz);
	    
	  } // end for idx
	} // end for idy
      } // end for idz
      
    } // end guard - ghostcells
    
  } // end operator () - 3d

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

  ppkMHD::EulerEquations<3> euler;
  DataArray Udata;

}; // class ComputeDt_Functor_3d

} // namespace sdm

#endif // SDM_DT_FUNCTOR_H_
