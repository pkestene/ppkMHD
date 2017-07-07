#ifndef SDM_RUN_FUNCTORS_H_
#define SDM_RUN_FUNCTORS_H_

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
 * This functor takes as an input conservative variables
 * at solution points and perform interpolation at flux points.
 *
 * Perform exactly the inverse of Interpolate_At_FluxPoints_Functor.
 *
 */
template<int dim, int N, int dir>
class Interpolate_At_FluxPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  Interpolate_At_FluxPoints_Functor(HydroParams         params,
				    SDM_Geometry<dim,N> sdm_geom,
				    DataArray           UdataSol,
				    DataArray           UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    UdataFlux(UdataFlux)
  {};

  
  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get solution values vector along X direction
	  for (int idx=0; idx<N; ++idx) {
	  
	    sol[idx] = UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar));

	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar)) = flux[idx];
	    
	  }

	  if(i==1 and j==1 and ivar==ID) {
	    printf("DEBUG sol %d | ",idy);
	    for (int kk=0; kk<N; ++kk) {
	      printf(" %f",sol[kk]);
	    }
	    printf("\n");
	  }
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get solution values vector along Y direction
	  for (int idy=0; idy<N; ++idy) {
	  
	    sol[idy] = UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N+1; ++idy) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar)) = flux[idy];
	    
	  }
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir

    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
  } // end operator () - 3d
  
  DataArray UdataSol, UdataFlux;

}; // Interpolate_At_FluxPoints_Functor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor takes as an input variables
 * at flux points and perform interpolation at solution points.
 *
 * Perform exactly the inverse of Interpolate_At_FluxPoints_Functor
 */
template<int dim, int N, int dir>
class Interpolate_At_SolutionPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;
  
  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  Interpolate_At_SolutionPoints_Functor(HydroParams         params,
					SDM_Geometry<dim,N> sdm_geom,
					DataArray           UdataFlux,
					DataArray           UdataSol) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataFlux(UdataFlux),
    UdataSol(UdataSol)
  {};

  
  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    solution_values_t sol;
    flux_values_t     flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get values at flux point along X direction
	  for (int idx=0; idx<N+1; ++idx) {
	  
	    flux[idx] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  this->flux2sol_vector(flux, sol);
	  
	  // copy back interpolated value
	  for (int idx=0; idx<N; ++idx) {
	    
	    UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) = sol[idx];
	    
	  }
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir

    // loop over cell DoF's
    if (dir == IY) {

      for (int idx=0; idx<N; ++idx) {

	// for each variables
	for (int ivar = 0; ivar<nbvar; ++ivar) {
	
	  // get values at flux point along Y direction
	  for (int idy=0; idy<N+1; ++idy) {
	  
	    flux[idy] = UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar));
	    
	  }
	  
	  // interpolate at flux points for this given variable
	  this->flux2sol_vector(flux, sol);
	  
	  // copy back interpolated value
	  for (int idy=0; idy<N; ++idy) {
	    
	    UdataSol(i  ,j  , dofMapS(idx,idy,0,ivar)) = sol[idy];
	    
	  }
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir

    
  } // end operator () - 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
  } // end operator () - 3d
  
  DataArray UdataFlux, UdataSol;

}; // Interpolate_At_SolutionPoints_Functor

} // namespace sdm

#endif //SDM_RUN_FUNCTORS_H_

