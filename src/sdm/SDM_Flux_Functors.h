#ifndef SDM_FLUX_FUNCTORS_H_
#define SDM_FLUX_FUNCTORS_H_

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
 * This functor compute fluxes at fluxes points taking
 * as input conservative variables at fluxes points.
 *
 */
template<int dim, int N, int dir>
class ComputeFluxInterior_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  ComputeFluxInterior_Functor(HydroParams         params,
			      SDM_Geometry<dim,N> sdm_geom,
			      DataArray           UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
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

  } // 2d

  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // state variable for conservative variables, and flux
    HydroState q, flux;
    
    // loop over cell DoF's
    if (dir == IX) {

      for (int idy=0; idy<N; ++idy) {

	// interior
	for (int idx=1; idx<N; ++idx) {

	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	    q[ivar] = UdataFlux(i,j, dofMapF(idx,idy,0,ivar));

	  }

	  // compute pressure
	  p = euler->compute_pressure(q);
	  
	  // compute flux along X direction
	  euler->flux_x(q, p, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 1; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[idx];
	    
	  } // end for ivar
	  
	} // end for ivar
	
      } // end for idy

    } // end for dir IX

    
  } // 3d

  DataArray UdataFlux;

}; // class

} // namespace sdm

#endif // SDM_FLUX_FUNCTORS_H_
