#ifndef SDM_VISCOUS_FLUX_FUNCTORS_H_
#define SDM_VISCOUS_FLUX_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "sdm/SDMBaseFunctor.h"

#include "sdm/SDM_Geometry.h"
#include "sdm/sdm_shared.h" // for DofMap

#include "shared/RiemannSolvers.h"
#include "shared/EulerEquations.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes viscous fluxes at fluxes points taking
 * as input conservative variables and their gradients at fluxes points.
 */
template<int dim, int N, int dir>
class ComputeViscousFluxAtFluxPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  ComputeViscousFluxAtFluxPoints_Functor(HydroParams                 params,
					 SDM_Geometry<dim,N>         sdm_geom,
					 ppkMHD::EulerEquations<dim> euler,
					 DataArray                   UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    euler(euler),
    UdataFlux(UdataFlux)
  {};

  // ================================================
  //
  // 2D version.
  //
  // ================================================
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    // state variable for conservative variables, and flux
    HydroState q, flux;

    /*
     * first handle interior point, then end points.
     */
    
    // =========================
    // ========= DIR X =========
    // =========================
    // loop over cell DoF's
    if (dir == IX) {
      
      for (int idy=0; idy<N; ++idy) {
	
	// interior points along direction X
	for (int idx=1; idx<N; ++idx) {
	  
	  // retrieve state conservative variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	    q[ivar] = UdataFlux(i,j, dofMapF(idx,idy,0,ivar));

	  }

	  // compute pressure
	  real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	  // compute flux along X direction
	  euler.flux_x(q, p, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idx
	
      } // end for idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (i>0 and i<isize) {
	
	for (int idy=0; idy<N; ++idy) {
	  
	  // conservative state
	  HydroState qL, qR;
	  
	  // primitive state
	  HydroState wL, wR;
	  
	  HydroState qgdnv;
	  
	  // when idx == 0, get right and left state
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qL[ivar] = UdataFlux(i-1,j, dofMapF(N,idy,0,ivar));
	    qR[ivar] = UdataFlux(i  ,j, dofMapF(0,idy,0,ivar));
	  }
	  
	  // convert to primitive
	  euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	  euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
	  // riemann solver
	  ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy back result in current cell and in neighbor
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    UdataFlux(i-1,j, dofMapF(N,idy,0,ivar)) = flux[ivar];
	    UdataFlux(i  ,j, dofMapF(0,idy,0,ivar)) = flux[ivar];
	  }
	  	  
	} // end for idy

      } // end safe-guard
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    // loop over cell DoF's
    if (dir == IY) {
      
      for (int idx=0; idx<N; ++idx) {
	
	// interior points along direction X
	for (int idy=1; idy<N; ++idy) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	    q[ivar] = UdataFlux(i,j, dofMapF(idx,idy,0,ivar));

	  }

	  // compute pressure
	  real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	  // compute flux along Y direction
	  euler.flux_y(q, p, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idy
	
      } // end for idx

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j>0 and j<jsize) {
	
	for (int idx=0; idx<N; ++idx) {
	  
	  // conservative state
	  HydroState qL, qR;
	  
	  // primitive state
	  HydroState wL, wR;
	  
	  HydroState qgdnv;
	  
	  // when idy == 0, get right and left state
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qL[ivar] = UdataFlux(i,j-1, dofMapF(idx,N,0,ivar));
	    qR[ivar] = UdataFlux(i,j  , dofMapF(idx,0,0,ivar));
	  }
	  
	  // convert to primitive : q -> w
	  euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	  euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
	  // riemann solver
	  this->swap( wL[IU], wL[IV] );
	  this->swap( wR[IU], wR[IV] );
	  ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy back results in current cell as well as in neighbor
	  UdataFlux(i,j  , dofMapF(idx,0,0,ID)) = flux[ID];
	  UdataFlux(i,j  , dofMapF(idx,0,0,IE)) = flux[IE];
	  UdataFlux(i,j  , dofMapF(idx,0,0,IU)) = flux[IV]; // swap again
	  UdataFlux(i,j  , dofMapF(idx,0,0,IV)) = flux[IU]; // swap again

	  UdataFlux(i,j-1, dofMapF(idx,N,0,ID)) = flux[ID];
	  UdataFlux(i,j-1, dofMapF(idx,N,0,IE)) = flux[IE];
	  UdataFlux(i,j-1, dofMapF(idx,N,0,IU)) = flux[IV]; // swap again
	  UdataFlux(i,j-1, dofMapF(idx,N,0,IV)) = flux[IU]; // swap again
	  
	} // end for idx

      } // end safe-guard

    } // end for dir IY

  } // 2d
  
  ppkMHD::EulerEquations<dim> euler;
  DataArray UdataFlux;

}; // class ComputeViscousFluxAtFluxPoints_Functor

} // namespace sdm

#endif // SDM_VISCOUS_FLUX_FUNCTORS_H_
