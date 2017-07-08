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
 * We first compute fluxes at flux points interior to cell, then 
 * handle the end points.
 */
template<int dim, int N, int dir>
class ComputeFluxAtFluxPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
  ComputeFluxAtFluxPoints_Functor(HydroParams                 params,
				  SDM_Geometry<dim,N>         sdm_geom,
				  ppkMHD::EulerEquations<dim> euler,
				  DataArray                   UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    euler(euler),
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

    // state variable for conservative variables, and flux
    HydroState q, flux;

    /*
     * first handle interior point, then end points.
     */
    
    // loop over cell DoF's
    if (dir == IX) {
      
      for (int idy=0; idy<N; ++idy) {
	
	// interior points along direction X
	for (int idx=1; idx<N; ++idx) {
	  
	  // for each variables
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	    q[ivar] = UdataFlux(i,j, dofMapF(idx,idy,0,ivar));

	  }

	  // compute pressure
	  real_t p = euler->compute_pressure(q, this->params.gamma0);
	  
	  // compute flux along X direction
	  euler->flux_x(q, p, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idx
	
      } // end for idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // make sure to exclude ghost cells
      if (i>0 and i<isize-1) {
	
	for (int idy=0; idy<N; ++idy) {
	  
	  // conservative state
	  HydroState qL, qR;
	  
	  // primitive state
	  HydroState wL, wR;
	  
	  HydroState qgdnv;
	  
	  // when idx == 0, get right and left state
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qR[ivar] = UdataFlux(i  ,j, dofMapF(0,idy,0,ivar));
	    qL[ivar] = UdataFlux(i-1,j, dofMapF(N,idy,0,ivar));
	  }
	  
	  // convert to primitive
	  euler->convert_to_primitive(qR,wR,this->params.gamma0);
	  euler->convert_to_primitive(qL,wL,this->params.gamma0);
	  
	  // riemann solver
	  riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy flux
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    UdataFlux(i,j, dofMapF(0,idy,0,ivar)) = flux[ivar];
	  }
	  
	  // when idx == N
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qR[ivar] = UdataFlux(i+1,j, dofMapF(0,idy,0,ivar));
	    qL[ivar] = UdataFlux(i  ,j, dofMapF(N,idy,0,ivar));
	  }

	  // convert to primitive
	  euler->convert_to_primitive(qR,wR,this->params.gamma0);
	  euler->convert_to_primitive(qL,wL,this->params.gamma0);
	  
	  // riemann solver
	  swapValues( &(wL[IU]), &(wL[IV]) );
	  swapValues( &(wR[IU]), &(wR[IV]) );
	  riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy flux
	  UdataFlux(i,j, dofMapF(N,idy,0,ID)) = flux[ID];
	  UdataFlux(i,j, dofMapF(N,idy,0,IE)) = flux[IE];
	  UdataFlux(i,j, dofMapF(N,idy,0,IU)) = flux[IV];
	  UdataFlux(i,j, dofMapF(N,idy,0,IV)) = flux[IU];
	  
	} // end for idy

      } // end safe-guard
      
    } // end for dir IX

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
	  real_t p = euler->compute_pressure(q, this->params.gamma0);
	  
	  // compute flux along Y direction
	  euler->flux_y(q, p, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idy
	
      } // end for idx

    } // end for dir IY

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
        
  } // 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray UdataFlux;

}; // class ComputeFluxInterior_Functor


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * \class ComputeFluxExterior_Functor
 */
// template<int dim, int N, int dir>
// class ComputeFluxExterior_Functor : public SDMBaseFunctor<dim,N> {

// public:
//   using typename SDMBaseFunctor<dim,N>::DataArray;
  
//   static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  
//   ComputeFluxExterior_Functor(HydroParams                 params,
// 			      SDM_Geometry<dim,N>         sdm_geom,
// 			      ppkMHD::EulerEquations<dim> euler,
// 			      DataArray                   UdataFlux) :
//     SDMBaseFunctor<dim,N>(params,sdm_geom),
//     euler(euler),
//     UdataFlux(UdataFlux)
//   {};

//   /*
//    * 2D version.
//    */
//   //! functor for 2d 
//   template<int dim_ = dim>
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index) const
//   {
//     const int isize = this->params.isize;
//     const int jsize = this->params.jsize;
//     const int ghostWidth = this->params.ghostWidth;

//     const int nbvar = this->params.nbvar;

//     // local cell index
//     int i,j;
//     index2coord(index,i,j,isize,jsize);

//   } // 2d

//   ppkMHD::EulerEquations<dim> euler;
//   DataArray UdataFlux;

// }; // class

} // namespace sdm

#endif // SDM_FLUX_FUNCTORS_H_
