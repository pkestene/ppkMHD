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

  using Vector     = typename ppkMHD::EulerEquations<dim>::Vector;
  using GradTensor = typename ppkMHD::EulerEquations<dim>::GradTensor;
  
  using SDMBaseFunctor<dim,N>::IGU;
  using SDMBaseFunctor<dim,N>::IGV;
  using SDMBaseFunctor<dim,N>::IGW;

  using SDMBaseFunctor<dim,N>::IGUX;
  using SDMBaseFunctor<dim,N>::IGVX;
  using SDMBaseFunctor<dim,N>::IGWX;

  using SDMBaseFunctor<dim,N>::IGUY;
  using SDMBaseFunctor<dim,N>::IGVY;
  using SDMBaseFunctor<dim,N>::IGWY;

  using SDMBaseFunctor<dim,N>::IGUZ;
  using SDMBaseFunctor<dim,N>::IGVZ;
  using SDMBaseFunctor<dim,N>::IGWZ;
  
  //using SDMBaseFunctor<dim,N>::IGT;

  using SDMBaseFunctor<dim,N>::U_X;
  using SDMBaseFunctor<dim,N>::U_Y;
  using SDMBaseFunctor<dim,N>::U_Z;

  using SDMBaseFunctor<dim,N>::V_X;
  using SDMBaseFunctor<dim,N>::V_Y;
  using SDMBaseFunctor<dim,N>::V_Z;

  using SDMBaseFunctor<dim,N>::W_X;
  using SDMBaseFunctor<dim,N>::W_Y;
  using SDMBaseFunctor<dim,N>::W_Z;

  /**
   *
   * \param[in] euler data structure to compute local viscous flux
   * \param[in] FUgrad array of velocity and velocity gradients at flux points
   * \param[out] UdataFlux array of viscous fluxes at flux points
   *
   */
  ComputeViscousFluxAtFluxPoints_Functor(HydroParams                 params,
					 SDM_Geometry<dim,N>         sdm_geom,
					 ppkMHD::EulerEquations<dim> euler,
					 DataArray                   FUgrad,
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

    // velocity vector
    Vector vel;

    // forcing term (unused for now)
    Vector forcing = {0.0, 0.0};
    
    // velocity gradient tensor
    GradTensor grad;
    
    // local viscous flux at current flux point
    HydroState flux;
    
    // =========================
    // ========= DIR X =========
    // =========================
    // loop over cell DoF's
    if (dir == IX) {
      
      for (int idy=0; idy<N; ++idy) {
	
	// all flux points along direction X
	for (int idx=0; idx<N+1; ++idx) {
	  
	  // retrieve velocity vector
	  vel[IX] = FUgrad(i,j, dofMapF(idx,idy,0,IGU));
	  vel[IY] = FUgrad(i,j, dofMapF(idx,idy,0,IGV));

	  // retrieve velocity gradient tensor
	  grad[U_X] = FUgrad(i,j, dofMapF(idx,idy,0,IGUX));
	  grad[U_Y] = FUgrad(i,j, dofMapF(idx,idy,0,IGUY));
	  grad[V_X] = FUgrad(i,j, dofMapF(idx,idy,0,IGVX));
	  grad[V_Y] = FUgrad(i,j, dofMapF(idx,idy,0,IGVY));
	  	  
	  // compute flux along X direction
	  euler.flux_visc_x(grad, vel, forcing, this->params.settings.mu, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idx
	
      } // end for idy
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    // loop over cell DoF's
    if (dir == IY) {
      
      for (int idx=0; idx<N; ++idx) {
	
	// interior points along direction X
	for (int idy=0; idy<N+1; ++idy) {
	  
	  // retrieve velocity vector
	  vel[IX] = FUgrad(i,j, dofMapF(idx,idy,0,IGU));
	  vel[IY] = FUgrad(i,j, dofMapF(idx,idy,0,IGV));

	  // retrieve velocity gradient tensor
	  grad[U_X] = FUgrad(i,j, dofMapF(idx,idy,0,IGUX));
	  grad[U_Y] = FUgrad(i,j, dofMapF(idx,idy,0,IGUY));
	  grad[V_X] = FUgrad(i,j, dofMapF(idx,idy,0,IGVX));
	  grad[V_Y] = FUgrad(i,j, dofMapF(idx,idy,0,IGVY));
	  	  
	  // compute flux along Y direction
	  euler.flux_visc_y(grad, vel, forcing, this->params.settings.mu, flux);
	  
	  // copy back interpolated value
	  for (int ivar = 0; ivar<nbvar; ++ivar) {
	    
	    UdataFlux(i,j, dofMapF(idx,idy,0,ivar)) = flux[ivar];
	    
	  } // end for ivar
	  
	} // end for idy
	
      } // end for idx

    } // end for dir IY

  } // 2d
  
  // ================================================
  //
  // 3D version.
  //
  // ================================================
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;

    const int nbvar = this->params.nbvar;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    // velocity vector
    Vector vel;
    
    // forcing term (unused for now)
    Vector forcing = {0.0, 0.0};
    
    // velocity gradient tensor
    GradTensor grad;
    
    // local viscous flux at current flux point
    HydroState flux;

    // =========================
    // ========= DIR X =========
    // =========================
    // loop over cell DoF's
    if (dir == IX) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	
	  // all flux points along direction X
	  for (int idx=0; idx<N+1; ++idx) {
	  
	    // retrieve velocity vector
	    vel[IX] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGU));
	    vel[IY] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGV));
	    vel[IZ] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGW));

	    // retrieve velocity gradient tensor
	    grad[U_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUX));
	    grad[U_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUY));
	    grad[U_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUZ));

	    grad[V_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVX));
	    grad[V_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVY));
	    grad[V_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVZ));
	  	  
	    grad[W_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWX));
	    grad[W_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWY));
	    grad[W_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWZ));
	  	  
	    // compute flux along X direction
	    euler.flux_visc_x(grad, vel, forcing, this->params.settings.mu, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idx
	
	} // end for idy
      } // end for idz
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    // loop over cell DoF's
    if (dir == IY) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	
	  // all flux points along direction Y
	  for (int idy=0; idy<N+1; ++idy) {
	  
	    // retrieve velocity vector
	    vel[IX] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGU));
	    vel[IY] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGV));
	    vel[IZ] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGW));

	    // retrieve velocity gradient tensor
	    grad[U_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUX));
	    grad[U_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUY));
	    grad[U_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUZ));

	    grad[V_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVX));
	    grad[V_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVY));
	    grad[V_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVZ));
	  	  
	    grad[W_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWX));
	    grad[W_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWY));
	    grad[W_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWZ));
	  	  
	    // compute flux along Y direction
	    euler.flux_visc_y(grad, vel, forcing, this->params.settings.mu, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idx
	
	} // end for idy
      } // end for idz
      
    } // end for dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    // loop over cell DoF's
    if (dir == IZ) {
      
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	
	  // all flux points along direction Z
	  for (int idz=0; idz<N+1; ++idz) {
	  
	    // retrieve velocity vector
	    vel[IX] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGU));
	    vel[IY] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGV));
	    vel[IZ] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGW));

	    // retrieve velocity gradient tensor
	    grad[U_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUX));
	    grad[U_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUY));
	    grad[U_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGUZ));

	    grad[V_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVX));
	    grad[V_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVY));
	    grad[V_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGVZ));
	  	  
	    grad[W_X] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWX));
	    grad[W_Y] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWY));
	    grad[W_Z] = FUgrad(i,j,k, dofMapF(idx,idy,idz,IGWZ));
	  	  
	    // compute flux along Z direction
	    euler.flux_visc_z(grad, vel, forcing, this->params.settings.mu, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idx
	
	} // end for idy
      } // end for idz
      
    } // end for dir IZ

  } // 3d


  ppkMHD::EulerEquations<dim> euler;
  DataArray FUgrad;
  DataArray UdataFlux;

}; // class ComputeViscousFluxAtFluxPoints_Functor

} // namespace sdm

#endif // SDM_VISCOUS_FLUX_FUNCTORS_H_
