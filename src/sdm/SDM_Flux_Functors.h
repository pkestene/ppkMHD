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

#include "shared/RiemannSolvers.h"
#include "shared/EulerEquations.h"

namespace sdm {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes fluxes at fluxes points taking
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

    // state variable for conservative variables, and flux
    HydroState q, flux;
        
    /*
     * first handle interior point, then end points.
     */
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	
	  // interior points along direction X
	  for (int idx=1; idx<N; ++idx) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along X direction
	    euler.flux_x(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idx
	
	} // end for idy
      } // end for idz

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (i>0 and i<isize) {
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idy=0; idy<N; ++idy) {
	  
	    // conservative state
	    HydroState qL, qR;
	    
	    // primitive state
	    HydroState wL, wR;
	    
	    HydroState qgdnv;
	    
	    // when idx == 0, get right and left state
	    for (int ivar = 0; ivar<nbvar; ++ivar) {  
	      qL[ivar] = UdataFlux(i-1,j,k, dofMapF(N,idy,idz,ivar));
	      qR[ivar] = UdataFlux(i  ,j,k, dofMapF(0,idy,idz,ivar));
	    }
	    
	    // convert to primitive
	    euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	    euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	    
	    // riemann solver
	    ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	    
	    // copy flux
	    for (int ivar = 0; ivar<nbvar; ++ivar) {  
	      UdataFlux(i-1,j,k, dofMapF(N,idy,idz,ivar)) = flux[ivar];
	      UdataFlux(i  ,j,k, dofMapF(0,idy,idz,ivar)) = flux[ivar];
	    }
	    	    
	  } // end for idy
	} // end for idz

      } // end safe-guard
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	
	  // interior points along direction Y
	  for (int idy=1; idy<N; ++idy) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along Y direction
	    euler.flux_y(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idy
	
	} // end for idx
      } // end for idz

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j>0 and j<jsize) {
	
	for (int idz=0; idz<N; ++idz) {
	  for (int idx=0; idx<N; ++idx) {
	  
	    // conservative state
	    HydroState qL, qR;
	    
	    // primitive state
	    HydroState wL, wR;
	    
	    HydroState qgdnv;
	    
	    // when idy == 0, get right and left state
	    for (int ivar = 0; ivar<nbvar; ++ivar) {  
	      qL[ivar] = UdataFlux(i,j-1,k, dofMapF(idx,N,idz,ivar));
	      qR[ivar] = UdataFlux(i,j  ,k, dofMapF(idx,0,idz,ivar));
	    }
	    
	    // convert to primitive
	    euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	    euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	    
	    // riemann solver
	    this->swap( wL[IU], wL[IV] );
	    this->swap( wR[IU], wR[IV] );
	    ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	    
	    // copy back results in current and neighbor cells
	    UdataFlux(i,j-1,k, dofMapF(idx,N,idz,ID)) = flux[ID];
	    UdataFlux(i,j-1,k, dofMapF(idx,N,idz,IE)) = flux[IE];
	    UdataFlux(i,j-1,k, dofMapF(idx,N,idz,IU)) = flux[IV]; // swap again
	    UdataFlux(i,j-1,k, dofMapF(idx,N,idz,IV)) = flux[IU]; // swap again
	    UdataFlux(i,j-1,k, dofMapF(idx,N,idz,IW)) = flux[IW];

	    UdataFlux(i,j  ,k, dofMapF(idx,0,idz,ID)) = flux[ID];
	    UdataFlux(i,j  ,k, dofMapF(idx,0,idz,IE)) = flux[IE];
	    UdataFlux(i,j  ,k, dofMapF(idx,0,idz,IU)) = flux[IV]; // swap again
	    UdataFlux(i,j  ,k, dofMapF(idx,0,idz,IV)) = flux[IU]; // swap again
	    UdataFlux(i,j  ,k, dofMapF(idx,0,idz,IW)) = flux[IW];
	    
	  } // end for idx
	} // end for idz

      } // end safe-guard
      
    } // end for dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    if (dir == IZ) {
      
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	
	  // interior points along direction Z
	  for (int idz=1; idz<N; ++idz) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along Z direction
	    euler.flux_z(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idz
	
	} // end for idx
      } // end for idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (k>0 and k<ksize) {
	
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	  
	    // conservative state
	    HydroState qL, qR;
	    
	    // primitive state
	    HydroState wL, wR;
	    
	    HydroState qgdnv;
	    
	    // when idz == 0, get right and left state
	    for (int ivar = 0; ivar<nbvar; ++ivar) {  
	      qL[ivar] = UdataFlux(i,j,k-1, dofMapF(idx,idy,N,ivar));
	      qR[ivar] = UdataFlux(i,j,k  , dofMapF(idx,idy,0,ivar));
	    }
	    
	    // convert to primitive
	    euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	    euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	    
	    // riemann solver
	    this->swap( wL[IU], wL[IW] );
	    this->swap( wR[IU], wR[IW] );
	    ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	    
	    // copy back results in current and neighbor cells
	    UdataFlux(i,j,k-1, dofMapF(idx,idy,N,ID)) = flux[ID];
	    UdataFlux(i,j,k-1, dofMapF(idx,idy,N,IE)) = flux[IE];
	    UdataFlux(i,j,k-1, dofMapF(idx,idy,N,IU)) = flux[IW]; // swap again
	    UdataFlux(i,j,k-1, dofMapF(idx,idy,N,IV)) = flux[IV];
	    UdataFlux(i,j,k-1, dofMapF(idx,idy,N,IW)) = flux[IU]; // swap again
	    
	    UdataFlux(i,j,k  , dofMapF(idx,idy,0,ID)) = flux[ID];
	    UdataFlux(i,j,k  , dofMapF(idx,idy,0,IE)) = flux[IE];
	    UdataFlux(i,j,k  , dofMapF(idx,idy,0,IU)) = flux[IW]; // swap again
	    UdataFlux(i,j,k  , dofMapF(idx,idy,0,IV)) = flux[IV];
	    UdataFlux(i,j,k  , dofMapF(idx,idy,0,IW)) = flux[IU]; // swap again
	    	    
	  } // end for idx
	} // end for idy

      } // end safe-guard
      
    } // end for dir IZ

  } // 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray UdataFlux;

}; // class ComputeFluxAtFluxPoints_Functor


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * This functor computes fluxes at fluxes points taking
 * as input the conservative variables at fluxes points except
 * at end points on cell borders, where we just perform reconstruction; 
 * fluxes at end points are computed in 
 * ComputeFluxAtEndPoints_with_Limiter_Functor.
 *
 * In this functor, if at a cell border points, the reconstructed value
 * is outside the admissible range [Umin,Umax], we switch to a linear 
 * reconstruction from cell-center to border, after having evaluated 
 * the gradient at cell-center.
 *
 * The computation of fluxes at flux points interior to the cell is 
 * exactly as in ComputeFluxAtFluxPoints_Functor.
 *
 * At end points, we only apply a limiter to ensure the reconstructed
 * value in the TVD range [Umin,Umax]. For that purpose we apply the 
 * idea proposed in :
 *
 * "Spectral Difference Method for Unstructured Grids II: Extension to the
 * Euler Equations.", Wang, Liu, May and Jameson, J. of Sci Comp, vol 32, 
 * July 2007.
 * https://link.springer.com/content/pdf/10.1007/s10915-006-9113-9.pdf
 * 
 * This functor is very near to ComputeFluxAtFluxPoints_Functor, but the
 * computation of fluxes at cell-borders via Riemann solvers is done outside.
 *
 */
template<int dim, int N, int dir>
class ComputeFluxAtFluxPoints_with_Limiter_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;
  static constexpr auto dofMapS = DofMap<dim,N>;

  /**
   * \param[in] params hydro parameters
   * \params[in] sdm_geom is needed to have access to solution points locations
   * \param[in] euler is needed to compute fluxes
   * \param[in] Udata is need to estimate gradient at cell center from solution points
   * \param[in] Uaverage is used as an estimate of conservative variables at cell center
   * \param[in] Umin minimum values of conservative variables in neighborhood
   * \param[in] Umax maximum values of conservative variables in neighborhood
   * \param[in,out] UdataFlux will contain fluxes at interior flux points and reconstructed conservative variables at cell end points. 
   */
  ComputeFluxAtFluxPoints_with_Limiter_Functor(HydroParams         params,
					       SDM_Geometry<dim,N> sdm_geom,
					       ppkMHD::EulerEquations<dim> euler,
					       DataArray   Udata,       
					       DataArray   Uaverage,
					       DataArray   Umin,
					       DataArray   Umax,
					       DataArray   UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    euler(euler),
    Udata(Udata),
    Uaverage(Uaverage),
    Umin(Umin),
    Umax(Umax),
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
    // loop over cell DoF's to compute fluxes at interior  flux points
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
       * special treatment for the end points (only perform reconstruction)
       */      
      // check if limiter reconstruction is needed
      if (i>0 and i<isize) {

	real_t dx = this->params.dx;
	real_t dy = this->params.dy;
	
	for (int ivar = 0; ivar<nbvar; ++ivar) {  

	  /*
	   * compute cell center gradient:
	   * sweep solution points, a perform a simplified least square
	   * estimate of the partial derivative at cell center.
	   */
	  real_t gradx=0.0, grady=0.0;

	  // cell center
	  const real_t xc = 0.5;
	  const real_t yc = 0.5;

	  // least-square estimate of gradient
	  for (int idy=0; idy<N; ++idy) {
	    real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	    real_t delta_y = y-yc;
	    
	    for (int idx=0; idx<N; ++idx) {
	      real_t x = this->sdm_geom.solution_pts_1d_host(idx);
	      real_t delta_x = x-xc;
	      
	      gradx += delta_x*Udata(i,j,dofMapS(idx,idy,0,ivar));
	      grady += delta_y*Udata(i,j,dofMapS(idx,idy,0,ivar));
	      
	    }
	  }
	  gradx /= this->sdm_geom.sum_dx_square;
	  grady /= this->sdm_geom.sum_dy_square;

	  // retrieve admissible range for current cell borders reconstructions
	  real_t umin = Umin(i,j, ivar);
	  real_t umax = Umax(i,j, ivar);
	  
	  /*
	   * for each end point on x-border perform proper reconstruction
	   */
	  for (int idy=0; idy<N; ++idy) {

	    /*
	     * handle end point at idx = 0 and idx = N
	     */
	    for (int idx = 0; idx<=N; idx+=N) {

	      real_t qtmp = UdataFlux(i,j,dofMapF(idx,idy,0,ivar));
	      
	      // check if qtmp falls in range [Umin,Umax]
	      
	      real_t dq;
	      
	      // is a limited reconstruction required ?
	      // if not, don't do anything the regular reconstructed state
	      // qtmp is fine
	      if (qtmp < umin or qtmp > umax) {
		
		// offset from cell center to reconstructed location
		// in the reference cell
		real_t delta_x = (this->sdm_geom.flux_pts_1d(idx)-0.5)*dx;
		real_t delta_y = (this->sdm_geom.flux_pts_1d(idy)-0.5)*dy;
		
		// delta Q (dp) is used in q_recons = q_center + dq
		dq = gradx * delta_x + grady * delta_y;
		
		// proposed reconstructed state from cell center value
		real_t qr = Uaverage(i,j,ivar) + dq;
		
		// write back resulting linear limited reconstructed state
		// with a minmod limitation
		if (qr >= umin and qr <= umax)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = qr;
		if (qr <  umin)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = umin;
		if (qr >  umax)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = umax;
		
	      } // qtmp<umin or qtmp>umax

	    } // end checking end point at idx=0 and idx=N
	    	    
	  } // end for idy
	  	  	  
	} // end for ivar

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
       * special treatment for the end points (only perform reconstruction)
       */      
      // check if limiter reconstruction is needed
      if (j>0 and j<jsize) {

	real_t dx = this->params.dx;
	real_t dy = this->params.dy;
	
	for (int ivar = 0; ivar<nbvar; ++ivar) {  
	  
	  /*
	   * compute cell center gradient:
	   * sweep solution points, a perform a simplified least square
	   * estimate of the partial derivative at cell center.
	   */
	  real_t gradx=0.0, grady=0.0;

	  // cell center
	  const real_t xc = 0.5;
	  const real_t yc = 0.5;

	  // least-square estimate of gradient
	  for (int idy=0; idy<N; ++idy) {
	    real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	    real_t delta_y = y-yc;
	    
	    for (int idx=0; idx<N; ++idx) {
	      real_t x = this->sdm_geom.solution_pts_1d_host(idx);
	      real_t delta_x = x-xc;
	      
	      gradx += delta_x*Udata(i,j,dofMapS(idx,idy,0,ivar));
	      grady += delta_y*Udata(i,j,dofMapS(idx,idy,0,ivar));
	      
	    }
	  }
	  gradx /= this->sdm_geom.sum_dx_square;
	  grady /= this->sdm_geom.sum_dy_square;

	  // retrieve admissible range for current cell borders reconstructions
	  real_t umin = Umin(i,j, ivar);
	  real_t umax = Umax(i,j, ivar);
	  
	  /*
	   * for each end point on y-border perform proper reconstruction
	   */
	  for (int idx=0; idx<N; ++idx) {

	    /*
	     * handle end point at idy = 0 and idy = N
	     */
	    for (int idy = 0; idy<=N; idy+=N) {

	      real_t qtmp = UdataFlux(i,j,dofMapF(idx,idy,0,ivar));
	      
	      // check if qtmp falls in range [Umin,Umax]
	      
	      real_t dq;
	      
	      // is a limited reconstruction required ?
	      // if not, don't do anything the regular reconstructed state
	      // qtmp is fine
	      if (qtmp < umin or qtmp > umax) {
		
		// offset from cell center to reconstructed location
		// in the reference cell
		real_t delta_x = (this->sdm_geom.flux_pts_1d(idx)-0.5)*dx;
		real_t delta_y = (this->sdm_geom.flux_pts_1d(idy)-0.5)*dy;
		
		// delta Q (dp) is used in q_recons = q_center + dq
		dq = gradx * delta_x + grady * delta_y;
		
		// proposed reconstructed state from cell center value
		real_t qr = Uaverage(i,j,ivar) + dq;
		
		// write back resulting linear limited reconstructed state
		// with a minmod limitation
		if (qr >= umin and qr <= umax)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = qr;
		if (qr <  umin)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = umin;
		if (qr >  umax)
		  UdataFlux(i,j,dofMapF(idx,idy,0,ivar)) = umax;
		
	      } // qtmp<umin or qtmp>umax

	    } // end checking end point at idx=0 and idx=N
	    	    
	  } // end for idy
	  	  	  
	} // end for ivar

      } // end safe-guard

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

    // state variable for conservative variables, and flux
    HydroState q, flux;
        
    /*
     * first handle interior point, then end points.
     */
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idy=0; idy<N; ++idy) {
	
	  // interior points along direction X
	  for (int idx=1; idx<N; ++idx) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along X direction
	    euler.flux_x(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idx
	
	} // end for idy
      } // end for idz

      /*
       * special treatment for the end points (only perform reconstruction)
       */      
      // check if limiter reconstruction is needed
      if (i>0 and i<isize) {

	real_t dx = this->params.dx;
	real_t dy = this->params.dy;
	real_t dz = this->params.dz;
	
	for (int ivar = 0; ivar<nbvar; ++ivar) {  

	  /*
	   * compute cell center gradient:
	   * sweep solution points, a perform a simplified least square
	   * estimate of the partial derivative at cell center.
	   */
	  real_t gradx=0.0, grady=0.0, gradz=0.0;

	  // cell center
	  const real_t xc = 0.5;
	  const real_t yc = 0.5;
	  const real_t zc = 0.5;

	  // least-square estimate of gradient
	  for (int idz=0; idz<N; ++idz) {
	    real_t z = this->sdm_geom.solution_pts_1d_host(idz);
	    real_t delta_z = z-zc;
	    
	    for (int idy=0; idy<N; ++idy) {
	      real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	      real_t delta_y = y-yc;
	      
	      for (int idx=0; idx<N; ++idx) {
		real_t x = this->sdm_geom.solution_pts_1d_host(idx);
		real_t delta_x = x-xc;
		
		gradx += delta_x*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		grady += delta_y*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		gradz += delta_z*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
	      
	      } // end for idx
	    } // end for idy
	  } // end for idz
	  gradx /= this->sdm_geom.sum_dx_square;
	  grady /= this->sdm_geom.sum_dy_square;
	  gradz /= this->sdm_geom.sum_dz_square;

	  // retrieve admissible range for current cell borders reconstructions
	  real_t umin = Umin(i,j,k, ivar);
	  real_t umax = Umax(i,j,k, ivar);
	  
	  /*
	   * for each end point on x-border perform proper reconstruction
	   */
	  for (int idz=0; idz<N; ++idz) {
	    for (int idy=0; idy<N; ++idy) {

	      /*
	       * handle end point at idx = 0 and idx = N
	       */
	      for (int idx = 0; idx<=N; idx+=N) {

		real_t qtmp = UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar));
		
		// check if qtmp falls in range [Umin,Umax]
		
		real_t dq;
		
		// is a limited reconstruction required ?
		// if not, don't do anything the regular reconstructed state
		// qtmp is fine
		if (qtmp < umin or qtmp > umax) {
		  
		  // offset from cell center to reconstructed location
		  // in the reference cell
		  real_t delta_x = (this->sdm_geom.flux_pts_1d(idx)-0.5)*dx;
		  real_t delta_y = (this->sdm_geom.flux_pts_1d(idy)-0.5)*dy;
		  real_t delta_z = (this->sdm_geom.flux_pts_1d(idz)-0.5)*dz;
		
		  // delta Q (dp) is used in q_recons = q_center + dq
		  dq =
		    gradx * delta_x +
		    grady * delta_y +
		    gradz * delta_z;
		
		  // proposed reconstructed state from cell center value
		  real_t qr = Uaverage(i,j,k,ivar) + dq;
		
		  // write back resulting linear limited reconstructed state
		  // with a minmod limitation
		  if (qr >= umin and qr <= umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = qr;
		  if (qr <  umin)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umin;
		  if (qr >  umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umax;
		
		} // qtmp<umin or qtmp>umax

	      } // end checking end point at idx=0 and idx=N
	      
	    } // end for idy
	  } // end for idz
	  	  	  
	} // end for ivar

      } // end safe-guard
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {
      
      for (int idz=0; idz<N; ++idz) {
	for (int idx=0; idx<N; ++idx) {
	
	  // interior points along direction Y
	  for (int idy=1; idy<N; ++idy) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along Y direction
	    euler.flux_y(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idy
	
	} // end for idx
      } // end for idz

      /*
       * special treatment for the end points (only perform reconstruction)
       */      
      // check if limiter reconstruction is needed
      if (j>0 and j<jsize) {

	real_t dx = this->params.dx;
	real_t dy = this->params.dy;
	real_t dz = this->params.dz;
	
	for (int ivar = 0; ivar<nbvar; ++ivar) {  

	  /*
	   * compute cell center gradient:
	   * sweep solution points, a perform a simplified least square
	   * estimate of the partial derivative at cell center.
	   */
	  real_t gradx=0.0, grady=0.0, gradz=0.0;

	  // cell center
	  const real_t xc = 0.5;
	  const real_t yc = 0.5;
	  const real_t zc = 0.5;

	  // least-square estimate of gradient
	  for (int idz=0; idz<N; ++idz) {
	    real_t z = this->sdm_geom.solution_pts_1d_host(idz);
	    real_t delta_z = z-zc;
	    
	    for (int idy=0; idy<N; ++idy) {
	      real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	      real_t delta_y = y-yc;
	      
	      for (int idx=0; idx<N; ++idx) {
		real_t x = this->sdm_geom.solution_pts_1d_host(idx);
		real_t delta_x = x-xc;
		
		gradx += delta_x*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		grady += delta_y*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		gradz += delta_z*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
	      
	      } // end for idx
	    } // end for idy
	  } // end for idz
	  gradx /= this->sdm_geom.sum_dx_square;
	  grady /= this->sdm_geom.sum_dy_square;
	  gradz /= this->sdm_geom.sum_dz_square;

	  // retrieve admissible range for current cell borders reconstructions
	  real_t umin = Umin(i,j,k, ivar);
	  real_t umax = Umax(i,j,k, ivar);
	  
	  /*
	   * for each end point on x-border perform proper reconstruction
	   */
	  for (int idz=0; idz<N; ++idz) {
	    for (int idx=0; idx<N; ++idx) {

	      /*
	       * handle end point at idy = 0 and idy = N
	       */
	      for (int idy = 0; idy<=N; idy+=N) {
		
		real_t qtmp = UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar));
		
		// check if qtmp falls in range [Umin,Umax]
		
		real_t dq;
		
		// is a limited reconstruction required ?
		// if not, don't do anything the regular reconstructed state
		// qtmp is fine
		if (qtmp < umin or qtmp > umax) {
		  
		  // offset from cell center to reconstructed location
		  // in the reference cell
		  real_t delta_x = (this->sdm_geom.flux_pts_1d(idx)-0.5)*dx;
		  real_t delta_y = (this->sdm_geom.flux_pts_1d(idy)-0.5)*dy;
		  real_t delta_z = (this->sdm_geom.flux_pts_1d(idz)-0.5)*dz;
		
		  // delta Q (dp) is used in q_recons = q_center + dq
		  dq =
		    gradx * delta_x +
		    grady * delta_y +
		    gradz * delta_z;
		
		  // proposed reconstructed state from cell center value
		  real_t qr = Uaverage(i,j,k,ivar) + dq;
		
		  // write back resulting linear limited reconstructed state
		  // with a minmod limitation
		  if (qr >= umin and qr <= umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = qr;
		  if (qr <  umin)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umin;
		  if (qr >  umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umax;
		
		} // qtmp<umin or qtmp>umax

	      } // end checking end point at idy=0 and idy=N
	      
	    } // end for idx
	  } // end for idz
	  	  	  
	} // end for ivar

      } // end safe-guard
      
    } // end for dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    if (dir == IZ) {
      
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	
	  // interior points along direction Z
	  for (int idz=1; idz<N; ++idz) {
	    
	    // retrieve state conservative variables
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
  
	      q[ivar] = UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar));
	      
	    }

	    // compute pressure
	    real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
	    // compute flux along Z direction
	    euler.flux_z(q, p, flux);
	  
	    // copy back interpolated value
	    for (int ivar = 0; ivar<nbvar; ++ivar) {
	      
	      UdataFlux(i,j,k, dofMapF(idx,idy,idz,ivar)) = flux[ivar];
	      
	    } // end for ivar
	    
	  } // end for idz
	
	} // end for idx
      } // end for idy

      /*
       * special treatment for the end points (only perform reconstruction)
       */      
      // check if limiter reconstruction is needed
      if (k>0 and k<ksize) {

	real_t dx = this->params.dx;
	real_t dy = this->params.dy;
	real_t dz = this->params.dz;
	
	for (int ivar = 0; ivar<nbvar; ++ivar) {  

	  /*
	   * compute cell center gradient:
	   * sweep solution points, a perform a simplified least square
	   * estimate of the partial derivative at cell center.
	   */
	  real_t gradx=0.0, grady=0.0, gradz=0.0;

	  // cell center
	  const real_t xc = 0.5;
	  const real_t yc = 0.5;
	  const real_t zc = 0.5;

	  // least-square estimate of gradient
	  for (int idz=0; idz<N; ++idz) {
	    real_t z = this->sdm_geom.solution_pts_1d_host(idz);
	    real_t delta_z = z-zc;
	    
	    for (int idy=0; idy<N; ++idy) {
	      real_t y = this->sdm_geom.solution_pts_1d_host(idy);
	      real_t delta_y = y-yc;
	      
	      for (int idx=0; idx<N; ++idx) {
		real_t x = this->sdm_geom.solution_pts_1d_host(idx);
		real_t delta_x = x-xc;
		
		gradx += delta_x*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		grady += delta_y*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
		gradz += delta_z*Udata(i,j,k,dofMapS(idx,idy,idz,ivar));
	      
	      } // end for idx
	    } // end for idy
	  } // end for idz
	  gradx /= this->sdm_geom.sum_dx_square;
	  grady /= this->sdm_geom.sum_dy_square;
	  gradz /= this->sdm_geom.sum_dz_square;

	  // retrieve admissible range for current cell borders reconstructions
	  real_t umin = Umin(i,j,k, ivar);
	  real_t umax = Umax(i,j,k, ivar);
	  
	  /*
	   * for each end point on x-border perform proper reconstruction
	   */
	  for (int idy=0; idy<N; ++idy) {
	    for (int idx=0; idx<N; ++idx) {

	      /*
	       * handle end point at idz = 0 and idz = N
	       */
	      for (int idz = 0; idz<=N; idz+=N) {
		
		real_t qtmp = UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar));
		
		// check if qtmp falls in range [Umin,Umax]
		
		real_t dq;
		
		// is a limited reconstruction required ?
		// if not, don't do anything the regular reconstructed state
		// qtmp is fine
		if (qtmp < umin or qtmp > umax) {
		  
		  // offset from cell center to reconstructed location
		  // in the reference cell
		  real_t delta_x = (this->sdm_geom.flux_pts_1d(idx)-0.5)*dx;
		  real_t delta_y = (this->sdm_geom.flux_pts_1d(idy)-0.5)*dy;
		  real_t delta_z = (this->sdm_geom.flux_pts_1d(idz)-0.5)*dz;
		
		  // delta Q (dp) is used in q_recons = q_center + dq
		  dq =
		    gradx * delta_x +
		    grady * delta_y +
		    gradz * delta_z;
		
		  // proposed reconstructed state from cell center value
		  real_t qr = Uaverage(i,j,k,ivar) + dq;
		
		  // write back resulting linear limited reconstructed state
		  // with a minmod limitation
		  if (qr >= umin and qr <= umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = qr;
		  if (qr <  umin)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umin;
		  if (qr >  umax)
		    UdataFlux(i,j,k,dofMapF(idx,idy,idz,ivar)) = umax;
		
		} // qtmp<umin or qtmp>umax

	      } // end checking end point at idz=0 and idz=N
	      
	    } // end for idx
	  } // end for idy
	  	  	  
	} // end for ivar

      } // end safe-guard
      
    } // end for dir IZ

  } // 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray Udata;
  DataArray Uaverage;
  DataArray Umin;
  DataArray Umax;
  DataArray UdataFlux;

}; // class ComputeFluxAtFluxPoints_with_Limiter_Functor

} // namespace sdm

#endif // SDM_FLUX_FUNCTORS_H_
