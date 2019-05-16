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
 * Kokkos execution policy is Range with iterations mapping
 * the total number of flux point locations (flux Dofs).
 *
 * We first compute fluxes at flux points interior to cell, then 
 * handle the end points.
 */
template<int dim, int N, int dir>
class ComputeFluxAtFluxPoints_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::HydroState;
  
  ComputeFluxAtFluxPoints_Functor(HydroParams                 params,
				  SDM_Geometry<dim,N>         sdm_geom,
				  ppkMHD::EulerEquations<dim> euler,
				  DataArray                   UdataFlux) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    euler(euler),
    UdataFlux(UdataFlux),
    isize(params.isize),
    jsize(params.jsize),
    ksize(params.ksize),
    ghostWidth(params.ghostWidth),
    nbvar(params.nbvar)
  {};

  // static method which does it all: create and execute functor
  static void apply(HydroParams         params,
                    SDM_Geometry<dim,N> sdm_geom,
                    ppkMHD::EulerEquations<dim> euler,
                    DataArray           UdataFlux)
  {
    int64_t nbDofs = (dim==2) ? 
      params.isize * params.jsize * N * (N+1) :
      params.isize * params.jsize * params.ksize * N * N * (N+1);
    
    ComputeFluxAtFluxPoints_Functor functor(params, sdm_geom, 
                                            euler, UdataFlux);
    Kokkos::parallel_for("ComputeFluxAtFluxPoints_Functor", nbDofs, functor);
  }

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
    // global index
    int ii,jj;
    if (dir == IX)
      index2coord(index,ii,jj,isize*(N+1),jsize*N);
    else
      index2coord(index,ii,jj,isize*N,jsize*(N+1));

    // local cell index
    int i,j;

    // Dof index for flux
    int idx,idy;

    // mapping thread to solution Dof
    global2local_flux<dir>(ii,jj, i,j,idx,idy, N);

    // state variable for conservative variables, and flux
    HydroState q = {}, flux = {};

    /*
     * first handle interior points, then end points.
     */
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {
      
      // only interior points along direction X
      if (idx>0 and idx<N) {
	  
        // retrieve state conservative variables
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          q[ivar] = UdataFlux(ii,jj,ivar);
          
        }
        
        // compute pressure
        real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	  
        // compute flux along X direction
        euler.flux_x(q, p, flux);
        
        // copy back interpolated value
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          UdataFlux(ii,jj,ivar) = flux[ivar];
	  
        } // end for ivar
	  
      } // end if idx

      /*
       * special treatment for the end points (Riemann solver)
       */
      
      // compute left interface Riemann problems
      if (i>0 and i<isize) {

        // only first thread is working
	if (idx==0) {
          
	  // conservative state
	  HydroState qL = {}, qR = {};
	  
	  // primitive state
	  HydroState wL, wR;
	  
	  HydroState qgdnv;
	  
	  // when idx == 0, get right and left state
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qL[ivar] = UdataFlux(ii-1,jj,ivar);
	    qR[ivar] = UdataFlux(ii  ,jj,ivar);
	  }
	  
	  // convert to primitive
	  euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	  euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
	  // riemann solver
	  ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy back result in current cell and in neighbor
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    UdataFlux(ii-1,jj,ivar) = flux[ivar];
	    UdataFlux(ii  ,jj,ivar) = flux[ivar];
	  }
	  	  
	} // end if idx==0

      } // end safe-guard
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {
      
      // only interior points along direction X
      if (idy>0 and idy<N) {
        
        // for each variables
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          q[ivar] = UdataFlux(ii,jj,ivar);
          
        }
        
        // compute pressure
        real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	
        // compute flux along Y direction
        euler.flux_y(q, p, flux);
	
        // copy back interpolated value
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          UdataFlux(ii,jj,ivar) = flux[ivar];
	  
        } // end for ivar
	
      } // end if idy

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j>0 and j<jsize) {
	
        // only first thread is working
	if (idy==0) {
          
	  // conservative state
	  HydroState qL = {}, qR = {};
	  
	  // primitive state
	  HydroState wL, wR;
	  
	  HydroState qgdnv;
	  
	  // when idy == 0, get right and left state
	  for (int ivar = 0; ivar<nbvar; ++ivar) {  
	    qL[ivar] = UdataFlux(ii,jj-1,ivar);
	    qR[ivar] = UdataFlux(ii,jj  ,ivar);
	  }
	  
	  // convert to primitive : q -> w
	  euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
	  euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
	  // riemann solver
	  this->swap( wL[IU], wL[IV] );
	  this->swap( wR[IU], wR[IV] );
	  ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
	  // copy back results in current cell as well as in neighbor
	  UdataFlux(ii,jj  ,ID) = flux[ID];
	  UdataFlux(ii,jj  ,IE) = flux[IE];
	  UdataFlux(ii,jj  ,IU) = flux[IV]; // swap again
	  UdataFlux(ii,jj  ,IV) = flux[IU]; // swap again

	  UdataFlux(ii,jj-1,ID) = flux[ID];
	  UdataFlux(ii,jj-1,IE) = flux[IE];
	  UdataFlux(ii,jj-1,IU) = flux[IV]; // swap again
	  UdataFlux(ii,jj-1,IV) = flux[IU]; // swap again
	  
	} // end if idy==0

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

    // global index
    int ii,jj,kk;
    if (dir == IX)
      index2coord(index,ii,jj,kk,isize*(N+1),jsize*N,ksize*N);
    else if (dir == IY)
      index2coord(index,ii,jj,kk,isize*N,jsize*(N+1),ksize*N);
    else
      index2coord(index,ii,jj,kk,isize*N,jsize*N,ksize*(N+1));
    
    // local cell index
    int i,j,k;
    
    // Dof index for flux
    int idx,idy,idz;

    // mapping thread to solution Dof
    global2local_flux<dir>(ii,jj, kk,
                           i,j,k, idx,idy, idz, N);

    // state variable for conservative variables, and flux
    HydroState q = {}, flux = {};
        
    /*
     * first handle interior points, then end points.
     */
    
    // =========================
    // ========= DIR X =========
    // =========================
    if (dir == IX) {
      
      // only interior points along direction X
      if (idx>0 and idx<N) {
        
        // retrieve state conservative variables
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          q[ivar] = UdataFlux(ii,jj,kk,ivar);
	  
        }
        
        // compute pressure
        real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	
        // compute flux along X direction
        euler.flux_x(q, p, flux);
	
        // copy back interpolated value
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          UdataFlux(ii,jj,kk,ivar) = flux[ivar];
	  
        } // end for ivar
	
      } // end for idx interior

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (i>0 and i<isize) {
	
        if (idx==0) {
	  
          // conservative state
          HydroState qL = {}, qR = {};
	  
          // primitive state
          HydroState wL, wR;
	  
          HydroState qgdnv;
	  
          // when idx == 0, get right and left state
          for (int ivar = 0; ivar<nbvar; ++ivar) {  
            qL[ivar] = UdataFlux(ii-1,jj,kk,ivar);
            qR[ivar] = UdataFlux(ii  ,jj,kk,ivar);
          }
	  
          // convert to primitive
          euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
          euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
          // riemann solver
          ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
          // copy flux
          for (int ivar = 0; ivar<nbvar; ++ivar) {  
            UdataFlux(ii-1,jj,kk,ivar) = flux[ivar];
            UdataFlux(ii  ,jj,kk,ivar) = flux[ivar];
          }
	  
        } // end for idx==0

      } // end safe-guard
      
    } // end for dir IX

    // =========================
    // ========= DIR Y =========
    // =========================
    if (dir == IY) {
      
      // interior points along direction Y
      if (idy>0 and idy<N) {
        
        // retrieve state conservative variables
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          q[ivar] = UdataFlux(ii,jj,kk,ivar);
	  
        }
        
        // compute pressure
        real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	
        // compute flux along Y direction
        euler.flux_y(q, p, flux);
	
        // copy back interpolated value
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          UdataFlux(ii,jj,kk,ivar) = flux[ivar];
	  
        } // end for ivar
	
      } // end for idy interior

      /*
       * special treatment for the end points (Riemann solver)
       */

      // compute left interface Riemann problems
      if (j>0 and j<jsize) {
	
        if (idy==0) {
          
          // conservative state
          HydroState qL = {}, qR = {};
	  
          // primitive state
          HydroState wL, wR;
	  
          HydroState qgdnv;
	  
          // when idy == 0, get right and left state
          for (int ivar = 0; ivar<nbvar; ++ivar) {  
            qL[ivar] = UdataFlux(ii,jj-1,kk,ivar);
            qR[ivar] = UdataFlux(ii,jj  ,kk,ivar);
          }
	  
          // convert to primitive
          euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
          euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
          // riemann solver
          this->swap( wL[IU], wL[IV] );
          this->swap( wR[IU], wR[IV] );
          ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
          // copy back results in current and neighbor cells
          UdataFlux(ii,jj-1,kk,ID) = flux[ID];
          UdataFlux(ii,jj-1,kk,IE) = flux[IE];
          UdataFlux(ii,jj-1,kk,IU) = flux[IV]; // swap again
          UdataFlux(ii,jj-1,kk,IV) = flux[IU]; // swap again
          UdataFlux(ii,jj-1,kk,IW) = flux[IW];
          
          UdataFlux(ii,jj  ,kk,ID) = flux[ID];
          UdataFlux(ii,jj  ,kk,IE) = flux[IE];
          UdataFlux(ii,jj  ,kk,IU) = flux[IV]; // swap again
          UdataFlux(ii,jj  ,kk,IV) = flux[IU]; // swap again
          UdataFlux(ii,jj  ,kk,IW) = flux[IW];
	  
        } // end if idy==0

      } // end safe-guard
      
    } // end for dir IY

    // =========================
    // ========= DIR Z =========
    // =========================
    if (dir == IZ) {
      
      // interior points along direction Z
      if (idz>0 and idz<N) {
        
        // retrieve state conservative variables
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          q[ivar] = UdataFlux(ii,jj,kk,ivar);
	  
        }
        
        // compute pressure
        real_t p = euler.compute_pressure(q, this->params.settings.gamma0);
	
        // compute flux along Z direction
        euler.flux_z(q, p, flux);
	
        // copy back interpolated value
        for (int ivar = 0; ivar<nbvar; ++ivar) {
          
          UdataFlux(ii,jj,kk,ivar) = flux[ivar];
	  
        } // end for ivar
	
      } // end for idz interior
  
      /*
       * special treatment for the end points (Riemann solver)
       */
      
      // compute left interface Riemann problems
      if (k>0 and k<ksize) {
	
        if (idz==0) {
	  
          // conservative state
          HydroState qL = {}, qR = {};
	  
          // primitive state
          HydroState wL, wR;
	  
          HydroState qgdnv;
	  
          // when idz == 0, get right and left state
          for (int ivar = 0; ivar<nbvar; ++ivar) {  
            qL[ivar] = UdataFlux(ii,jj,kk-1,ivar);
            qR[ivar] = UdataFlux(ii,jj,kk  ,ivar);
          }
	  
          // convert to primitive
          euler.convert_to_primitive(qR,wR,this->params.settings.gamma0);
          euler.convert_to_primitive(qL,wL,this->params.settings.gamma0);
	  
          // riemann solver
          this->swap( wL[IU], wL[IW] );
          this->swap( wR[IU], wR[IW] );
          ppkMHD::riemann_hydro(wL,wR,qgdnv,flux,this->params);
	  
          // copy back results in current and neighbor cells
          UdataFlux(ii,jj,kk-1,ID) = flux[ID];
          UdataFlux(ii,jj,kk-1,IE) = flux[IE];
          UdataFlux(ii,jj,kk-1,IU) = flux[IW]; // swap again
          UdataFlux(ii,jj,kk-1,IV) = flux[IV];
          UdataFlux(ii,jj,kk-1,IW) = flux[IU]; // swap again
	  
          UdataFlux(ii,jj,kk  ,ID) = flux[ID];
          UdataFlux(ii,jj,kk  ,IE) = flux[IE];
          UdataFlux(ii,jj,kk  ,IU) = flux[IW]; // swap again
          UdataFlux(ii,jj,kk  ,IV) = flux[IV];
          UdataFlux(ii,jj,kk  ,IW) = flux[IU]; // swap again
	  
        } // end if idz==0
        
      } // end safe-guard
      
    } // end for dir IZ

  } // 3d

  ppkMHD::EulerEquations<dim> euler;
  DataArray UdataFlux;
  const int     isize, jsize, ksize;
  const int     ghostWidth;
  const int     nbvar;

}; // class ComputeFluxAtFluxPoints_Functor

} // namespace sdm

#endif // SDM_FLUX_FUNCTORS_H_
