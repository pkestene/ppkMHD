/**
 * \file SDM_Positivity_preserving.h
 *
 * Implement ideas from article
 * "On positivity-preserving high order discontinuous Galerkin schemes for
 * compressible Euler equations on rectangular meshes", Xiangxiong Zhang, 
 * Chi-Wang Shu, Journal of Computational Physics Volume 229, Issue 23,
 * 20 November 2010, Pages 8918-8934
 *
 * The idea is to ensure/preserve positivity after each Runge-Kutta stage.
 */
#ifndef SDM_POSITIVITY_PRESERVING_H_
#define SDM_POSITIVITY_PRESERVING_H_

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
 * This functor implements ideas from Zhang and Shu about 
 * positivity preserving.
 * It designed to be called right after functor 
 * Interpolate_At_FluxPoints_Functor.
 *
 * Its purpose is to modify conservative variable at flux to 
 * ensure positivity. These variables are computed by 
 * Interpolate_At_FluxPoints_Functor using the solution points 
 * Lagrange basis.
 *
 * Auxiliary variables theta1 and theta2 are defined in Zhang-Shu, 
 * Journal of Computational Physics, 229 (2010), pages 8918-8934.
 *
 */
template<int dim, int N, int dir>
class Apply_positivity_Functor : public SDMBaseFunctor<dim,N> {

public:
  using typename SDMBaseFunctor<dim,N>::DataArray;
  using typename SDMBaseFunctor<dim,N>::solution_values_t;
  using typename SDMBaseFunctor<dim,N>::flux_values_t;

  static constexpr auto dofMapS = DofMap<dim,N>;
  static constexpr auto dofMapF = DofMapFlux<dim,N,dir>;

  /**
   * \param[in] params contains hydrodynamics parameters
   * \param[in] sdm_geom contains parameters to init base class functor
   * \param[in,out] UdataSol contains conservative variables at solution points
   * \param[in,out] UdataFlux contains conservative variables at flux points
   * \params[in] Uaverage contains cell volume averaged conservative variables.
   */
  Apply_positivity_Functor(HydroParams         params,
			   SDM_Geometry<dim,N> sdm_geom,
			   DataArray           UdataSol,
			   DataArray           UdataFlux,
			   DataArray           Uaverage) :
    SDMBaseFunctor<dim,N>(params,sdm_geom),
    UdataSol(UdataSol),
    UdataFlux(UdataFlux),
    Uaverage(Uaverage)
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
    
    const int nbvar = this->params.nbvar;

    const real_t gamma0 = this->params.settings.gamma0;
    
    // local cell index
    int i,j;
    index2coord(index,i,j,isize,jsize);

    int idx_flux_end = N+1;
    int idy_flux_end = N;

    // bound for flux point sweep
    if (dir == IX) {
      idx_flux_end = N+1;
      idy_flux_end = N;
    }
    if (dir == IY) {
      idx_flux_end = N;
      idy_flux_end = N+1;
    }

    // average density at cell level
    const real_t rho_ave = Uaverage(i,j,ID);
    const real_t rhou_ave = Uaverage(i,j,IU);
    const real_t rhov_ave = Uaverage(i,j,IV);
    const real_t e_ave    = Uaverage(i,j,IE);
    
    /*
     * enforce density positivity
     */

    // first compute minimun density inside current cell
    real_t rho_min;
#ifdef __CUDA_ARCH__
    rho_min = CUDART_INF; // something big
#else
    rho_min = std::numeric_limits<real_t>::max();
#endif

    // compute rho_min over the flux points
    for (int idy=0; idy<idy_flux_end; ++idy) {
      for (int idx=0; idx<idx_flux_end; ++idx) {
	
	const real_t rho = UdataFlux(i,j,dofMapF(idx,idy,0,ID));
	rho_min = rho_min < rho ? rho_min : rho;
	
      } // end for idx
    } // end for idy
      
    const real_t eps1 = this->params.settings.smallr; // a small density
    const real_t ratio = (rho_ave - eps1)/(rho_ave - rho_min) + 1e-13;
    const real_t theta1 = ratio < 1.0 ? ratio : 1.0;

    // check if we need to modify density at solution points and flux points
    if (theta1 < 1.0) {
      
      // vector of values at solution points
      solution_values_t sol;
      flux_values_t     flux;

      if (dir == IX) {
	
	// sweep solution points
	for (int idy=0; idy<N; ++idy) {
	  for (int idx=0; idx<N; ++idx) {
	    
	    const real_t rho = UdataSol(i,j,dofMapS(idx,idy,0,ID));
	    const real_t rho_new = theta1 * (rho - rho_ave) + rho_ave;

	    // modify density at solution point
	    UdataSol(i,j,dofMapS(idx,idy,0,ID)) = rho_new;
	    
	    // prepare vector to recompute density at flux points
	    sol[idx] = rho_new;
	    
	  } // end for idx
	  
	  // interpolate density at flux points
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value in Fluxes data array
	  for (int idx=0; idx<N+1; ++idx) {
	    
	    // modify density at flux point
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ID)) = flux[idx];
	    
	  } // end for idx
	} // end for idy
	
      } else { // dir == IY, we need to swap idx <-> idy

	// sweep solution points
	for (int idx=0; idx<N; ++idx) {
	  for (int idy=0; idy<N; ++idy) {
	    
	    const real_t rho = UdataSol(i,j,dofMapS(idx,idy,0,ID));
	    const real_t rho_new = theta1 * (rho - rho_ave) + rho_ave;
	    
	    UdataSol(i,j,dofMapS(idx,idy,0,ID)) = rho_new;
	    
	    // prepare vector to recompute density at flux points
	    sol[idy] = rho_new;
	    
	  } // end for idy
	  
	  // interpolate density at flux points
	  this->sol2flux_vector(sol, flux);
	  
	  // copy back interpolated value in Fluxes data array
	  for (int idy=0; idy<N+1; ++idy) {
	    
	    UdataFlux(i  ,j  , dofMapF(idx,idy,0,ID)) = flux[idy];
	    
	  } // end for idy
	} // end for idx	

      } // end IY
      
    } // end if theta1

    
    /*
     * enforce pressure positivity
     *
     * theta2 is computed as the min value of t over all flux points, where
     * t itself is the solution of a 2nd order equation: a*t^2 + b*t + c = 0
     * t should be in range [0,1] as it is used as a weight in a convexe 
     * combination.
     */
    real_t theta2 = 1.0;

    const real_t eps2 = 1e-13;

    // compute primitive variable of the cell averaged value
    // const real_t pressure_ave = (gamma0-1)*(e_ave-0.5*(rhou_ave*rhou_ave+
    // 						       rhov_ave*rhov_ave)/rho_ave);
    
    // sweep flux points to find minimal theta2
    for (int idy=0; idy<idy_flux_end; ++idy) {
      for (int idx=0; idx<idx_flux_end; ++idx) {
	
	const real_t E    = UdataFlux(i,j,dofMapF(idx,idy,0,IE));
	const real_t rhou = UdataFlux(i,j,dofMapF(idx,idy,0,IU));
	const real_t rhov = UdataFlux(i,j,dofMapF(idx,idy,0,IV));
	const real_t rho  = UdataFlux(i,j,dofMapF(idx,idy,0,ID));
	const real_t pressure = (gamma0-1)*(E-0.5*(rhou*rhou+rhov*rhov)/rho);
	
	if (pressure < 1e-12) {
	  real_t drho  = rho - rho_ave;
	  real_t dE    = E   - e_ave;
	  real_t drhou = rhou-rhou_ave;
	  real_t drhov = rhov-rhov_ave;
	  
	  real_t dm2 = drhou*drhou + drhov*drhov;
	  
	  // solve 2nd order equation in t:
	  // a_1 t^2 + b_1 t + c_1 = 0
	  real_t a1 = 2.0*drho*dE - dm2;
	  real_t b1 = 2.0*drho*(e_ave - eps2/(gamma0-1.0))
	    + 2.0*rho_ave*dE
	    - 2.0*(rhou_ave*drhou + rhov_ave*drhov);
	  real_t c1 = 2.0*rho_ave*e_ave
	      - (rhou_ave*rhou_ave + rhov_ave*rhov_ave)
	    - 2.0*eps2*rho_ave/(gamma0-1.0);
	  // Divide by a1 to avoid round-off error
	  b1 /= a1;
	  c1 /= a1;
	  // discrimant
	  real_t D = sqrt( fabs(b1*b1 - 4.0*c1) );
	  
	  // possible solutions
	  real_t t1 = 0.5*(-b1 - D);
	  real_t t2 = 0.5*(-b1 + D);
	  real_t t=0.0;
	  if(     t1 > -1.0e-12 and t1 < 1.0 + 1.0e-12)
	    t = t1;
	  else if(t2 > -1.0e-12 and t2 < 1.0 + 1.0e-12)
	    t = t2;
	  else
	    ; // Houston, we have a problem
	  
	  // t should strictly lie in [0,1]
	  t = t<1.0 ? t : 1.0;
	  t = t>0.0 ? t : 0.0;
	  // Need t < 1.0. If t==1 upto machine precision
	  // then we are suffering from round off error.
	  // In this case we take the cell average value, t=0.
	  if (fabs(1.0-t) < 1.0e-14)
	    t = 0.0;
	  
	  theta2 = theta2 < t ? theta2 : t; // min(theta2, t);
	  
	} // end small pressure
	
      } // for idx
    } // for idy
    
    if (theta2 < 1.0) {
      
      // we need to modify the values at solution points:
      for (int idy=0; idy<N; ++idy) {
	for (int idx=0; idx<N; ++idx) {
	  
	  real_t val;
	  
	  val = UdataSol(i,j,dofMapS(idx,idy,0,ID));
	  val = theta2 * ( val - rho_ave ) + rho_ave;
	  UdataSol(i,j,dofMapS(idx,idy,0,ID)) = val;
	  
	  val = UdataSol(i,j,dofMapS(idx,idy,0,IE));
	  val = theta2 * ( val - e_ave ) + e_ave;
	  UdataSol(i,j,dofMapS(idx,idy,0,IE)) = val;
	  
	  val = UdataSol(i,j,dofMapS(idx,idy,0,IU));
	  val = theta2 * ( val - rhou_ave ) + rhou_ave;
	  UdataSol(i,j,dofMapS(idx,idy,0,IU)) = val;
	  
	  val = UdataSol(i,j,dofMapS(idx,idy,0,IV));
	  val = theta2 * ( val - rhov_ave ) + rhov_ave;
	  UdataSol(i,j,dofMapS(idx,idy,0,IV)) = val;
	  
	} // end for idx
      } // end for idy

      /*
       * modification at flux points
       */
      {
	// vector of values at solution points
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
	      
	      // positivity preserving for density
	      if (ivar==ID) {
		for (int idf=0; idf<N+1; ++idf)
		  flux[idf] = fmax(flux[idf], this->params.settings.smallr);
	      }
	      
	      // copy back interpolated value
	      for (int idx=0; idx<N+1; ++idx) {
		
		UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar)) = flux[idx];
		
	      } // end for idx
	      
	    } // end for ivar
	    
	  } // end for idy
	  
	} // end for dir IX
	
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
	      
	      // positivity preserving for density
	      if (ivar==ID) {
		for (int idf=0; idf<N+1; ++idf)
		  flux[idf] = fmax(flux[idf], this->params.settings.smallr);
	      }
	      
	      // copy back interpolated value
	      for (int idy=0; idy<N+1; ++idy) {
		
		UdataFlux(i  ,j  , dofMapF(idx,idy,0,ivar)) = flux[idy];
		
	      }
	      
	    } // end for ivar
	    
	  } // end for idx
	  
	} // end for dir IY
	
      } // end modification at flux points
      
    } // end theta2 < 1
          
  } // end operator() - 2d
    
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

    //const int nbvar = this->params.nbvar;

    // local cell index
    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);


  } // end operator () - 3d

  //! solution data array
  DataArray UdataSol;
  DataArray UdataFlux;
  DataArray Uaverage;
  
}; // class Apply_positivity_Functor

} // namespace sdm

#endif // SDM_POSITIVITY_PRESERVING_H_