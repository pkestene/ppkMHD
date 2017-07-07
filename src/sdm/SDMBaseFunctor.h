#ifndef SDM_BASE_FUNCTORS_H_
#define SDM_BASE_FUNCTORS_H_

#include <type_traits>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "sdm/SDM_Geometry.h"

namespace sdm {

/**
 * SDM base functor, this is not a functor, but a base class to derive an actual
 * Kokkos functor.
 */
template<int dim, int N>
class SDMBaseFunctor
{
  
public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim==2,HydroState2d,HydroState3d>::type;
  
  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;

  using solution_values_t = Kokkos::Array<real_t, N>;
  using flux_values_t     = Kokkos::Array<real_t, N+1>;
  
  SDMBaseFunctor(HydroParams params,
		 SDM_Geometry<dim,N> sdm_geom) :
    params(params),
    sdm_geom(sdm_geom) {};

  virtual ~SDMBaseFunctor() {};

  SDM_Geometry<dim,N> sdm_geom;
  HydroParams params;
  
  /**
   * a dummy swap device routine.
   */
  KOKKOS_INLINE_FUNCTION
  void swap ( real_t& a, real_t& b ) const {
    real_t c = a; a=b; b=c;
  }
  
  /**
   * Equation of state:
   * compute pressure p and speed of sound c, from density rho and
   * internal energy eint using the "calorically perfect gas" equation
   * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
   * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
   *  \f$ \left[ c_p/c_v \right] \f$.
   * 
   * @param[in]  rho  density
   * @param[in]  eint internal energy
   * @param[out] p    pressure
   * @param[out] c    speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void eos(real_t rho,
	   real_t eint,
	   real_t* p,
	   real_t* c) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallp = params.settings.smallp;
    
    *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = SQRT(gamma0 * (*p) / rho);
    
  } // eos

  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const typename Kokkos::Impl::enable_if<dim_==2, HydroState>::type& u,
			 real_t* c,
			 typename Kokkos::Impl::enable_if<dim_==2, HydroState>::type& q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    
  } // computePrimitive

  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const typename Kokkos::Impl::enable_if<dim_==3, HydroState>::type& u,
			 real_t* c,
			 typename Kokkos::Impl::enable_if<dim_==3, HydroState>::type& q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy, uz;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    uz = u[IW] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    q[IW] = uz;
    
  } // computePrimitive

  /**
   * This routine used SDM_Geometry information to perform interpolation at flux 
   * points using values located at solution points.
   *
   * \param[in] solution_values is a static array containings N values (solution pts)
   * \param[in] index is an integer in 0,1,..,N that identifies a flux point 
   *
   * \return the interpolated value at a given flux point
   */
  KOKKOS_INLINE_FUNCTION
  real_t sol2flux(const solution_values_t& solution_values,
		  int index) const
  {

    // compute interpolated value
    real_t val=0;
    
    for (int k=0; k<N; ++k) {
      val += solution_values[k] * sdm_geom.sol2flux(k,index);
    }

    return val;
    
  } // sol2flux
  
  /**
   * This routine used SDM_Geometry information to perform interpolation at flux 
   * points using values located at solution points.
   *
   * \param[in]  solution_values is a static array containings N values (solution pts)
   * \param[out] flux_values are interpolated values computed at all flux points
   *
   */
  KOKKOS_INLINE_FUNCTION
  void sol2flux_vector(const solution_values_t& solution_values,
		       flux_values_t& flux_values) const
  {

    // compute interpolated values
    for (int j=0; j<N+1; ++j) {
      real_t val=0;
      
      for (int k=0; k<N; ++k) {
	val += solution_values[k] * sdm_geom.sol2flux(k,j);
      }

      flux_values[j] = val;
    }
    
  } // sol2flux_vector

  /**
   * This routine used SDM_Geometry information to perform interpolation at flux 
   * points using values located at solution points.
   *
   * \param[in] flux_values is a static array containings N+1 values (flux pts)
   * \param[in] index is an integer in 0,1,..,N-1 that identifies a solution point 
   *
   * \return the interpolated value at a given solution point
   */
  KOKKOS_INLINE_FUNCTION
  real_t flux2sol(const flux_values_t& flux_values,
		  int index) const
  {

    // compute interpolated value
    real_t val=0;
    
    for (int k=0; k<N+1; ++k) {
      val += flux_values[k] * sdm_geom.flux2sol(k,index);
    }

    return val;
    
  } // flux2sol
  
  /**
   * This routine used SDM_Geometry information to perform interpolation at flux 
   * points using values located at solution points.
   *
   * \param[in]  flux_values is a static array containings N+1 values (flux pts)
   * \param[out] solution_values are interpolated values computed at all solution pts.
   *
   */
  KOKKOS_INLINE_FUNCTION
  void flux2sol_vector(const flux_values_t& flux_values,
		       solution_values_t& solution_values) const
  {

    // compute interpolated values
    for (int j=0; j<N; ++j) {
      real_t val=0;
      
      for (int k=0; k<N+1; ++k) {
	val += flux_values[k] * sdm_geom.flux2sol(k,j);
      }

      solution_values[j] = val;
    }
    
  } // flux2sol_vector
  
}; // class SDMBaseFunctor

} // namespace sdm

#endif // SDM_BASE_FUNCTORS_H_
