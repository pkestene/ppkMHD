#ifndef MOOD_BASE_FUNCTORS_H_
#define MOOD_BASE_FUNCTORS_H_

#include <type_traits>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace mood {

/**
 * MOOD base functor, this is not a functor, but a base class to derive an actual
 * Kokkos functor.
 */
template<int dim, int degree>
class MoodBaseFunctor : public PolynomialEvaluator<dim,degree>
{
  
public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim==2,HydroState2d,HydroState3d>::type;

  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
  
  MoodBaseFunctor(HydroParams params) :
    PolynomialEvaluator<dim,degree>(),
    params(params) {};
  virtual ~MoodBaseFunctor() {};

  HydroParams params;
  const int nbvar = params.nbvar;

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

}; // class MoodBaseFunctor

} // namespace mood

#endif // MOOD_BASE_FUNCTORS_H_
