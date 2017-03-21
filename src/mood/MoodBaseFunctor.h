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
template<unsigned int dim, unsigned int order>
class MoodBaseFunctor : public PolynomialEvaluator<dim,order>
{
  
public:
  //! Decide at compile-time which HydroState to use
  using HydroState = typename std::conditional<dim==2,HydroState2d,HydroState3d>::type;

  //! Decide at compile-time which data array to use
  using DataArray  = typename std::conditional<dim==2,DataArray2d,DataArray3d>::type;
  
  MoodBaseFunctor(HydroParams params) :
    PolynomialEvaluator<dim,order>(),
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

}; // class MoodBaseFunctor

} // namespace mood

#endif // MOOD_BASE_FUNCTORS_H_
