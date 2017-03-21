#ifndef MOOD_FUNCTORS_H_
#define MOOD_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/Polynomial.h"

namespace mood {

/**
 * Compute MOOD fluxes.
 * 
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 */
template<unsigned int dim, unsigned int order>
class ComputeFluxesFunctor : public MoodBaseFunctor<dim,order>
{
    
public:

  ComputeFluxesFunctor(HydroParams params,
		       DataArray Udata,
		       DataArray FluxData_x,
		       DataArray FluxData_y,
		       DataArray FluxData_z,
		       ) :
    PolynomialEvaluator<dim,order>(),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z)    
  {};
  ~ComputeFluxesFunctor() {};

  //template<typename std::enable_if<dim==2>::type
  //KOKKOS_INLINE_FUNCTION
  
  DataArray Udata;
  DataArray FluxData_x, FluxData_y, FluxData_z;
  
}; // class ComputeFluxesFunctor

} // namespace mood

#endif // MOOD_FUNCTORS_H_
