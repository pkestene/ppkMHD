#ifndef MOOD_FUNCTORS_H_
#define MOOD_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/Polynomial.h"
#include "mood/MoodBaseFunctor.h"

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
  using typename MoodBaseFunctor<dim,order>::DataArray;

  
  /**
   * Constructor for 2D/3D.
   */
  ComputeFluxesFunctor(HydroParams params,
		       DataArray Udata,
		       DataArray FluxData_x,
		       DataArray FluxData_y,
		       DataArray FluxData_z) :
    MoodBaseFunctor<dim,order>(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z)    
  {};

  ~ComputeFluxesFunctor() {};

  //! functor for 2d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& i)  const
  {
    //printf("2\n");
  }

  //! functor for 3d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& i) const
  {
    //printf("3\n");
  }
  
  DataArray Udata;
  DataArray FluxData_x, FluxData_y, FluxData_z;

  //Stencil          stencil;
  //mood_matrix_pi_t mood_matrix_pi;

}; // class ComputeFluxesFunctor

} // namespace mood

#endif // MOOD_FUNCTORS_H_
