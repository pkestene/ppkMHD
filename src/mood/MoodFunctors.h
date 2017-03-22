#ifndef MOOD_FUNCTORS_H_
#define MOOD_FUNCTORS_H_

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

#include "mood/Polynomial.h"
#include "mood/MoodBaseFunctor.h"
#include "mood/mood_shared.h"

namespace mood {

/**
 * Compute MOOD fluxes.
 * 
 * Please note:
 * - DataArray and HydroState are typedef'ed in MoodBaseFunctor
 * - FluxData_z may or may not be allocated (depending dim==2 or 3).
 *
 * stencilId must be known at compile time, so that stencilSize is too.
 */
template<unsigned int dim,
	 unsigned int order,
	 STENCIL_ID stencilId>
class ComputeFluxesFunctor : public MoodBaseFunctor<dim,order>
{
    
public:
  using typename MoodBaseFunctor<dim,order>::DataArray;
  using typename PolynomialEvaluator<dim,order>::coefs_t;
  
  /**
   * Constructor for 2D/3D.
   */
  ComputeFluxesFunctor(DataArray        Udata,
		       DataArray        FluxData_x,
		       DataArray        FluxData_y,
		       DataArray        FluxData_z,
		       HydroParams      params,
		       Stencil          stencil,
		       mood_matrix_pi_t mat_pi) :
    MoodBaseFunctor<dim,order>(params),
    Udata(Udata),
    FluxData_x(FluxData_x),
    FluxData_y(FluxData_y),
    FluxData_z(FluxData_z),
    stencil(stencil),
    mat_pi(mat_pi)
  {};

  ~ComputeFluxesFunctor() {};

  //! functor for 2d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& index)  const
  {

    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ghostWidth = this->params.ghostWidth;

    const real_t dx = this->params.dx;
    const real_t dy = this->params.dy;
    
    int i,j;
    index2coord(index,i,j,isize,jsize);

    //! rhs is sized upon stencil, just remove central point
    Kokkos::Array<real_t,stencil_size-1> rhs;

    //! rhs for neighbor cell (accross an x-face, y-face or z-face)
    Kokkos::Array<real_t,stencil_size-1> rhs_n;
    
    
    if(j >= ghostWidth && j < jsize-ghostWidth+1  &&
       i >= ghostWidth && i < isize-ghostWidth+1 ) {

      // retrieve neighbors data for ID, and build rhs
      int irhs = 0;
      for (int is=0; is<stencil_size; ++is) {
	int x = stencil.offsets(is,0);
	int y = stencil.offsets(is,1);
	if (x != 0 or y != 0) {
	  rhs[irhs] = Udata(i+x,j+y,ID) - Udata(i,j,ID);
	  irhs++;
	  // if (i==2 and j==2)
	  //   printf("x y rhs %d %d %f\n",x,y,rhs[irhs-1]+Udata(i,j,ID));
	}	
      } // end for is

      // retrieve reconstruction polynomial coefficients in current cell
      coefs_t coefs_c;
      coefs_c[0] = Udata(i,j,ID);
      for (int icoef=0; icoef<mat_pi.dimension_0(); ++icoef) {
	real_t tmp = 0;
	for (int k=0; k<mat_pi.dimension_1(); ++k) {
	  tmp += mat_pi(icoef,k) * rhs[k];
	}
	coefs_c[icoef+1] = tmp;
      }

      // reconstruct Udata on the left face along X direction
      FluxData_x(i,j,ID) = this->eval(-0.5*dx, 0.0   ,coefs_c);
      FluxData_y(i,j,ID) = this->eval( 0.0   ,-0.5*dy,coefs_c);

      if (i==4 and j==4) {
	for (int icoef=0; icoef<mat_pi.dimension_0()+1; ++icoef)
	  printf("coef e0 e1 %f %d %d\n",
		 coefs_c[icoef],
		 this->monomialMap.data(icoef,0),
		 this->monomialMap.data(icoef,1));
      }
    } // end if
    
  } // end functor 2d

  //! functor for 3d 
  template<unsigned int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& index) const
  {
    const int isize = this->params.isize;
    const int jsize = this->params.jsize;
    const int ksize = this->params.ksize;
    const int ghostWidth = this->params.ghostWidth;

    int i,j,k;
    index2coord(index,i,j,k,isize,jsize,ksize);

    if(k >= ghostWidth && k < ksize - ghostWidth &&
       j >= ghostWidth && j < jsize - ghostWidth &&
       i >= ghostWidth && i < isize - ghostWidth) {
    }
    
  }  // end functor 3d
  
  DataArray        Udata;
  DataArray        FluxData_x, FluxData_y, FluxData_z;

  Stencil          stencil;
  mood_matrix_pi_t mat_pi;

  static const int stencil_size = STENCIL_SIZE[stencilId];
  
}; // class ComputeFluxesFunctor

} // namespace mood

#endif // MOOD_FUNCTORS_H_
